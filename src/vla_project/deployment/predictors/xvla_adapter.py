"""XVLAAdapterChunkPredictor — Phase 1 implementation.

Takes an obs dict produced by ``DomainAdapter.preprocess`` and returns a
single chunk in MODEL NATIVE physical units (i.e. after Q99 denormalize
but BEFORE DomainAdapter.postprocess applies contract gripper / frame
conversion).

obs schema (after DomainAdapter.preprocess):
  - scene_image: np.uint8 [H, W, 3]    (decoded from JPEG)
  - wrist_image: np.uint8 [H, W, 3]
  - wrist_was_provided: bool
  - proprio: np.float32 [proprio_dim]   (Q99-NORMALIZED already)
  - language: str

We delegate the model + tokenizer + image_transform to ModelRuntime
(Phase 1) and build the batch ourselves; this keeps predictor-specific
contract logic (no LIBERO gripper transform, etc.) out of ModelRuntime.
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np
import torch

from vla_project.data import constants as C
from vla_project.data.normalization import (
    Q99Stats,
    denormalize_action_q99,
    q99_stats_from_block,
)
from vla_project.data.transforms.image import SiglipImageTransform
from vla_project.data.transforms.language import GemmaPromptTokenizer
from vla_project.deployment.predictors.base import ChunkPredictor
from vla_project.deployment.runtime import ModelRuntime


class XVLAAdapterChunkPredictor(ChunkPredictor):
    def __init__(
        self,
        runtime: ModelRuntime,
        tokenizer: GemmaPromptTokenizer,
        image_transform: SiglipImageTransform,
        action_q99: Dict[str, Any] | Q99Stats,
        action_chunk_len: int,
        action_dim: int,
        domain_id: int,
    ) -> None:
        self._T = int(action_chunk_len)
        self._A = int(action_dim)
        self.runtime = runtime
        self.tokenizer = tokenizer
        self.image_transform = image_transform
        # Accept either the raw block dict (from meta.json) or an already-
        # constructed Q99Stats. The server passes the dict; tests may pass
        # Q99Stats directly.
        if isinstance(action_q99, Q99Stats):
            self.action_q99 = action_q99
        else:
            self.action_q99 = q99_stats_from_block(action_q99)
        self.domain_id = int(domain_id)
        # batch["last_action_chunk"] is currently IGNORED by VLAPolicy.forward
        # (see vla_policy.py:530-537 — it's still emitted by the dataset for
        # potential reinstatement but x_init=zeros at the action head). We
        # feed zeros to match the training-time `last_action_chunk_mode: zero`
        # convention and to avoid pretending that streaming history is
        # consumed.
        self._zero_last_chunk = torch.zeros(self._T, self._A, dtype=torch.float32)

    @property
    def chunk_len(self) -> int:
        return self._T

    @property
    def action_dim(self) -> int:
        return self._A

    def predict(self, obs: Dict[str, Any]) -> np.ndarray:
        device = self.runtime.device
        model_dtype = self.runtime.dtype

        scene = self._np_image_to_chw(obs["scene_image"]).unsqueeze(0).to(device).to(model_dtype)
        wrist = self._np_image_to_chw(obs["wrist_image"]).unsqueeze(0).to(device).to(model_dtype)
        proprio = torch.from_numpy(np.asarray(obs["proprio"], dtype=np.float32))
        proprio = proprio.unsqueeze(0).to(device).to(model_dtype)
        prompt = self.tokenizer(obs["language"])

        batch = {
            "domain_id": torch.tensor([self.domain_id], dtype=torch.long, device=device),
            "scene_image": scene,
            "wrist_image": wrist,
            "prompt_input_ids": prompt["input_ids"].unsqueeze(0).to(device),
            "prompt_attention_mask": prompt["attention_mask"].unsqueeze(0).to(device),
            "proprio": proprio,
            "last_action_chunk": self._zero_last_chunk.unsqueeze(0).to(device).to(model_dtype),
            "target_action": torch.zeros(
                1, self._T, self._A, device=device, dtype=model_dtype,
            ),
            "action_mask": torch.ones(1, self._T, dtype=torch.bool, device=device),
            # SO101 wrist is always present in our converted dataset; for
            # wrist_in_llm + dropout=0 it's hard-required, so this should
            # always be True at deploy time (request validation gates this).
            "wrist_mask": torch.tensor(
                [bool(obs.get("wrist_was_provided", True))],
                dtype=torch.bool, device=device,
            ),
        }

        pred, _ = self.runtime(batch)
        pred_cpu = pred.detach().to(torch.float32).cpu()[0]  # [T, A]
        denormed = denormalize_action_q99(pred_cpu, self.action_q99)
        return denormed.numpy().astype(np.float32)

    def _np_image_to_chw(self, img: np.ndarray) -> torch.Tensor:
        """Convert uint8 HWC image (any size) → SigLIP-normalized float CHW 224x224."""
        if img.dtype != np.uint8:
            img = img.astype(np.uint8)
        # HWC uint8 -> CHW float in [0, 1]
        t = torch.from_numpy(img).permute(2, 0, 1).to(torch.float32) / 255.0
        target = C.SIGLIP_IMAGE_SIZE
        if t.shape[1] != target or t.shape[2] != target:
            import torch.nn.functional as F
            t = F.interpolate(
                t.unsqueeze(0), size=(target, target),
                mode="bicubic", antialias=True,
            ).squeeze(0).clamp(0.0, 1.0)
        return self.image_transform(t)
