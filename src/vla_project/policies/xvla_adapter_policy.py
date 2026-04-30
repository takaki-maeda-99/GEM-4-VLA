"""Concrete runtime policy for X-VLA-Adapter.

Wraps a trained VLAPolicy + tokenizer + image transform + Q99Stats and
exposes ``select_action(obs)`` that:

  1. Preprocess scene/wrist images via SiglipImageTransform.
  2. Tokenize the language string.
  3. Build a one-batch internal Batch dict with dummy target_action /
     action_mask (the model's forward signature requires them; loss is
     ignored at inference).
  4. Run model forward (under torch.no_grad / eval()).
  5. Denormalize the predicted chunk via Q99Stats.
  6. Push all H_act actions into an internal buffer.
  7. Pop and return one action per call.

``last_action_chunk`` is the previously-emitted normalized chunk; we keep
it in normalized space so it matches what the head was trained against
(Plan 1 yields zeros for cold-start). This wrapper carries a copy of the
last normalized prediction for use as the next ``last_action_chunk``.

The buffer is reset via ``reset()`` (called per episode by the rollout loop).
"""
from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, Optional, Union

import numpy as np
import torch

from vla_project.data import constants as C
from vla_project.data.normalization import (
    Q99Stats,
    denormalize_action_q99,
    load_q99_stats,
)
from vla_project.data.transforms.image import SiglipImageTransform
from vla_project.data.transforms.language import GemmaPromptTokenizer
from vla_project.models.vla_policy import VLAPolicy
from vla_project.policies.base_policy import BasePolicy
from vla_project.training.checkpoint import load_checkpoint


class XVLAAdapterPolicy(BasePolicy):
    def __init__(
        self,
        model: VLAPolicy,
        tokenizer: GemmaPromptTokenizer,
        image_transform: SiglipImageTransform,
        norm_stats: Q99Stats,
        *,
        action_chunk_len: int = C.ACTION_CHUNK_LEN,
        domain_id: int = 0,
        compile_mode: str = "off",
    ) -> None:
        self.model = model
        if compile_mode != "off":
            # torch.compile speeds up the chunk-refill forward pass. For
            # inference at bs=1 with stable shapes, mode='reduce-overhead'
            # (CUDA graphs) usually wins; 'default' / 'max-autotune' also
            # valid. fullgraph=False allows graph breaks (Gemma's HF code
            # has Python control flow that can't always trace whole-graph).
            #
            # First call pays a 30-60s JIT compile cost. Multi-episode
            # rollouts amortize this; single-episode smoke runs see net
            # slowdown.
            self.model = torch.compile(self.model, mode=compile_mode, fullgraph=False)
        self.tokenizer = tokenizer
        self.image_transform = image_transform
        self.norm_stats = norm_stats
        self.action_chunk_len = action_chunk_len
        self.domain_id = int(domain_id)
        self.compile_mode = compile_mode
        self._buffer: Deque[np.ndarray] = deque()
        self._last_chunk_norm: torch.Tensor = torch.zeros(
            action_chunk_len, C.ACTION_DIM, dtype=torch.float32
        )

    @classmethod
    def from_checkpoint(
        cls,
        ckpt_dir: Union[str, Path],
        model: VLAPolicy,
        tokenizer: GemmaPromptTokenizer,
        image_transform: SiglipImageTransform,
        unnorm_key: str,
        *,
        action_chunk_len: int = C.ACTION_CHUNK_LEN,
        domain_id: int = 0,
        compile_mode: str = "off",
    ) -> "XVLAAdapterPolicy":
        meta = load_checkpoint(ckpt_dir, model)
        ns = meta.get("norm_stats")
        if ns is None or unnorm_key not in ns:
            raise KeyError(
                f"checkpoint at {ckpt_dir} has no norm_stats[{unnorm_key!r}]; "
                f"available: {list((ns or {}).keys())}"
            )
        a = ns[unnorm_key]["action"]
        stats = Q99Stats(
            q01=torch.tensor(a["q01"], dtype=torch.float32),
            q99=torch.tensor(a["q99"], dtype=torch.float32),
            mask=torch.tensor(a["mask"], dtype=torch.bool),
        )
        return cls(
            model=model,
            tokenizer=tokenizer,
            image_transform=image_transform,
            norm_stats=stats,
            action_chunk_len=action_chunk_len,
            compile_mode=compile_mode,
            domain_id=domain_id,
        )

    def reset(self) -> None:
        self._buffer.clear()
        self._last_chunk_norm.zero_()

    def _np_image_to_chw(self, img: np.ndarray) -> torch.Tensor:
        if img.dtype != np.uint8 or img.ndim != 3 or img.shape[-1] != 3:
            raise ValueError(
                f"image must be uint8 (H, W, 3); got dtype={img.dtype} shape={img.shape}"
            )
        t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        return self.image_transform(t)

    def _build_batch(self, obs: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        # Match Trainer's per-batch dtype/device handling: float tensors get
        # cast to the model's parameter dtype (e.g. bf16); long/bool tensors
        # keep their integer dtypes.
        first_param = next(self.model.parameters())
        device = first_param.device
        model_dtype = first_param.dtype
        scene = self._np_image_to_chw(obs["scene_image"]).unsqueeze(0).to(device).to(model_dtype)
        wrist = self._np_image_to_chw(obs["wrist_image"]).unsqueeze(0).to(device).to(model_dtype)
        proprio = torch.from_numpy(np.asarray(obs["proprio"], dtype=np.float32)).unsqueeze(0).to(device).to(model_dtype)
        prompt = self.tokenizer(obs["language"])
        return {
            "domain_id": torch.tensor([self.domain_id], dtype=torch.long, device=device),
            "scene_image": scene,
            "wrist_image": wrist,
            "prompt_input_ids": prompt["input_ids"].unsqueeze(0).to(device),
            "prompt_attention_mask": prompt["attention_mask"].unsqueeze(0).to(device),
            "proprio": proprio,
            "last_action_chunk": self._last_chunk_norm.unsqueeze(0).to(device).to(model_dtype),
            "target_action": torch.zeros(1, self.action_chunk_len, C.ACTION_DIM, device=device, dtype=model_dtype),
            "action_mask": torch.ones(1, self.action_chunk_len, dtype=torch.bool, device=device),
        }

    def _refill_buffer(self, obs: Dict[str, Any]) -> None:
        was_training = self.model.training
        self.model.eval()
        try:
            with torch.no_grad():
                batch = self._build_batch(obs)
                pred, _ = self.model(batch)
            pred_cpu = pred.detach().to(torch.float32).cpu()
            denormed = denormalize_action_q99(pred_cpu[0], self.norm_stats)
            self._last_chunk_norm = pred_cpu[0].clone()
            for i in range(self.action_chunk_len):
                self._buffer.append(denormed[i].numpy().astype(np.float32))
        finally:
            if was_training:
                self.model.train()

    def select_action(self, obs: Dict[str, Any]) -> np.ndarray:
        if not self._buffer:
            self._refill_buffer(obs)
        return self._buffer.popleft()
