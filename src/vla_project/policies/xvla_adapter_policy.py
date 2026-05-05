"""Concrete runtime policy for X-VLA-Adapter.

Wraps a trained VLAPolicy + tokenizer + image transform + Q99Stats and
exposes ``select_action(obs)`` that:

  1. Preprocess scene/wrist images via SiglipImageTransform.
  2. Tokenize the language string.
  3. Build a one-batch internal Batch dict with dummy target_action /
     action_mask (the model's forward signature requires them; loss is
     ignored at inference).
  4. Run model forward (under torch.no_grad / eval()).
  5. Native mode: denormalize the predicted 7-dim chunk via Q99Stats.
     EE6D mode: convert each predicted 20-dim anchor to a LIBERO-native
     7-dim delta-EE action via :func:`action20_to_ee_delta` using the
     observation's current proprio as the reference frame; the gripper
     qpos is mapped to a [-1, 1] command via ``stats.q01/q99[6]``.
  6. Push all H_act actions into an internal buffer.
  7. Pop and return one action per call.

``last_action_chunk`` is the previously-emitted normalized chunk; we keep
it in normalized space so it matches what the head was trained against
(Plan 1 yields zeros for cold-start). This wrapper carries a copy of the
last normalized prediction for use as the next ``last_action_chunk``.
For EE6D mode, last_action_chunk is the previously-emitted EE6D anchor
chunk (raw, since EE6D is not Q99-normalized).

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
    load_q99_proprio_stats,
    load_q99_stats,
    normalize_proprio_q99,
    q99_stats_from_block,
)
from vla_project.data.transforms.action_alignment import action20_to_ee_delta
from vla_project.data.transforms.image import DINOv2ImageTransform, SiglipImageTransform
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
        action_format: str = "native",
        proprio_stats: Optional[Q99Stats] = None,
    ) -> None:
        if action_format not in ("native", "ee6d"):
            raise ValueError(
                f"action_format must be 'native' or 'ee6d'; got {action_format!r}"
            )
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
        self.dinov2_image_transform = DINOv2ImageTransform(size=C.SIGLIP_IMAGE_SIZE)
        self.norm_stats = norm_stats
        # Proprio normalization (BOUNDS_Q99). When provided, applied at
        # ``_build_batch`` so the model sees the same normalized proprio
        # the dataset fed during training. Mirrors vla-gemma-4
        # ``vla_evaluation.normalize_proprio`` (RLDS-auto-normalized at train
        # time + explicit normalize at eval time).
        self.proprio_stats = proprio_stats
        self.action_chunk_len = action_chunk_len
        self.domain_id = int(domain_id)
        self.compile_mode = compile_mode
        self.action_format = action_format
        # Internal buffer is the model's *raw output* dim:
        #   native: 7-dim normalized actions (denormalized at pop)
        #   ee6d: 20-dim anchors (converted to 7-dim delta at pop)
        # last_action_chunk is fed back into the next forward unchanged.
        action_dim_internal = 20 if action_format == "ee6d" else C.ACTION_DIM
        self._buffer: Deque[np.ndarray] = deque()
        self._last_chunk_norm: torch.Tensor = torch.zeros(
            action_chunk_len, action_dim_internal, dtype=torch.float32
        )
        # EE6D-only: gripper qpos→command mapping. The model predicts gripper
        # *qpos* (from proprio) but LIBERO's OSC_POSE controller wants a
        # *command* in [-1, +1]. norm_stats.q01/q99[6] is the action gripper
        # (already a command), NOT proprio qpos — different ranges. Hardcode
        # the Franka Panda gripper qpos extents (≈ 0 closed, 0.04 open). This
        # is a constant of the robot, not a dataset stat.
        self._gripper_qpos_closed = 0.0
        self._gripper_qpos_open = 0.04

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
        action_format: str = "native",
    ) -> "XVLAAdapterPolicy":
        meta = load_checkpoint(ckpt_dir, model)
        ns = meta.get("norm_stats")
        if ns is None or unnorm_key not in ns:
            raise KeyError(
                f"checkpoint at {ckpt_dir} has no norm_stats[{unnorm_key!r}]; "
                f"available: {list((ns or {}).keys())}"
            )
        per_dataset = ns[unnorm_key]
        stats = q99_stats_from_block(per_dataset["action"])
        proprio_stats = (
            q99_stats_from_block(per_dataset["proprio"])
            if "proprio" in per_dataset
            else None
        )
        return cls(
            model=model,
            tokenizer=tokenizer,
            image_transform=image_transform,
            norm_stats=stats,
            action_chunk_len=action_chunk_len,
            compile_mode=compile_mode,
            domain_id=domain_id,
            action_format=action_format,
            proprio_stats=proprio_stats,
        )

    def reset(self) -> None:
        self._buffer.clear()
        self._last_chunk_norm.zero_()

    def _np_image_to_chw(self, img: np.ndarray, *, transform: Optional[torch.nn.Module] = None) -> torch.Tensor:
        if img.dtype != np.uint8 or img.ndim != 3 or img.shape[-1] != 3:
            raise ValueError(
                f"image must be uint8 (H, W, 3); got dtype={img.dtype} shape={img.shape}"
            )
        t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        # Train-eval FOV match: RLDS train pipeline resizes images to 224×224
        # BEFORE the SigLIP transform (Resize 248 + CC 224 — effectively an
        # upscale-then-crop). LIBERO sim renders at 256×256, so we must also
        # pre-resize 256→224 here, otherwise the SigLIP transform downscales
        # 256→248 + crops to 224 = a different field of view than training.
        # Bicubic+antialias matches the RLDS lanczos-like resize closely
        # enough; the residual error is dominated by the SigLIP backbone's
        # low-frequency tolerance.
        target = C.SIGLIP_IMAGE_SIZE  # 224
        if t.shape[1] != target or t.shape[2] != target:
            import torch.nn.functional as F
            t = F.interpolate(
                t.unsqueeze(0), size=(target, target),
                mode="bicubic", antialias=True,
            ).squeeze(0).clamp(0.0, 1.0)
        tx = self.image_transform if transform is None else transform
        return tx(t)

    def _qpos_to_gripper_cmd(self, qpos: float) -> float:
        """Map gripper qpos in [closed, open] → command in [-1, +1], clipped.

        LIBERO OSC_POSE gripper convention: -1 commands close, +1 opens.
        Linear map between the Franka Panda's qpos extents, with hard
        clipping to [-1, +1] so an out-of-range prediction (e.g. early in
        training before the head is converged) doesn't blow past the
        controller's expected range."""
        span = self._gripper_qpos_open - self._gripper_qpos_closed
        if span <= 0:
            return 0.0
        u = (qpos - self._gripper_qpos_closed) / span
        return float(max(-1.0, min(1.0, 2.0 * u - 1.0)))

    def _build_batch(self, obs: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        # Match Trainer's per-batch dtype/device handling: float tensors get
        # cast to the model's parameter dtype (e.g. bf16); long/bool tensors
        # keep their integer dtypes.
        first_param = next(self.model.parameters())
        device = first_param.device
        model_dtype = first_param.dtype
        scene = self._np_image_to_chw(obs["scene_image"]).unsqueeze(0).to(device).to(model_dtype)
        wrist = self._np_image_to_chw(obs["wrist_image"]).unsqueeze(0).to(device).to(model_dtype)
        proprio = torch.from_numpy(np.asarray(obs["proprio"], dtype=np.float32))
        if self.proprio_stats is not None:
            proprio = normalize_proprio_q99(proprio, self.proprio_stats)
        proprio = proprio.unsqueeze(0).to(device).to(model_dtype)
        prompt = self.tokenizer(obs["language"])
        action_dim_internal = 20 if self.action_format == "ee6d" else C.ACTION_DIM
        batch = {
            "domain_id": torch.tensor([self.domain_id], dtype=torch.long, device=device),
            "scene_image": scene,
            "wrist_image": wrist,
            "prompt_input_ids": prompt["input_ids"].unsqueeze(0).to(device),
            "prompt_attention_mask": prompt["attention_mask"].unsqueeze(0).to(device),
            "proprio": proprio,
            "last_action_chunk": self._last_chunk_norm.unsqueeze(0).to(device).to(model_dtype),
            "target_action": torch.zeros(1, self.action_chunk_len, action_dim_internal, device=device, dtype=model_dtype),
            "action_mask": torch.ones(1, self.action_chunk_len, dtype=torch.bool, device=device),
            # v36 wrist_in_llm contract: wrist is always present in LIBERO sim,
            # so mark mask=True. forward()'s default fallback also yields True
            # when missing, but explicit-belt-and-suspenders here.
            "wrist_mask": torch.ones(1, dtype=torch.bool, device=device),
        }
        if getattr(self.model, "wrist_dinov2_encoder", None) is not None:
            wrist_dino = self._np_image_to_chw(
                obs["wrist_image"], transform=self.dinov2_image_transform
            )
            batch["wrist_image_dinov2"] = wrist_dino.unsqueeze(0).to(device).to(model_dtype)
            if getattr(self.model.cfg, "use_scene_wrist_dinov2_llm", False):
                scene_dino = self._np_image_to_chw(
                    obs["scene_image"], transform=self.dinov2_image_transform
                )
                batch["scene_image_dinov2"] = scene_dino.unsqueeze(0).to(device).to(model_dtype)
        return batch

    def _refill_buffer(self, obs: Dict[str, Any]) -> None:
        was_training = self.model.training
        self.model.eval()
        try:
            with torch.no_grad():
                batch = self._build_batch(obs)
                pred, _ = self.model(batch)
            pred_cpu = pred.detach().to(torch.float32).cpu()
            self._last_chunk_norm = pred_cpu[0].clone()

            if self.action_format == "native":
                denormed = denormalize_action_q99(pred_cpu[0], self.norm_stats)
                # RLDS-trained models output gripper in the RLDS convention
                # ([0=close, 1=open], mask=False so it passes through Q99).
                # LIBERO env expects [-1=open, +1=close]. Apply
                # normalize_gripper_action ([0,1]→[-1,+1]) + invert
                # (sign flip), matching upstream
                # ``run_libero_eval.process_action`` for openvla model_family.
                # vla-gemma-4 ``eval_libero_gemma4.py:248-264`` does the same.
                denormed_np = denormed.numpy().astype(np.float32)
                # gripper transform: 2x-1 then binarize sign, then invert.
                g = denormed_np[:, -1]
                g_norm = 2.0 * g - 1.0
                g_bin = np.sign(g_norm)
                # zero-handling: sign(0)=0 — round to closed (-1) like baseline
                g_bin = np.where(g_bin == 0, -1.0, g_bin)
                denormed_np[:, -1] = -g_bin   # invert
                for i in range(self.action_chunk_len):
                    self._buffer.append(denormed_np[i])
                return

            # EE6D: convert each (abs xyz, rot6d, gripper qpos) anchor to a
            # 7-dim delta-EE action relative to the *current* proprio. The
            # current proprio stays fixed for the whole emitted chunk; the
            # rollout loop will re-call us before consuming further anchors.
            current_proprio = torch.from_numpy(
                np.asarray(obs["proprio"], dtype=np.float32)
            )  # (8,)
            current_proprio_b = current_proprio.unsqueeze(0).expand(
                self.action_chunk_len, -1
            )
            delta = action20_to_ee_delta(pred_cpu[0], current_proprio_b)  # (T, 7)
            delta_np = delta.numpy().astype(np.float32)
            # Replace gripper qpos with OSC_POSE command [-1, +1].
            for i in range(self.action_chunk_len):
                delta_np[i, 6] = self._qpos_to_gripper_cmd(float(delta_np[i, 6]))
                self._buffer.append(delta_np[i])
        finally:
            if was_training:
                self.model.train()

    def select_action(self, obs: Dict[str, Any]) -> np.ndarray:
        if not self._buffer:
            self._refill_buffer(obs)
        return self._buffer.popleft()
