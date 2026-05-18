"""Run vla-gemma-4 73% baseline ckpt through OUR scripts/eval.py harness.

Validates whether our `LIBEROSimRobot`, `evaluate_libero`, and rollout code
behave equivalently to vla-gemma-4's `eval_libero_gemma4.py`. If the
baseline ckpt evaluates to ~baseline accuracy here, the eval pipeline is
correct and v23's 0/50 result points to a training-side or
arch-mismatch issue (e.g. our `scene_proj` capacity). If it evaluates to
0/50, our eval pipeline itself has a bug.

Run via the vla-gemma-4 venv (transformers 5.5.4 required):

    PYTHONPATH=/misc/dl00/takaki/GEM-4-VLA/src:\\
/misc/dl00/takaki/vla-gemma-4/VLA-Adapter:\\
/misc/dl00/takaki/vla-gemma-4 \\
    /misc/dl00/takaki/vla-gemma-4/.venv-gemma4/bin/python \\
        tools/eval_baseline_shim.py <config.yaml>

Config schema: see configs/eval/libero_baseline_shim_step10000.yaml
"""
from __future__ import annotations

import json
import os
import sys
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional

import numpy as np
import torch
from omegaconf import OmegaConf

from prismatic.extern.hf.modeling_prismatic_gemma4 import VLAAdapterGemma4
from prismatic.vla.constants_gemma4 import (
    ACTION_TOKEN_BEGIN_IDX,
    NUM_ACTION_TOKENS,
    NUM_VISION_TOKENS,
    PROPRIO_PLACEHOLDER_IDX,
    VISION_PLACEHOLDER_BEGIN_IDX,
)
from transformers import AutoTokenizer, Gemma4ForConditionalGeneration

from vla_project.evaluation.libero_eval import evaluate_libero
from vla_project.policies.base_policy import BasePolicy
from vla_project.robots.sim_robot import LIBEROSimRobot
from vla_project.utils.seed import set_seed

PROMPT_MAX_LEN = 20
NUM_ACTIONS_CHUNK = 8
ACTION_DIM = 7
PROPRIO_DIM = 8
POLICY_IMAGE_SIZE = 224


def _load_action_proprio_stats(stats_path: Path, unnorm_key: str):
    all_stats = json.loads(Path(stats_path).read_text())
    per_dataset = all_stats[unnorm_key]
    a, p = per_dataset["action"], per_dataset["proprio"]
    action_stats = {
        "q01": np.array(a["q01"], dtype=np.float32),
        "q99": np.array(a["q99"], dtype=np.float32),
        "mask": np.array(a.get("mask", [True] * len(a["q01"])), dtype=bool),
    }
    proprio_stats = {
        "q01": np.array(p["q01"], dtype=np.float32),
        "q99": np.array(p["q99"], dtype=np.float32),
        "mask": np.array(p.get("mask", [True] * len(p["q01"])), dtype=bool),
    }
    return action_stats, proprio_stats


def _normalize_proprio_q99(raw: np.ndarray, stats: dict) -> np.ndarray:
    q01, q99, mask = stats["q01"], stats["q99"], stats["mask"]
    out = raw.astype(np.float32).copy()
    span = np.where(q99 - q01 > 0, q99 - q01, 1.0)
    norm = 2.0 * (out - q01) / span - 1.0
    norm = np.clip(norm, -1.0, 1.0)
    return np.where(mask, norm, out).astype(np.float32)


def _denormalize_action_q99(action_norm: np.ndarray, stats: dict) -> np.ndarray:
    q01, q99, mask = stats["q01"], stats["q99"], stats["mask"]
    half_range = 0.5 * (q99 - q01)
    mid = 0.5 * (q01 + q99)
    out = action_norm.astype(np.float32) * half_range + mid
    return np.where(mask, out, action_norm).astype(np.float32)


def _resize_lanczos3(img: np.ndarray, size: int = POLICY_IMAGE_SIZE) -> np.ndarray:
    """Match vla-gemma-4 `resize_image_for_policy` (TF lanczos3)."""
    import tensorflow as tf

    encoded = tf.image.encode_jpeg(img)
    decoded = tf.io.decode_image(encoded, expand_animations=False, dtype=tf.uint8)
    resized = tf.image.resize(decoded, (size, size), method="lanczos3", antialias=True)
    clipped = tf.cast(tf.clip_by_value(tf.round(resized), 0, 255), tf.uint8)
    return clipped.numpy()


def _build_input_ids(
    language: str,
    tokenizer,
    prompt_max_len: int = PROMPT_MAX_LEN,
    *,
    prompt_first: bool = True,
) -> torch.Tensor:
    """Mirror vla-gemma-4 `build_input_ids`.

    ``prompt_first=True`` reproduces the pre-e033e64 layout
    ``[BOS]+prompt+V+propriop+A+EOS`` that the wristb_v2 ckpt was trained
    against (equivalent to baseline's ``VLA_OLD_PROMPT_FIRST=1``). Required
    for that ckpt: without it, RoPE positions shift and language is
    effectively ignored, dropping eval from 73% → 14%.

    ``prompt_first=False`` is the post-e033e64 layout
    ``[BOS]+V+prompt+propriop+A+EOS``.
    """
    text = f"What action should the robot take to {language.lower().strip()}?"
    ids: List[int] = tokenizer(text, add_special_tokens=False).input_ids
    if len(ids) > prompt_max_len:
        ids = ids[:prompt_max_len]
    else:
        ids = ids + [tokenizer.pad_token_id] * (prompt_max_len - len(ids))
    if prompt_first:
        full = (
            [tokenizer.bos_token_id]
            + ids
            + list(range(VISION_PLACEHOLDER_BEGIN_IDX, VISION_PLACEHOLDER_BEGIN_IDX + NUM_VISION_TOKENS))
            + [PROPRIO_PLACEHOLDER_IDX]
            + list(range(ACTION_TOKEN_BEGIN_IDX, ACTION_TOKEN_BEGIN_IDX + NUM_ACTION_TOKENS))
            + [tokenizer.eos_token_id]
        )
    else:
        full = (
            [tokenizer.bos_token_id]
            + list(range(VISION_PLACEHOLDER_BEGIN_IDX, VISION_PLACEHOLDER_BEGIN_IDX + NUM_VISION_TOKENS))
            + ids
            + [PROPRIO_PLACEHOLDER_IDX]
            + list(range(ACTION_TOKEN_BEGIN_IDX, ACTION_TOKEN_BEGIN_IDX + NUM_ACTION_TOKENS))
            + [tokenizer.eos_token_id]
        )
    return torch.tensor(full, dtype=torch.long)


class BaselineShimPolicy(BasePolicy):
    """Wrap a `VLAAdapterGemma4` instance to satisfy our `BasePolicy` interface.

    Observation contract (from our `LIBEROSimRobot._wrap_obs`):
      - scene_image: (H, W, 3) uint8, already 180-rotated, sized at robot.image_size
      - wrist_image: (H, W, 3) uint8, already 180-rotated
      - proprio: (8,) float32 raw (eef_pos[3] + axis_angle[3] + gripper_qpos[2])
      - language: str
    """

    def __init__(
        self,
        model: VLAAdapterGemma4,
        tokenizer,
        action_stats: dict,
        proprio_stats: dict,
        device: torch.device,
        action_chunk_len: int = NUM_ACTIONS_CHUNK,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.action_stats = action_stats
        self.proprio_stats = proprio_stats
        self.device = device
        self.action_chunk_len = action_chunk_len
        self._buffer: Deque[np.ndarray] = deque()

    def reset(self) -> None:
        self._buffer.clear()

    def _refill(self, obs: Dict[str, Any]) -> None:
        scene_resized = _resize_lanczos3(obs["scene_image"], POLICY_IMAGE_SIZE)
        wrist_resized = _resize_lanczos3(obs["wrist_image"], POLICY_IMAGE_SIZE)

        scene_t = (
            torch.from_numpy(scene_resized).permute(2, 0, 1).float().unsqueeze(0)
        )  # (1, 3, H, W) CPU float [0,255]

        import torch.nn.functional as F

        w = (
            torch.from_numpy(wrist_resized).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        )
        w = F.interpolate(w, size=(224, 224), mode="bilinear", antialias=True)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        w = (w - mean) / std
        wrist_t = w.to(self.device, dtype=torch.bfloat16)

        pv = {"scene": scene_t, "wrist": wrist_t}
        input_ids = _build_input_ids(obs["language"], self.tokenizer).unsqueeze(0).to(self.device)
        proprio_norm = _normalize_proprio_q99(np.asarray(obs["proprio"], dtype=np.float32), self.proprio_stats)
        proprio = torch.tensor(proprio_norm, dtype=torch.bfloat16).unsqueeze(0).to(self.device)

        was_training = self.model.training
        self.model.eval()
        with torch.no_grad():
            predicted = self.model(pv, input_ids, proprio, actions=None)
        if was_training:
            self.model.train()

        action_norm = predicted[0].detach().float().cpu().numpy()  # (8, 7)
        action = _denormalize_action_q99(action_norm, self.action_stats)

        # Gripper: [0,1] → 2x-1 → sign → invert (matches baseline LIBERO eval)
        out = action.copy()
        out[..., -1] = 2.0 * out[..., -1] - 1.0
        out[..., -1] = np.sign(out[..., -1])
        out[..., -1] = np.where(out[..., -1] == 0, -1.0, out[..., -1])
        out[..., -1] = -out[..., -1]

        for i in range(self.action_chunk_len):
            self._buffer.append(out[i])

    def select_action(self, obs: Dict[str, Any]) -> np.ndarray:
        if not self._buffer:
            self._refill(obs)
        return self._buffer.popleft()


def _build_baseline_model(cfg, device: torch.device):
    gemma_id = cfg.model.gemma_model_id
    tok = AutoTokenizer.from_pretrained(gemma_id)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    gemma = Gemma4ForConditionalGeneration.from_pretrained(
        gemma_id, dtype=torch.bfloat16, attn_implementation="sdpa",
    ).to(device).eval()
    gemma.config.use_cache = True
    for p in gemma.parameters():
        p.requires_grad = False

    model = VLAAdapterGemma4(
        gemma_model=gemma,
        max_soft_tokens=cfg.model.get("max_soft_tokens", 280),
        feature_norm=torch.nn.Identity(),
        proprio_dim=PROPRIO_DIM,
        action_dim=ACTION_DIM,
        num_action_chunks=NUM_ACTIONS_CHUNK,
        num_pretrain_datasets=0,
        num_soft_prompt_tokens=32,
        training_mode=cfg.model.get("training_mode", "speed"),
        vision_backbone_type=cfg.model.get("vision_backbone_type", "siglip"),
        siglip_use_tensor_transform=cfg.model.get("siglip_use_tensor_transform", True),
        use_xvla_style=cfg.model.get("use_xvla_style", False),
        use_wrist_bridge=cfg.model.get("use_wrist_bridge", True),
        use_proper_ffn=cfg.model.get("use_proper_ffn", False),
        wrist_bridge_layer_mode=cfg.model.get("wrist_bridge_layer_mode", "per_layer"),
        num_action_head_blocks=cfg.model.get("num_action_head_blocks", 24),
    )
    model = model.to(device, dtype=torch.bfloat16).eval()
    return model, tok


def _load_ckpt(model: VLAAdapterGemma4, ckpt_path: Path, device: torch.device) -> dict:
    payload = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    state = payload["trainable_state_dict"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    relevant_missing = [
        k for k in missing
        if not k.startswith("vision_backbone.")
        and not k.startswith("llm.")
        and "soft_prompt_library" not in k
    ]
    if relevant_missing:
        print(f"[baseline_shim] WARN missing (non-frozen): {relevant_missing[:8]} total={len(relevant_missing)}")
    if unexpected:
        print(f"[baseline_shim] WARN unexpected ckpt keys: {unexpected[:3]} total={len(unexpected)}")
    print(
        f"[baseline_shim] loaded ckpt step={payload.get('gradient_step_idx')!r} "
        f"lr={payload.get('current_lr')!r}"
    )
    return payload


def main(cfg_path: str) -> None:
    cfg = OmegaConf.load(cfg_path)
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[baseline_shim] device={device}")

    model, tok = _build_baseline_model(cfg, device)
    _load_ckpt(model, Path(cfg.checkpoint.path), device)

    action_stats, proprio_stats = _load_action_proprio_stats(
        Path(cfg.data.stats_path), cfg.data.unnorm_key
    )

    policy = BaselineShimPolicy(
        model=model,
        tokenizer=tok,
        action_stats=action_stats,
        proprio_stats=proprio_stats,
        device=device,
    )

    def _make_robot(task_idx: int) -> LIBEROSimRobot:
        return LIBEROSimRobot(
            bddl_path_root=cfg.robot.bddl_path_root,
            task_suite=cfg.robot.task_suite,
            task_idx=task_idx,
            image_size=cfg.robot.image_size,
            seed=cfg.seed,
            libero_path=cfg.robot.libero_path,
        )

    metrics = evaluate_libero(
        policy=policy,
        robot_factory=_make_robot,
        task_idxs=list(cfg.eval.task_idxs),
        num_episodes_per_task=int(cfg.eval.num_episodes_per_task),
        max_steps=int(cfg.eval.max_steps),
        num_steps_wait=int(cfg.eval.num_steps_wait),
        video_dir=cfg.eval.get("video_dir", None),
        video_fps=int(cfg.eval.get("video_fps", 10)),
        video_ext=str(cfg.eval.get("video_ext", "gif")),
    )
    summary = {"overall": metrics["overall"], "per_task": metrics["per_task"]}
    print(f"[baseline_shim] metrics={json.dumps(summary, indent=2)}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <config.yaml>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])
