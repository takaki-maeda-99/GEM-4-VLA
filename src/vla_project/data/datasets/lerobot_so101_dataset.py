"""LeRobot-HF based SO101 step-level dataset (single-domain, FT).

Built for ``takaki99/test_so101`` after running
``tools/convert_so101_v3_to_v21.py`` to filter success episodes (62/73)
and write a v2.1-compatible local copy. Reads via ``LeRobotDataset(root=...)``
with ``delta_timestamps`` on the EE columns (action + observation) so each
yielded sample contains an 8-frame chunk of both. Computes EE-delta
actions on the fly:

  action[t] = [
      action.ee_pos[t] - observation.state.ee_pos[t]      (dxyz, 3-dim)
      logmap(R(action.ee_rotvec[t]) @ R(obs.state.ee_rotvec[t])^T)  (3-dim)
      action.gripper_pos[t] / 100                          (1-dim, [0, 1])
  ]
  → 7-dim, padded already to ACTION_DIM=7 (no extra padding needed)

Proprio at t=0:
  [obs.state.ee_pos (3), obs.state.ee_rotvec (3), obs.state.gripper_pos/100 (1), 0]
  → 8-dim, padded to PROPRIO_DIM=8

Q99 normalization uses the per-dataset payload at the ``stats_path`` /
``unnorm_key`` (see ``tools/compute_norm_stats_so101.py``).

Image keys map: scene = ``observation.images.front``,
wrist = ``observation.images.wrist``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union

import torch
from scipy.spatial.transform import Rotation
from torch.utils.data import IterableDataset

from vla_project.data import constants as C
from vla_project.data.normalization import (
    Q99Stats,
    load_q99_proprio_stats,
    load_q99_stats,
    normalize_action_q99,
    normalize_proprio_q99,
)
from vla_project.data.transforms.image import DINOv2ImageTransform, SiglipImageTransform
from vla_project.data.transforms.language import GemmaPromptTokenizer


# Gripper raw values in the dataset are in degrees per
# meta/info.json::gripper_convention (closed=0, open=100). Rescale to
# [0, 1] to mirror LIBERO's absolute-gripper convention.
_GRIPPER_DIVISOR: float = 100.0


def _default_lerobot_cls():
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    return LeRobotDataset


_LeRobotDatasetCls = None  # populated lazily; tests can override via monkeypatch


def _rotvec_delta_chunk(act_rotvec: torch.Tensor, obs_rotvec: torch.Tensor) -> torch.Tensor:
    """Per-frame canonical SO(3) delta: log(R_act @ R_obs^T).

    Plain ``act_rotvec - obs_rotvec`` crosses the antipodal discontinuity
    at ``‖rotvec‖ = π`` and produces spurious ~2π jumps (codex round 2
    measured 109/6303 such frames on this dataset). Building the relative
    rotation matrix and taking its log yields a rotation vector in
    ``[-π, π]`` per axis with magnitudes matching the actual per-step
    angular increment.

    Args:
        act_rotvec, obs_rotvec: float32 tensors of shape ``[H, 3]``.

    Returns:
        Tensor of shape ``[H, 3]`` (float32), same device as inputs.
    """
    if act_rotvec.shape != obs_rotvec.shape:
        raise ValueError(
            f"act_rotvec {tuple(act_rotvec.shape)} != obs_rotvec {tuple(obs_rotvec.shape)}"
        )
    if act_rotvec.shape[-1] != 3:
        raise ValueError(f"expected last dim 3, got {act_rotvec.shape}")
    a_np = act_rotvec.detach().cpu().numpy()
    o_np = obs_rotvec.detach().cpu().numpy()
    R_act = Rotation.from_rotvec(a_np)
    R_obs = Rotation.from_rotvec(o_np)
    d_np = (R_act * R_obs.inv()).as_rotvec()
    return torch.as_tensor(d_np, dtype=act_rotvec.dtype, device=act_rotvec.device)


class LeRobotSO101Dataset(IterableDataset):
    def __init__(
        self,
        root: Union[str, Path],
        stats_path: Union[str, Path],
        unnorm_key: str,
        fps: int,
        tokenizer: GemmaPromptTokenizer,
        action_chunk_len: int = C.ACTION_CHUNK_LEN,
        domain_id: int = 0,
        max_samples: Optional[int] = None,
        last_action_chunk_mode: str = "zero",
        download_videos: bool = False,
        include_scene_dinov2: bool = False,
        include_wrist_dinov2: bool = False,
        repo_id: str = "takaki99/test_so101",
    ) -> None:
        super().__init__()
        if last_action_chunk_mode != "zero":
            # 'real' mode (negative offsets for past chunk) is intentionally
            # not implemented here — the FT recipe matches v44 LIBERO
            # which trains with zero-mode. Add later if needed.
            raise ValueError(
                f"last_action_chunk_mode={last_action_chunk_mode!r} not supported "
                f"by LeRobotSO101Dataset (only 'zero')."
            )
        global _LeRobotDatasetCls
        if _LeRobotDatasetCls is None:
            _LeRobotDatasetCls = _default_lerobot_cls()

        # delta_timestamps fetches H frames at offsets [0/fps, ..., (H-1)/fps]
        # for both action.ee_* and observation.state.ee_*. With the same
        # offset list applied to both, frame index t of the chunk gives the
        # paired (action, obs) at the same timestep — required to compute
        # the per-frame EE-delta target.
        H = action_chunk_len
        offsets = [i / fps for i in range(H)]
        delta = {
            "action.ee_pos": offsets,
            "action.ee_rotvec": offsets,
            "action.gripper_pos": offsets,
            "observation.state.ee_pos": offsets,
            "observation.state.ee_rotvec": offsets,
            "observation.state.gripper_pos": offsets,
        }
        # tolerance_s=1e9 disables intra-episode timestamp-sync check
        # (same workaround used by LeRobotLiberoDataset for v2→v3 converted
        # datasets).
        self.ds = _LeRobotDatasetCls(
            repo_id,
            root=str(root),
            delta_timestamps=delta,
            download_videos=download_videos,
            tolerance_s=1e9,
        )
        self.action_chunk_len = action_chunk_len
        self.domain_id = int(domain_id)
        self.max_samples = max_samples
        self.last_action_chunk_mode = last_action_chunk_mode
        self.include_scene_dinov2 = bool(include_scene_dinov2)
        self.include_wrist_dinov2 = bool(include_wrist_dinov2)
        # The action stats are 7-dim (delta_xyz, delta_rotvec, abs_gripper);
        # proprio stats are 8-dim (proprio + zero pad).
        self.stats: Q99Stats = load_q99_stats(stats_path, unnorm_key)
        self.proprio_stats: Optional[Q99Stats] = load_q99_proprio_stats(
            stats_path, unnorm_key
        )
        if self.proprio_stats is None:
            raise ValueError(
                f"SO101 dataset requires proprio stats in {stats_path!r} "
                f"under key {unnorm_key!r}/proprio (run tools/compute_norm_stats_so101.py)."
            )
        if self.stats.q01.shape[0] != C.ACTION_DIM:
            raise ValueError(
                f"action stats dim {self.stats.q01.shape[0]} != ACTION_DIM={C.ACTION_DIM}"
            )
        if self.proprio_stats.q01.shape[0] != C.PROPRIO_DIM:
            raise ValueError(
                f"proprio stats dim {self.proprio_stats.q01.shape[0]} != PROPRIO_DIM={C.PROPRIO_DIM}"
            )
        self.image_tx = SiglipImageTransform(size=C.SIGLIP_IMAGE_SIZE, training=False)
        self.dinov2_image_tx = DINOv2ImageTransform(size=C.SIGLIP_IMAGE_SIZE)
        self.tokenizer = tokenizer
        self._task_idx_to_str: Dict[int, str] = self._build_task_map()

    def _build_task_map(self) -> Dict[int, str]:
        out: Dict[int, str] = {}
        tasks = self.ds.meta.tasks
        if hasattr(tasks, "iterrows"):
            for task_str, row in tasks.iterrows():
                out[int(row["task_index"])] = str(task_str).strip()
        elif isinstance(tasks, dict):
            for k, v in tasks.items():
                out[int(k)] = str(v).strip()
        elif isinstance(tasks, (list, tuple)):
            for i, v in enumerate(tasks):
                out[i] = str(v).strip()
        else:
            raise TypeError(f"unsupported tasks meta type: {type(tasks)!r}")
        return out

    def _resize_image(self, lerobot_img: torch.Tensor) -> torch.Tensor:
        if lerobot_img.shape[0] != 3:
            raise ValueError(f"expected (3, H, W), got {tuple(lerobot_img.shape)}")
        return self.image_tx(lerobot_img)

    def _sample_to_batch_item(self, sample: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        scene_raw = sample["observation.images.front"]
        scene = self._resize_image(scene_raw)
        wrist_raw = sample["observation.images.wrist"]
        wrist = self._resize_image(wrist_raw)

        H = self.action_chunk_len
        # All keyed by H-length chunks at the same offsets.
        a_pos = sample["action.ee_pos"].to(torch.float32)           # [H, 3]
        a_rot = sample["action.ee_rotvec"].to(torch.float32)        # [H, 3]
        a_grip = sample["action.gripper_pos"].to(torch.float32)     # [H] or [H, 1]
        o_pos = sample["observation.state.ee_pos"].to(torch.float32)
        o_rot = sample["observation.state.ee_rotvec"].to(torch.float32)
        o_grip = sample["observation.state.gripper_pos"].to(torch.float32)

        if a_grip.ndim == 1:
            a_grip = a_grip.unsqueeze(-1)
        if o_grip.ndim == 1:
            o_grip = o_grip.unsqueeze(-1)
        for name, t in [("a_pos", a_pos), ("a_rot", a_rot), ("o_pos", o_pos), ("o_rot", o_rot)]:
            if t.shape != (H, 3):
                raise ValueError(f"{name} shape {tuple(t.shape)} != ({H}, 3)")
        if a_grip.shape != (H, 1):
            raise ValueError(f"a_grip shape {tuple(a_grip.shape)} != ({H}, 1)")
        if o_grip.shape != (H, 1):
            raise ValueError(f"o_grip shape {tuple(o_grip.shape)} != ({H}, 1)")

        # EE-delta action target: [d_pos (3), d_rot (3), abs_gripper (1)] = 7-dim.
        d_pos = a_pos - o_pos
        d_rot = _rotvec_delta_chunk(a_rot, o_rot)
        grip = a_grip / _GRIPPER_DIVISOR
        action_raw = torch.cat([d_pos, d_rot, grip], dim=-1)  # [H, 7]
        if action_raw.shape != (H, C.ACTION_DIM):
            raise ValueError(
                f"action_raw shape {tuple(action_raw.shape)} != ({H}, {C.ACTION_DIM})"
            )
        target_action = normalize_action_q99(action_raw, self.stats)

        # Proprio at t=0: [obs.ee_pos (3), obs.ee_rotvec (3), obs.gripper/100 (1), 0 pad].
        proprio_raw = torch.cat(
            [o_pos[0], o_rot[0], o_grip[0] / _GRIPPER_DIVISOR, torch.zeros(1, dtype=torch.float32)],
            dim=-1,
        )  # [8]
        if proprio_raw.shape != (C.PROPRIO_DIM,):
            raise ValueError(
                f"proprio_raw shape {tuple(proprio_raw.shape)} != ({C.PROPRIO_DIM},)"
            )
        proprio = normalize_proprio_q99(proprio_raw, self.proprio_stats)

        # Last action chunk = zeros (FT recipe v46 mirrors v44 LIBERO zero-mode).
        last_action_chunk = torch.zeros(H, C.ACTION_DIM, dtype=torch.float32)

        # Prompt: task string lookup (single-task dataset → 1 string).
        task_idx_t = sample["task_index"]
        task_idx = int(task_idx_t.item()) if torch.is_tensor(task_idx_t) else int(task_idx_t)
        prompt_text = self._task_idx_to_str.get(task_idx, "")
        prompt = self.tokenizer(prompt_text)

        item = {
            "domain_id": torch.tensor(self.domain_id, dtype=torch.long),
            "scene_image": scene,
            "wrist_image": wrist,
            "prompt_input_ids": prompt["input_ids"],
            "prompt_attention_mask": prompt["attention_mask"],
            "proprio": proprio,
            "last_action_chunk": last_action_chunk,
            "target_action": target_action,
            "action_mask": torch.ones(H, dtype=torch.bool),
            # SO101 always carries a wrist camera, like LIBERO.
            "wrist_mask": torch.tensor(True, dtype=torch.bool),
        }
        if self.include_wrist_dinov2:
            item["wrist_image_dinov2"] = self.dinov2_image_tx(wrist_raw)
        if self.include_scene_dinov2:
            item["scene_image_dinov2"] = self.dinov2_image_tx(scene_raw)
        return item

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        # DDP + worker shard, shuffle per (rank, worker). Same pattern as
        # LeRobotLiberoDataset.
        import os
        import random as _random
        ddp_rank = int(os.environ.get("RANK", 0))
        ddp_world = int(os.environ.get("WORLD_SIZE", 1))
        info = torch.utils.data.get_worker_info()
        if info is None:
            worker_id, num_workers = 0, 1
        else:
            worker_id, num_workers = info.id, info.num_workers
        start = ddp_rank * num_workers + worker_id
        stride = ddp_world * num_workers
        rng = _random.Random(start)
        full_indices = list(range(start, len(self.ds), stride))
        rng.shuffle(full_indices)
        emitted = 0
        # lerobot 0.3.3 does NOT shorten chunks at episode end — it clamps
        # out-of-episode query indices to the boundary frame and returns
        # full-H chunks along with ``<key>_is_pad`` boolean masks marking
        # the clamped positions (codex round 3). Training on those clamped
        # actions would teach the model to predict zero EE-delta at the
        # padded frames, polluting the loss at every episode tail. Skip
        # samples where any of the delta_timestamps keys reports padding.
        pad_keys = (
            "action.ee_pos_is_pad",
            "action.ee_rotvec_is_pad",
            "action.gripper_pos_is_pad",
            "observation.state.ee_pos_is_pad",
            "observation.state.ee_rotvec_is_pad",
            "observation.state.gripper_pos_is_pad",
        )
        for i in full_indices:
            if self.max_samples is not None and emitted >= self.max_samples:
                return
            sample = self.ds[i]
            if any(pk in sample and bool(sample[pk].any()) for pk in pad_keys):
                continue
            yield self._sample_to_batch_item(sample)
            emitted += 1

    @staticmethod
    def collate_fn(samples: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        keys = samples[0].keys()
        return {k: torch.stack([s[k] for s in samples]) for k in keys}
