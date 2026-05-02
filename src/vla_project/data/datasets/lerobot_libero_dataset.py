"""LeRobot-HF based LIBERO step-level dataset.

Yields the project's internal Batch schema (see `data/schema.py`). Wraps
`lerobot.datasets.LeRobotDataset` with `delta_timestamps` so each yielded
sample contains an action chunk of length `ACTION_CHUNK_LEN`. Images are
resized to SigLIP's 224x224, normalized by SigLIP statistics. Action chunks
are normalized with BOUNDS_Q99 stats loaded from JSON. Prompts are tokenized
with the project's `GemmaPromptTokenizer`.

Two ``action_format`` modes:

- ``"native"``: the original LIBERO 7-dim delta-EE actions, normalized with
  BOUNDS_Q99 (Q99Stats from ``stats_path``). chunk_len = ``action_chunk_len``.
- ``"ee6d"``: X-VLA's common 20-dim ``[xyz, rot6d, gripper, 10×pad]`` action
  built per anchor from ``observation.state``. ``action_chunk_len`` is
  re-interpreted as the anchor count and ``anchor_window_s`` defines the
  total time window. EE6D actions are NOT Q99-normalized — values are
  already in reasonable ranges (xyz ~ meters, rot6d ∈ [-1, 1], gripper ∈
  [0, 1], pad = 0).

Single-domain only (Plan 1). With ``last_action_chunk_mode='zero'`` the
past-chunk slot is filled with zeros; with ``'real'`` it carries the prior
H actions / past N anchors fetched at negative offsets.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import torch
from torch.utils.data import IterableDataset

from vla_project.data import constants as C
from vla_project.data.normalization import (
    Q99Stats,
    load_q99_proprio_stats,
    load_q99_stats,
    normalize_action_q99,
    normalize_proprio_q99,
)
from vla_project.data.transforms.action_alignment import (
    anchor_offsets,
    ee_pose_to_action20,
)
from vla_project.data.transforms.image import SiglipImageTransform
from vla_project.data.transforms.language import GemmaPromptTokenizer


# Indirection so tests can monkey-patch without importing lerobot at import time.
def _default_lerobot_cls():
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    return LeRobotDataset


_LeRobotDatasetCls = None  # populated lazily; tests can override via monkeypatch


class LeRobotLiberoDataset(IterableDataset):
    def __init__(
        self,
        repo_id: str,
        stats_path: Union[str, Path],
        unnorm_key: str,
        fps: int,
        tokenizer: GemmaPromptTokenizer,
        episodes: Optional[List[int]] = None,
        action_chunk_len: int = C.ACTION_CHUNK_LEN,
        download_videos: bool = True,
        domain_id: int = 0,
        max_samples: Optional[int] = None,
        last_action_chunk_mode: str = "zero",
        action_format: str = "native",
        anchor_window_s: float = 0.0,
        task_index_filter: Optional[int] = None,
    ) -> None:
        super().__init__()
        if last_action_chunk_mode not in ("zero", "real"):
            raise ValueError(
                f"last_action_chunk_mode must be 'zero' or 'real'; got {last_action_chunk_mode!r}"
            )
        if action_format not in ("native", "ee6d"):
            raise ValueError(
                f"action_format must be 'native' or 'ee6d'; got {action_format!r}"
            )
        if action_format == "ee6d":
            if action_chunk_len < 2:
                raise ValueError(
                    f"action_format='ee6d' needs action_chunk_len >= 2 anchors; got {action_chunk_len}"
                )
            if anchor_window_s <= 0:
                raise ValueError(
                    f"action_format='ee6d' needs anchor_window_s > 0; got {anchor_window_s}"
                )
        global _LeRobotDatasetCls
        if _LeRobotDatasetCls is None:
            _LeRobotDatasetCls = _default_lerobot_cls()
        # Build delta_timestamps for the chunk we want.
        # native: action chunk indexed by frames at [-H, H-1]/fps (real) or
        #   [0, H-1]/fps (zero).
        # ee6d:   observation.state at anchor times spaced over anchor_window_s.
        #   spacing = window / (H - 1). Real mode uses the action-style 2H
        #   pattern: past N at offsets [-N*sp, ..., -sp]; future N at [0, ..., window].
        if action_format == "native":
            if last_action_chunk_mode == "real":
                offsets = [(i - action_chunk_len) / fps for i in range(2 * action_chunk_len)]
            else:
                offsets = [i / fps for i in range(action_chunk_len)]
            delta = {"action": offsets}
        else:  # ee6d
            future_offs = anchor_offsets(anchor_window_s, action_chunk_len, fps)
            spacing = anchor_window_s / (action_chunk_len - 1)
            if last_action_chunk_mode == "real":
                past_offs = [(k - action_chunk_len) * spacing for k in range(action_chunk_len)]
                offsets = past_offs + future_offs
            else:
                offsets = list(future_offs)
            delta = {"observation.state": offsets}
        # tolerance_s=1e9 disables LeRobot's intra-episode timestamp-sync check
        # at __init__. The check fires on lerobot/libero_*_image (v2.0->v3.0
        # converted) because some intra-episode diffs do not equal 1/fps within
        # the default 1e-4. Stats / training only need the action values; the
        # tolerance is irrelevant. Same workaround as tools/compute_norm_stats.py.
        self.ds = _LeRobotDatasetCls(
            repo_id,
            delta_timestamps=delta,
            episodes=episodes,
            download_videos=download_videos,
            tolerance_s=1e9,
        )
        self.action_chunk_len = action_chunk_len
        self.domain_id = int(domain_id)
        self.max_samples = max_samples
        self.last_action_chunk_mode = last_action_chunk_mode
        self.action_format = action_format
        self.anchor_window_s = float(anchor_window_s)
        # Optional post-hoc task filter: skip samples whose task_index doesn't
        # match. Use this when you need a single-task subset and the wrapped
        # LeRobotDataset's ``episodes=[...]`` filter would mis-reindex (its
        # ``episode_data_index`` ends up sized to the filter length but
        # samples retain their ORIGINAL ``episode_index`` values, causing
        # IndexError on _get_query_indices for non-contiguous filters).
        self.task_index_filter = (
            int(task_index_filter) if task_index_filter is not None else None
        )
        # Q99 stats only used in native mode; EE6D values are pre-bounded.
        self.stats: Q99Stats = load_q99_stats(stats_path, unnorm_key)
        # Optional proprio stats: 73% vla-gemma-4 baseline trained with RLDS-
        # auto-normalized proprio (BOUNDS_Q99). Without normalization the
        # proprio_proj DA-Linear sees axis-angle/m units (z≈0.92-1.29, rx≈
        # 2.78-3.28) that are far from the LLM embedding scale. Returns None
        # when the stats file lacks a proprio block, preserving raw behavior
        # for older configs / smoke tests; new datasets should ship with
        # proprio q01/q99/mask alongside the action block.
        self.proprio_stats: Optional[Q99Stats] = load_q99_proprio_stats(
            stats_path, unnorm_key
        )
        self.image_tx = SiglipImageTransform(size=C.SIGLIP_IMAGE_SIZE, training=False)
        self.tokenizer = tokenizer
        self._task_idx_to_str: Dict[int, str] = self._build_task_map()
        # Pre-compute the list of valid frame indices when task_index_filter is
        # set. Iterating the full dataset and skipping at sample-load time is
        # too slow (every skipped sample still pays full image-decode I/O); we
        # walk the episode metadata + episode_data_index instead and restrict
        # iteration to frames that belong to the target task. None means
        # "iterate all frames", checked in __iter__.
        self._task_filtered_indices: Optional[List[int]] = None
        if self.task_index_filter is not None:
            target_str = self._task_idx_to_str.get(self.task_index_filter)
            if target_str is None:
                raise ValueError(
                    f"task_index_filter={self.task_index_filter} not in "
                    f"task map (have {sorted(self._task_idx_to_str)})"
                )
            ep_meta = self.ds.meta.episodes
            edi_from = self.ds.episode_data_index["from"]
            edi_to = self.ds.episode_data_index["to"]
            valid: List[int] = []
            for ep_id, info in ep_meta.items():
                tasks = info.get("tasks", [])
                if target_str.strip() in {str(t).strip() for t in tasks}:
                    f, t = int(edi_from[ep_id]), int(edi_to[ep_id])
                    valid.extend(range(f, t))
            if not valid:
                raise ValueError(
                    f"task_index_filter={self.task_index_filter} ({target_str!r}) "
                    f"matched 0 episodes; check task index ↔ string mapping"
                )
            self._task_filtered_indices = valid

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
        scene = self._resize_image(sample["observation.images.image"])
        wrist = self._resize_image(sample["observation.images.wrist_image"])
        H = self.action_chunk_len

        if self.action_format == "native":
            proprio = sample["observation.state"].to(torch.float32)
            if proprio.shape != (C.PROPRIO_DIM,):
                raise ValueError(
                    f"proprio shape {tuple(proprio.shape)} != ({C.PROPRIO_DIM},)"
                )
            action_raw = sample["action"].to(torch.float32)
            if self.last_action_chunk_mode == "real":
                expected = (2 * H, C.ACTION_DIM)
                if action_raw.shape != expected:
                    raise ValueError(
                        f"action shape {tuple(action_raw.shape)} != {expected} "
                        f"(real-mode expects 2*H entries)"
                    )
                past_raw = action_raw[:H]
                future_raw = action_raw[H:]
                last_action_chunk = normalize_action_q99(past_raw, self.stats)
                target_action = normalize_action_q99(future_raw, self.stats)
                # Zero-pad the past at episode start: LeRobot clamps offsets
                # that fall before frame 0 to the first action, leaking values.
                frame_index_t = sample.get("frame_index")
                if frame_index_t is not None:
                    fi = int(frame_index_t.item()) if torch.is_tensor(frame_index_t) else int(frame_index_t)
                    n_clamped = max(0, H - fi)
                    if n_clamped > 0:
                        last_action_chunk = last_action_chunk.clone()
                        last_action_chunk[:n_clamped] = 0.0
            else:
                if action_raw.shape != (H, C.ACTION_DIM):
                    raise ValueError(
                        f"action shape {tuple(action_raw.shape)} != "
                        f"({H}, {C.ACTION_DIM})"
                    )
                target_action = normalize_action_q99(action_raw, self.stats)
                last_action_chunk = torch.zeros(H, C.ACTION_DIM, dtype=torch.float32)
        else:  # ee6d
            state_chunk = sample["observation.state"].to(torch.float32)
            if self.last_action_chunk_mode == "real":
                expected_state = (2 * H, C.PROPRIO_DIM)
                if state_chunk.shape != expected_state:
                    raise ValueError(
                        f"observation.state shape {tuple(state_chunk.shape)} != "
                        f"{expected_state} (ee6d real-mode expects 2*N anchors)"
                    )
                past_state = state_chunk[:H]
                future_state = state_chunk[H:]
            else:
                if state_chunk.shape != (H, C.PROPRIO_DIM):
                    raise ValueError(
                        f"observation.state shape {tuple(state_chunk.shape)} != "
                        f"({H}, {C.PROPRIO_DIM})"
                    )
                past_state = None
                future_state = state_chunk
            # Current proprio is the t=0 anchor (first future entry).
            proprio = future_state[0].clone()
            target_action = ee_pose_to_action20(
                future_state[..., :3], future_state[..., 3:7], future_state[..., 7:8]
            )
            if past_state is not None:
                last_action_chunk = ee_pose_to_action20(
                    past_state[..., :3], past_state[..., 3:7], past_state[..., 7:8]
                )
                # LeRobot clamps offsets that fall before frame 0 to the first
                # frame's state. For ee6d that's "robot at rest" — semantically
                # correct, no masking needed.
            else:
                last_action_chunk = torch.zeros(H, 20, dtype=torch.float32)

        task_idx_t = sample["task_index"]
        task_idx = int(task_idx_t.item()) if torch.is_tensor(task_idx_t) else int(task_idx_t)
        prompt_text = self._task_idx_to_str.get(task_idx, "")
        prompt = self.tokenizer(prompt_text)

        if self.proprio_stats is not None:
            proprio = normalize_proprio_q99(proprio, self.proprio_stats)
        return {
            "domain_id": torch.tensor(self.domain_id, dtype=torch.long),
            "scene_image": scene,
            "wrist_image": wrist,
            "prompt_input_ids": prompt["input_ids"],
            "prompt_attention_mask": prompt["attention_mask"],
            "proprio": proprio,
            "last_action_chunk": last_action_chunk,
            "target_action": target_action,
            "action_mask": torch.ones(H, dtype=torch.bool),
        }

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        # Shard the frame range across DataLoader workers: worker 0 reads
        # indices [0, N, 2N, ...], worker 1 reads [1, N+1, 2N+1, ...] etc.
        # This avoids each worker iterating the full dataset (and yielding
        # duplicate samples) under DataLoader(num_workers > 0).
        info = torch.utils.data.get_worker_info()
        if info is None:
            start, stride = 0, 1
        else:
            start, stride = info.id, info.num_workers
        emitted = 0
        expected_chunk_len = (
            2 * self.action_chunk_len
            if self.last_action_chunk_mode == "real"
            else self.action_chunk_len
        )
        # Single-pass iteration. Trainer calls iter(dataloader) again to restart.
        chunk_key = "action" if self.action_format == "native" else "observation.state"
        if self._task_filtered_indices is not None:
            # Pre-computed: only frames whose episode matches task_index_filter.
            indices = self._task_filtered_indices[start::stride]
        else:
            indices = range(start, len(self.ds), stride)
        for i in indices:
            if self.max_samples is not None and emitted >= self.max_samples:
                return
            sample = self.ds[i]
            if sample[chunk_key].shape[0] != expected_chunk_len:
                # delta_timestamps near episode end may yield a short chunk; skip.
                continue
            yield self._sample_to_batch_item(sample)
            emitted += 1

    @staticmethod
    def collate_fn(samples: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        keys = samples[0].keys()
        return {k: torch.stack([s[k] for s in samples]) for k in keys}
