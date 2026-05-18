"""LeRobot v2.1 dataset with pre-extracted uint8 frames (no mp4 decode at
batch time).

Use ``tools/extract_lerobot_frames.py`` first to produce the per-(episode,
camera) ``frames_uint8/<camera>/episode_NNNNNN.npy`` files of shape
``[T, 224, 224, 3]``. This class reads parquets directly for the
proprio/action chunks (via pyarrow) and memmaps the npy files for image
fetch, bypassing LeRobotDataset entirely.

The output sample matches ``LeRobotSO101Dataset`` so VLAPolicy / collator
need no changes. EE-delta computation (with rotvec logmap) is identical.

Required parquet columns:
  - observation.state.ee_pos, observation.state.ee_rotvec,
    observation.state.gripper_pos, action.ee_pos, action.ee_rotvec,
    action.gripper_pos, task_index, frame_index, episode_index
  - any extra columns are ignored.
"""
from __future__ import annotations

import bisect
import os
import random
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.distributed as dist
from torch.utils.data import IterableDataset, get_worker_info

from vla_project.data import constants as C
from vla_project.data.normalization import (
    Q99Stats,
    load_q99_proprio_stats,
    load_q99_stats,
    normalize_action_q99,
    normalize_proprio_q99,
)
from vla_project.data.transforms.language import GemmaPromptTokenizer
from vla_project.data.datasets.lerobot_so101_dataset import _rotvec_delta_chunk

_GRIPPER_DIVISOR = 100.0
_SIGLIP_MEAN = 0.5
_SIGLIP_STD = 0.5


class LeRobotPreExtractedDataset(IterableDataset):
    """LeRobot v2.1 + pre-extracted uint8 frames dataset.

    ``root`` is the v2.1 dataset root (with ``data/`` parquets and
    ``meta/info.json`` + ``meta/tasks.jsonl``). ``frames_root`` points to
    the directory containing one subdir per camera key with per-episode
    npy files.
    """

    def __init__(
        self,
        root: Union[str, Path],
        frames_root: Union[str, Path],
        stats_path: Union[str, Path],
        unnorm_key: str,
        fps: int,
        tokenizer: GemmaPromptTokenizer,
        action_chunk_len: int = C.ACTION_CHUNK_LEN,
        domain_id: int = 0,
        max_samples: Optional[int] = None,
        last_action_chunk_mode: str = "zero",
        scene_key: str = "observation.images.front",
        wrist_key: str = "observation.images.wrist",
        seed: int = 0,
    ) -> None:
        super().__init__()
        if last_action_chunk_mode != "zero":
            raise ValueError(
                f"last_action_chunk_mode={last_action_chunk_mode!r} not supported"
            )
        self.root = Path(root).resolve()
        self.frames_root = Path(frames_root).resolve()
        self.action_chunk_len = int(action_chunk_len)
        self.domain_id = int(domain_id)
        self.max_samples = max_samples
        self.scene_key = scene_key
        self.wrist_key = wrist_key
        self.seed = int(seed)

        self.stats: Q99Stats = load_q99_stats(stats_path, unnorm_key)
        self.proprio_stats: Optional[Q99Stats] = load_q99_proprio_stats(
            stats_path, unnorm_key
        )
        if self.proprio_stats is None:
            raise ValueError(
                f"missing proprio stats in {stats_path!r} under {unnorm_key!r}"
            )
        if self.stats.q01.shape[0] != C.ACTION_DIM:
            raise ValueError(f"action stats dim != {C.ACTION_DIM}")
        if self.proprio_stats.q01.shape[0] != C.PROPRIO_DIM:
            raise ValueError(f"proprio stats dim != {C.PROPRIO_DIM}")

        # Walk parquets, collect per-episode dataframes (small dataset, full
        # in-memory is fine: bottle is ~50 MB total parquet).
        ep_parquets = sorted((self.root / "data/chunk-000").glob("episode_*.parquet"))
        if not ep_parquets:
            raise FileNotFoundError(f"no parquets under {self.root}/data/chunk-000")
        self._ep_tables: List[Dict[str, np.ndarray]] = []
        self._ep_lens: List[int] = []
        cum_offsets: List[int] = [0]
        for pth in ep_parquets:
            tbl = pq.read_table(str(pth), columns=[
                "observation.state.ee_pos",
                "observation.state.ee_rotvec",
                "observation.state.gripper_pos",
                "action.ee_pos",
                "action.ee_rotvec",
                "action.gripper_pos",
                "task_index",
                "frame_index",
                "episode_index",
            ])
            # Materialize nested arrays as np arrays.
            df = tbl.to_pandas()
            cols: Dict[str, np.ndarray] = {}
            for c in tbl.column_names:
                arr = df[c].to_numpy()
                if arr.dtype == object:
                    arr = np.stack([np.asarray(v, dtype=np.float32) for v in arr])
                cols[c] = arr
            self._ep_tables.append(cols)
            ep_len = len(df)
            self._ep_lens.append(ep_len)
            cum_offsets.append(cum_offsets[-1] + ep_len)
        self._cum_offsets = cum_offsets  # [ep_count + 1]
        self.total_frames = cum_offsets[-1]

        # Memmap pre-extracted frames per (camera, episode).
        self._scene_mmaps: List[np.memmap] = []
        self._wrist_mmaps: List[np.memmap] = []
        for ep in range(len(ep_parquets)):
            sp = self.frames_root / self.scene_key / f"episode_{ep:06d}.npy"
            wp = self.frames_root / self.wrist_key / f"episode_{ep:06d}.npy"
            if not sp.exists() or not wp.exists():
                raise FileNotFoundError(
                    f"missing pre-extracted frames for ep {ep}: {sp} / {wp}"
                )
            self._scene_mmaps.append(np.load(str(sp), mmap_mode="r"))
            self._wrist_mmaps.append(np.load(str(wp), mmap_mode="r"))

        # Valid (ep, frame) indices: frame + H <= ep_len so the action chunk
        # is fully within-episode. Beyond that, the last chunk is zero-padded
        # but actions still need a valid horizon — keep within-bounds.
        H = self.action_chunk_len
        self._valid_pairs: List[tuple] = []
        for ep, L in enumerate(self._ep_lens):
            for f in range(max(0, L - H + 1)):
                self._valid_pairs.append((ep, f))
        if not self._valid_pairs:
            raise RuntimeError("no valid samples (H exceeds every episode length?)")
        print(
            f"[LeRobotPreExtractedDataset] {len(ep_parquets)} eps, "
            f"{self.total_frames} frames, {len(self._valid_pairs)} valid sample positions"
        )

        # Tasks map.
        self.tokenizer = tokenizer
        self._task_idx_to_str: Dict[int, str] = self._build_task_map()

    def _build_task_map(self) -> Dict[int, str]:
        # Read tasks.jsonl (lerobot v2.1 layout).
        import jsonlines
        out: Dict[int, str] = {}
        path = self.root / "meta/tasks.jsonl"
        if not path.exists():
            raise FileNotFoundError(path)
        with jsonlines.open(path) as r:
            for row in r:
                out[int(row["task_index"])] = str(row["task"]).strip()
        return out

    def _sample(self, ep: int, frame: int) -> Dict[str, torch.Tensor]:
        H = self.action_chunk_len
        cols = self._ep_tables[ep]
        sl = slice(frame, frame + H)

        # Image fetch at frame t only (current frame).
        scene_u8 = np.asarray(self._scene_mmaps[ep][frame])  # [224,224,3] uint8
        wrist_u8 = np.asarray(self._wrist_mmaps[ep][frame])
        scene = torch.from_numpy(scene_u8).permute(2, 0, 1).float() / 255.0
        wrist = torch.from_numpy(wrist_u8).permute(2, 0, 1).float() / 255.0
        scene = (scene - _SIGLIP_MEAN) / _SIGLIP_STD
        wrist = (wrist - _SIGLIP_MEAN) / _SIGLIP_STD

        a_pos = torch.from_numpy(cols["action.ee_pos"][sl]).float()
        a_rot = torch.from_numpy(cols["action.ee_rotvec"][sl]).float()
        a_grip = cols["action.gripper_pos"][sl]
        a_grip = torch.from_numpy(np.asarray(a_grip)).float()
        if a_grip.ndim == 1:
            a_grip = a_grip.unsqueeze(-1)
        o_pos = torch.from_numpy(cols["observation.state.ee_pos"][sl]).float()
        o_rot = torch.from_numpy(cols["observation.state.ee_rotvec"][sl]).float()
        o_grip = cols["observation.state.gripper_pos"][sl]
        o_grip = torch.from_numpy(np.asarray(o_grip)).float()
        if o_grip.ndim == 1:
            o_grip = o_grip.unsqueeze(-1)

        d_pos = a_pos - o_pos
        d_rot = _rotvec_delta_chunk(a_rot, o_rot)
        grip = a_grip / _GRIPPER_DIVISOR
        action_raw = torch.cat([d_pos, d_rot, grip], dim=-1)  # [H, 7]
        target_action = normalize_action_q99(action_raw, self.stats)

        proprio_raw = torch.cat(
            [o_pos[0], o_rot[0], o_grip[0] / _GRIPPER_DIVISOR,
             torch.zeros(1, dtype=torch.float32)],
            dim=-1,
        )
        proprio = normalize_proprio_q99(proprio_raw, self.proprio_stats)

        last_action_chunk = torch.zeros(H, C.ACTION_DIM, dtype=torch.float32)

        task_idx = int(cols["task_index"][frame])
        prompt_text = self._task_idx_to_str.get(task_idx, "")
        prompt = self.tokenizer(prompt_text)

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
            "wrist_mask": torch.tensor(True, dtype=torch.bool),
        }

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        # DDP + worker shard with same pattern as LeRobotSO101Dataset.
        if dist.is_available() and dist.is_initialized():
            ddp_rank = dist.get_rank()
            ddp_world = dist.get_world_size()
        else:
            ddp_rank, ddp_world = 0, 1
        info = get_worker_info()
        if info is None:
            worker_id, num_workers = 0, 1
        else:
            worker_id, num_workers = info.id, info.num_workers
        start = ddp_rank * num_workers + worker_id
        stride = ddp_world * num_workers
        rng = random.Random(self.seed + start)
        emitted = 0
        # Stream indefinitely (training loop controls termination via max_steps).
        local = self._valid_pairs[start::stride]
        while True:
            rng.shuffle(local)
            for ep, f in local:
                yield self._sample(ep, f)
                emitted += 1
                if self.max_samples is not None and emitted >= self.max_samples:
                    return

    @staticmethod
    def collate_fn(samples: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}
        keys = samples[0].keys()
        for k in keys:
            out[k] = torch.stack([s[k] for s in samples])
        return out
