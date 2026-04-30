"""LeRobot-HF based LIBERO step-level dataset.

Yields the project's internal Batch schema (see `data/schema.py`). Wraps
`lerobot.datasets.LeRobotDataset` with `delta_timestamps` so each yielded
sample contains an action chunk of length `ACTION_CHUNK_LEN`. Images are
resized to SigLIP's 224x224, normalized by SigLIP statistics. Action chunks
are normalized with BOUNDS_Q99 stats loaded from JSON. Prompts are tokenized
with the project's `GemmaPromptTokenizer`.

Single-domain only (Plan 1). `last_action_chunk` is zeros (cold-start; real
prior-chunk fetching is Plan 3 / future work).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union

import torch
from torch.utils.data import IterableDataset

from vla_project.data import constants as C
from vla_project.data.normalization import Q99Stats, load_q99_stats, normalize_action_q99
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
    ) -> None:
        super().__init__()
        global _LeRobotDatasetCls
        if _LeRobotDatasetCls is None:
            _LeRobotDatasetCls = _default_lerobot_cls()
        delta = {"action": [i / fps for i in range(action_chunk_len)]}
        self.ds = _LeRobotDatasetCls(
            repo_id,
            delta_timestamps=delta,
            episodes=episodes,
            download_videos=download_videos,
        )
        self.action_chunk_len = action_chunk_len
        self.domain_id = int(domain_id)
        self.max_samples = max_samples
        self.stats: Q99Stats = load_q99_stats(stats_path, unnorm_key)
        self.image_tx = SiglipImageTransform(size=C.SIGLIP_IMAGE_SIZE, training=False)
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
        scene = self._resize_image(sample["observation.images.image"])
        wrist = self._resize_image(sample["observation.images.wrist_image"])
        proprio = sample["observation.state"].to(torch.float32)
        if proprio.shape != (C.PROPRIO_DIM,):
            raise ValueError(f"proprio shape {tuple(proprio.shape)} != ({C.PROPRIO_DIM},)")
        action_raw = sample["action"].to(torch.float32)
        if action_raw.shape != (self.action_chunk_len, C.ACTION_DIM):
            raise ValueError(
                f"action shape {tuple(action_raw.shape)} != "
                f"({self.action_chunk_len}, {C.ACTION_DIM})"
            )
        target_action = normalize_action_q99(action_raw, self.stats)

        task_idx_t = sample["task_index"]
        task_idx = int(task_idx_t.item()) if torch.is_tensor(task_idx_t) else int(task_idx_t)
        prompt_text = self._task_idx_to_str.get(task_idx, "")
        prompt = self.tokenizer(prompt_text)

        return {
            "domain_id": torch.tensor(self.domain_id, dtype=torch.long),
            "scene_image": scene,
            "wrist_image": wrist,
            "prompt_input_ids": prompt["input_ids"],
            "prompt_attention_mask": prompt["attention_mask"],
            "proprio": proprio,
            "last_action_chunk": torch.zeros(
                self.action_chunk_len, C.ACTION_DIM, dtype=torch.float32
            ),
            "target_action": target_action,
            "action_mask": torch.ones(self.action_chunk_len, dtype=torch.bool),
        }

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        emitted = 0
        # Single-pass iteration. Trainer calls iter(dataloader) again to restart.
        for i in range(len(self.ds)):
            if self.max_samples is not None and emitted >= self.max_samples:
                return
            sample = self.ds[i]
            if sample["action"].shape[0] != self.action_chunk_len:
                # delta_timestamps near episode end may yield a short chunk; skip.
                continue
            yield self._sample_to_batch_item(sample)
            emitted += 1

    @staticmethod
    def collate_fn(samples: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        keys = samples[0].keys()
        return {k: torch.stack([s[k] for s in samples]) for k in keys}
