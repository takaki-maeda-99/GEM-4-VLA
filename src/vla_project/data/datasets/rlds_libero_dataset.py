"""RLDS-backed LIBERO dataset matching the 73% vla-gemma-4 baseline data
distribution. Wraps ``vla-gemma-4/scripts/gemma4/libero_loader.py``'s
``LiberoChunkIterableDataset`` so we can train on the SAME RLDS source the
baseline used (``data/modified_libero_rlds``), converting each yielded
sample into our internal Batch schema.

Key differences from ``LeRobotLiberoDataset``:
  - Action and proprio come out of RLDS already BOUNDS_Q99-normalized
    (``action_proprio_normalization_type=NormalizationType.BOUNDS_Q99`` in
    libero_loader.py:178). We do NOT re-normalize.
  - Images come as ``(H, W, 3) uint8`` from RLDS instead of LeRobot's
    ``(3, H, W) float [0, 1]``. Convert HWC→CHW + cast to [0, 1] before
    passing through SiglipImageTransform.
  - Shuffle is handled inside RLDS (``shuffle_buffer_size`` arg). No need
    to shuffle ourselves.

Requires the vla-gemma-4 venv (tensorflow + dlimp + prismatic.vla.datasets
on PYTHONPATH). Use ``/misc/dl00/takaki/vla-gemma-4/.venv-gemma4/bin/python``
when launching trainer with this dataset.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterator, Optional, Union

import numpy as np
import torch
from torch.utils.data import IterableDataset

from vla_project.data import constants as C
from vla_project.data.transforms.image import SiglipImageTransform
from vla_project.data.transforms.language import GemmaPromptTokenizer


class RLDSLiberoDataset(IterableDataset):
    """Yield internal Batch items from the vla-gemma-4 RLDS LIBERO loader."""

    def __init__(
        self,
        data_dir: Union[str, Path],
        dataset_name: str = "libero_spatial_no_noops",
        tokenizer: Optional[GemmaPromptTokenizer] = None,
        action_chunk_len: int = C.ACTION_CHUNK_LEN,
        shuffle_buffer_size: int = 256000,
        train: bool = True,
        domain_id: int = 0,
        seed: int = 42,
    ) -> None:
        super().__init__()
        if tokenizer is None:
            raise ValueError("tokenizer is required")
        self.data_dir = str(data_dir)
        self.dataset_name = dataset_name
        self.tokenizer = tokenizer
        self.action_chunk_len = action_chunk_len
        self.shuffle_buffer_size = shuffle_buffer_size
        self.train = train
        self.domain_id = int(domain_id)
        self.seed = seed
        self.image_tx = SiglipImageTransform(size=C.SIGLIP_IMAGE_SIZE, training=False)

    def _build_rlds(self):
        # Local import: requires vla-gemma-4 venv (tensorflow + prismatic.vla).
        from prismatic.vla.datasets.rlds.dataset import make_interleaved_dataset
        from prismatic.vla.datasets.rlds.oxe import get_oxe_dataset_kwargs_and_weights
        from prismatic.vla.datasets.rlds.utils.data_utils import NormalizationType

        IMAGE_SIZE = C.SIGLIP_IMAGE_SIZE
        mixture_spec = [(self.dataset_name, 1.0)]
        per_dataset_kwargs, weights = get_oxe_dataset_kwargs_and_weights(
            self.data_dir,
            mixture_spec,
            load_camera_views=("primary", "wrist"),
            load_depth=False,
            load_proprio=True,
            load_language=True,
            action_proprio_normalization_type=NormalizationType.BOUNDS_Q99,
        )
        rlds_config = dict(
            traj_transform_kwargs=dict(
                window_size=1,
                future_action_window_size=self.action_chunk_len - 1,
                skip_unlabeled=True,
                goal_relabeling_strategy="uniform",
            ),
            frame_transform_kwargs=dict(
                resize_size=(IMAGE_SIZE, IMAGE_SIZE),
                num_parallel_calls=16,
            ),
            dataset_kwargs_list=per_dataset_kwargs,
            shuffle_buffer_size=self.shuffle_buffer_size,
            sample_weights=weights,
            balance_weights=True,
            traj_transform_threads=len(mixture_spec),
            traj_read_threads=len(mixture_spec),
            train=self.train,
        )
        dataset, _length, _stats = make_interleaved_dataset(**rlds_config)
        return dataset

    @staticmethod
    def _hwc_uint8_to_chw_float01(img: np.ndarray) -> torch.Tensor:
        """RLDS yields uint8 HWC; SigLIP expects float CHW in [0, 1]."""
        t = torch.from_numpy(img).permute(2, 0, 1).contiguous().float() / 255.0
        return t

    def _to_batch_item(self, raw: Dict) -> Dict[str, torch.Tensor]:
        scene_img = raw["observation"]["image_primary"][0]  # (H, W, 3) uint8
        wrist_img = raw["observation"]["image_wrist"][0]
        proprio = np.asarray(raw["observation"]["proprio"][0], dtype=np.float32)  # (8,)
        action_chunk = np.asarray(raw["action"], dtype=np.float32)                # (H, 7)

        # SigLIP transform expects (3, H, W) float; convert HWC→CHW + scale.
        scene = self.image_tx(self._hwc_uint8_to_chw_float01(scene_img))
        wrist = self.image_tx(self._hwc_uint8_to_chw_float01(wrist_img))

        lang_raw = raw["task"]["language_instruction"]
        if isinstance(lang_raw, bytes):
            language = lang_raw.decode("utf-8").strip()
        else:
            language = str(lang_raw).strip()
        prompt = self.tokenizer(language)

        target_action = torch.from_numpy(action_chunk).to(torch.float32)
        # Last action chunk: zero (matches our v6+ Phase A Bridge match
        # `last_action_chunk_mode='zero'`).
        last_action = torch.zeros(self.action_chunk_len, C.ACTION_DIM, dtype=torch.float32)

        return {
            "domain_id": torch.tensor(self.domain_id, dtype=torch.long),
            "scene_image": scene,
            "wrist_image": wrist,
            "prompt_input_ids": prompt["input_ids"],
            "prompt_attention_mask": prompt["attention_mask"],
            "proprio": torch.from_numpy(proprio),  # already BOUNDS_Q99 normalized
            "last_action_chunk": last_action,
            "target_action": target_action,         # already BOUNDS_Q99 normalized
            "action_mask": torch.ones(self.action_chunk_len, dtype=torch.bool),
        }

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        # RLDS internally handles per-worker sharding via shuffle and the
        # tf.data parallelism config. We just iterate forever (train=True
        # implies infinite repeat).
        ds_tf = self._build_rlds()
        for rlds_batch in ds_tf.as_numpy_iterator():
            yield self._to_batch_item(rlds_batch)

    @staticmethod
    def collate_fn(samples):
        """Stack list of per-sample dicts into batched tensors. Reuses the
        same shape contract as ``LeRobotLiberoDataset.collate_fn``."""
        keys = samples[0].keys()
        out = {}
        for k in keys:
            v0 = samples[0][k]
            if torch.is_tensor(v0):
                out[k] = torch.stack([s[k] for s in samples], dim=0)
            else:
                out[k] = [s[k] for s in samples]
        return out
