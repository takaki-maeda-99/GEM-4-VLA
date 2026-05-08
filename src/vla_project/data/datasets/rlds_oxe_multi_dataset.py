"""Single-shuffle multi-source OXE RLDS loader.

Architectural fix for the 286 GB rank-0 host-RAM baseline observed in v37
nb18even bs=8 (commit `2270a0c` and earlier). The previous wiring was
``WeightedMultiDataset`` over 9 ``RLDSOxeDataset`` children, where each
child built its own tf.data pipeline ending in ``shuffle(65536)``. With
Accelerate's default ``dispatch_batches=True`` for ``IterableDataset``,
all 9 shuffle buffers lived in rank-0's process — a steady-state
~9 × 65536 × ~485 KB ≈ 286 GB anonymous heap.

This module replaces those 9 buffers with **one** combined tf.data graph::

  for each source:
    make_dataset_from_rlds → filter(short trajs) → repeat
    → apply_trajectory_transforms → flatten
    → frame_map(inject domain_id)

  sample_from_datasets(weights)  # frame-level interleave
  → shuffle(N)                   # single shuffle for the whole mix
  → apply_frame_transforms       # decode + resize, once
  → with_ram_budget(1)

Expected rank-0 baseline: 65536 × ~485 KB ≈ 30 GB instead of 286 GB.

Behavior parity with :class:`RLDSOxeDataset`:
  - ``_to_batch_item`` produces the same internal batch schema (image
    transforms, proprio padding, wrist_mask from RLDS pad_mask_dict, etc).
  - The only difference is that ``domain_id`` is read from the per-frame
    tf.data record (injected per-source above) instead of the dataset
    instance attribute.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch.utils.data import IterableDataset

from vla_project.data import constants as C
from vla_project.data.transforms.image import DINOv2ImageTransform, SiglipImageTransform
from vla_project.data.transforms.language import GemmaPromptTokenizer


class RLDSOxeMultiDataset(IterableDataset):
    """Multi-source OXE loader with a single shuffle buffer."""

    def __init__(
        self,
        data_dir: Union[str, Path],
        sources: Sequence[Tuple[str, int, float]],
        tokenizer: GemmaPromptTokenizer,
        action_chunk_len: int = C.ACTION_CHUNK_LEN,
        shuffle_buffer_size: int = 65536,
        train: bool = True,
        seed: int = 42,
        include_scene_dinov2: bool = False,
        include_wrist_dinov2: bool = False,
        check_contract: bool = True,
    ) -> None:
        super().__init__()
        if not sources:
            raise ValueError("sources is empty")
        self.data_dir = str(data_dir)
        self.sources: List[Tuple[str, int, float]] = [
            (str(name), int(did), float(w)) for name, did, w in sources
        ]
        ids = [did for _, did, _ in self.sources]
        if len(set(ids)) != len(ids):
            raise ValueError(f"duplicate domain_id in sources: {ids!r}")
        names = [name for name, _, _ in self.sources]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate dataset_name in sources: {names!r}")
        weights = [w for _, _, w in self.sources]
        if any(w < 0 for w in weights):
            raise ValueError(f"negative weight in {weights!r}")
        if sum(weights) <= 0:
            raise ValueError(f"weights sum to {sum(weights)}; must be > 0")

        self.tokenizer = tokenizer
        self.action_chunk_len = int(action_chunk_len)
        self.shuffle_buffer_size = int(shuffle_buffer_size)
        self.train = bool(train)
        self.seed = int(seed)
        self.include_scene_dinov2 = bool(include_scene_dinov2)
        self.include_wrist_dinov2 = bool(include_wrist_dinov2)
        self.check_contract = bool(check_contract)
        self._contract_checked = False
        self._domain_to_name: Dict[int, str] = {did: name for name, did, _ in self.sources}
        self.image_tx = SiglipImageTransform(size=C.SIGLIP_IMAGE_SIZE, training=False)
        self.dinov2_image_tx = DINOv2ImageTransform(size=C.SIGLIP_IMAGE_SIZE)

    def _build_rlds(self):
        """Build the single tf.data graph: per-source → sample_from_datasets → shuffle → frame_transforms."""
        import tensorflow as tf
        import dlimp as dl
        from prismatic.vla.datasets.rlds.dataset import (
            apply_frame_transforms,
            apply_trajectory_transforms,
            make_dataset_from_rlds,
        )
        from prismatic.vla.datasets.rlds.oxe import get_oxe_dataset_kwargs_and_weights
        from prismatic.vla.datasets.rlds.utils.data_utils import NormalizationType

        image_size = C.SIGLIP_IMAGE_SIZE
        mixture_spec = [(name, weight) for name, _, weight in self.sources]
        per_dataset_kwargs, mixture_weights = get_oxe_dataset_kwargs_and_weights(
            self.data_dir,
            mixture_spec,
            load_camera_views=("primary", "wrist"),
            load_depth=False,
            load_proprio=True,
            load_language=True,
            action_proprio_normalization_type=NormalizationType.BOUNDS_Q99,
        )

        per_source_datasets = []
        for (_, domain_id, _), kwargs in zip(self.sources, per_dataset_kwargs):
            dk = dict(kwargs)
            ds_raw, _stats = make_dataset_from_rlds(**dk, train=self.train)
            chunk = self.action_chunk_len
            ds_raw = ds_raw.filter(
                lambda traj: tf.shape(traj["action"])[0] >= chunk
            )
            ds_t = apply_trajectory_transforms(
                ds_raw.repeat(),
                window_size=1,
                future_action_window_size=self.action_chunk_len - 1,
                skip_unlabeled=True,
                goal_relabeling_strategy="uniform",
                num_parallel_calls=1,
                train=self.train,
            ).flatten(num_parallel_calls=1)

            # Inject per-source domain_id into each frame. The default-arg
            # `did` captures the loop's domain_id at function-definition time
            # to avoid Python late-binding (every closure would otherwise see
            # the final iteration's id).
            did_const = tf.constant(int(domain_id), dtype=tf.int64)

            def _inject_domain(frame, did=did_const):
                frame = dict(frame)
                frame["domain_id"] = did
                return frame

            ds_t = ds_t.frame_map(_inject_domain)
            per_source_datasets.append(ds_t)

        dataset = dl.DLataset.sample_from_datasets(per_source_datasets, mixture_weights)
        dataset = dataset.shuffle(self.shuffle_buffer_size)
        dataset = apply_frame_transforms(
            dataset,
            resize_size=(image_size, image_size),
            num_parallel_calls=16,
            train=self.train,
        )
        dataset = dataset.with_ram_budget(1)
        return dataset

    @staticmethod
    def _hwc_uint8_to_chw_float01(img: np.ndarray) -> torch.Tensor:
        t = torch.from_numpy(img).permute(2, 0, 1).contiguous().float() / 255.0
        return t

    def _check_contract(self, raw: Dict, dataset_name: str) -> None:
        action_shape = np.asarray(raw["action"]).shape
        proprio_shape = np.asarray(raw["observation"]["proprio"][0]).shape
        if action_shape[-1] != C.ACTION_DIM:
            raise ValueError(
                f"[{dataset_name}] action dim {action_shape[-1]} != "
                f"expected {C.ACTION_DIM} (EEF_POS = delta XYZ + RPY + gripper)"
            )
        if proprio_shape[-1] > C.PROPRIO_DIM:
            raise ValueError(
                f"[{dataset_name}] proprio dim {proprio_shape[-1]} > "
                f"PROPRIO_DIM={C.PROPRIO_DIM} (single-arm OXE expected ≤8)"
            )
        if action_shape[0] != self.action_chunk_len:
            raise ValueError(
                f"[{dataset_name}] action chunk len {action_shape[0]} != "
                f"expected {self.action_chunk_len}"
            )
        for img_key in ("image_primary", "image_wrist"):
            img = np.asarray(raw["observation"][img_key][0])
            if img.dtype != np.uint8:
                raise ValueError(
                    f"[{dataset_name}] {img_key} dtype {img.dtype} != uint8"
                )
            if img.ndim != 3 or img.shape[-1] != 3:
                raise ValueError(
                    f"[{dataset_name}] {img_key} shape {img.shape} != (H,W,3)"
                )
            if img.shape[0] != C.SIGLIP_IMAGE_SIZE or img.shape[1] != C.SIGLIP_IMAGE_SIZE:
                raise ValueError(
                    f"[{dataset_name}] {img_key} resize mismatch: "
                    f"{img.shape[:2]} != ({C.SIGLIP_IMAGE_SIZE},{C.SIGLIP_IMAGE_SIZE})"
                )
        wrist_mask_arr = np.asarray(raw["observation"]["pad_mask_dict"]["image_wrist"])
        if wrist_mask_arr.dtype != np.bool_:
            raise ValueError(
                f"[{dataset_name}] pad_mask_dict[image_wrist] dtype "
                f"{wrist_mask_arr.dtype} != bool"
            )
        if wrist_mask_arr.shape != (1,):
            raise ValueError(
                f"[{dataset_name}] pad_mask_dict[image_wrist] shape "
                f"{wrist_mask_arr.shape} != (1,)"
            )

    def _to_batch_item(self, raw: Dict) -> Dict[str, torch.Tensor]:
        # domain_id is per-frame (injected per-source by _build_rlds), not
        # an instance attribute as in RLDSOxeDataset.
        domain_id = int(np.asarray(raw["domain_id"]).item())
        dataset_name = self._domain_to_name.get(domain_id, f"domain_{domain_id}")

        scene_img = raw["observation"]["image_primary"][0]
        wrist_img = raw["observation"]["image_wrist"][0]
        proprio = np.asarray(raw["observation"]["proprio"][0], dtype=np.float32)
        action_chunk = np.asarray(raw["action"], dtype=np.float32)

        if proprio.shape[-1] < C.PROPRIO_DIM:
            pad = np.zeros((C.PROPRIO_DIM - proprio.shape[-1],), dtype=np.float32)
            proprio = np.concatenate([proprio, pad], axis=-1)
        elif proprio.shape[-1] > C.PROPRIO_DIM:
            proprio = proprio[: C.PROPRIO_DIM]

        if self.check_contract and not self._contract_checked:
            self._check_contract(raw, dataset_name)
            self._contract_checked = True

        wrist_pad_mask = bool(raw["observation"]["pad_mask_dict"]["image_wrist"][0])

        scene_raw = self._hwc_uint8_to_chw_float01(scene_img)
        wrist_raw = self._hwc_uint8_to_chw_float01(wrist_img)
        scene = self.image_tx(scene_raw)
        wrist = self.image_tx(wrist_raw)

        lang_raw = raw["task"]["language_instruction"]
        if isinstance(lang_raw, bytes):
            language = lang_raw.decode("utf-8").strip()
        else:
            language = str(lang_raw).strip()
        prompt = self.tokenizer(language)

        target_action = torch.from_numpy(action_chunk).to(torch.float32)
        last_action = torch.zeros(self.action_chunk_len, C.ACTION_DIM, dtype=torch.float32)

        item = {
            "domain_id": torch.tensor(domain_id, dtype=torch.long),
            "scene_image": scene,
            "wrist_image": wrist,
            "prompt_input_ids": prompt["input_ids"],
            "prompt_attention_mask": prompt["attention_mask"],
            "proprio": torch.from_numpy(proprio),
            "last_action_chunk": last_action,
            "target_action": target_action,
            "action_mask": torch.ones(self.action_chunk_len, dtype=torch.bool),
            "wrist_mask": torch.tensor(wrist_pad_mask, dtype=torch.bool),
        }
        if self.include_wrist_dinov2:
            item["wrist_image_dinov2"] = self.dinov2_image_tx(wrist_raw)
        if self.include_scene_dinov2:
            item["scene_image_dinov2"] = self.dinov2_image_tx(scene_raw)
        return item

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        ds_tf = self._build_rlds()
        for rlds_batch in ds_tf.as_numpy_iterator():
            yield self._to_batch_item(rlds_batch)

    @staticmethod
    def collate_fn(samples):
        keys = samples[0].keys()
        out = {}
        for k in keys:
            v0 = samples[0][k]
            if torch.is_tensor(v0):
                out[k] = torch.stack([s[k] for s in samples], dim=0)
            else:
                out[k] = [s[k] for s in samples]
        return out
