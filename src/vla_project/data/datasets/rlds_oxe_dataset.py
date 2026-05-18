"""Generic OXE single-arm RLDS loader for the v37 multi-domain pretrain.

Adapted from ``rlds_libero_dataset.py``. Differences:

  - ``dataset_name`` is REQUIRED (no LIBERO default). Caller must specify
    one OXE EEF_POS dataset registered in
    ``VLA-Adapter/prismatic/vla/datasets/rlds/oxe/configs.py``.
  - ``wrist_mask`` is derived per-sample from RLDS
    ``pad_mask_dict["image_wrist"]`` instead of being stamped True. For
    OXE datasets without a wrist camera (fractal20220817_data, kuka,
    nyu_franka_play, bridge_orig in our v37 mixture), RLDS substitutes a
    zero image (see VLA-Adapter/prismatic/vla/datasets/rlds/obs_transforms.py:65)
    and the pad mask reads False. The model's wrist_in_llm path zero-gates
    the LLM wrist slot when wrist_mask=False
    (vla_policy.py:712-723), matching the v36 π₀ "missing view"
    convention without needing extra Python branching.
  - First-sample action/proprio dim contract check (see ``_check_contract``):
    ``target_action.shape[-1] == 7`` and ``proprio.shape[-1] == 8``,
    matching ``ACTION_DIM`` and ``PROPRIO_DIM`` in
    ``data/constants.py``. OXE EEF_POS canonicalizes to 7-dim
    (delta XYZ + RPY + gripper) and StateEncoding standardizes proprio
    to 8-dim — see VLA-Adapter/prismatic/vla/datasets/rlds/oxe/configs.py:43.

Requires a venv with tensorflow + dlimp installed; the GEM-4-VLA
``.venv`` (built via ``uv sync``) covers this. prismatic itself is
vendored under ``src/prismatic/`` so PYTHONPATH=src/ is sufficient.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Union

import numpy as np
import torch
from torch.utils.data import IterableDataset

from vla_project.data import constants as C
from vla_project.data.transforms.image import DINOv2ImageTransform, SiglipImageTransform
from vla_project.data.transforms.language import GemmaPromptTokenizer


class RLDSOxeDataset(IterableDataset):
    """Yield internal Batch items from a single OXE EEF_POS dataset."""

    def __init__(
        self,
        data_dir: Union[str, Path],
        dataset_name: str,
        tokenizer: GemmaPromptTokenizer,
        action_chunk_len: int = C.ACTION_CHUNK_LEN,
        shuffle_buffer_size: int = 65536,
        train: bool = True,
        domain_id: int = 0,
        seed: int = 42,
        include_scene_dinov2: bool = False,
        include_wrist_dinov2: bool = False,
        shared_stats: Optional[Union[Dict[str, Any], str, Path]] = None,
        check_contract: bool = True,
    ) -> None:
        super().__init__()
        if not dataset_name:
            raise ValueError("dataset_name is required (OXE registry name)")
        self.data_dir = str(data_dir)
        self.dataset_name = dataset_name
        self.tokenizer = tokenizer
        self.action_chunk_len = action_chunk_len
        self.shuffle_buffer_size = shuffle_buffer_size
        self.train = train
        self.domain_id = int(domain_id)
        self.seed = seed
        self.include_scene_dinov2 = bool(include_scene_dinov2)
        self.include_wrist_dinov2 = bool(include_wrist_dinov2)
        self.check_contract = bool(check_contract)
        self._contract_checked = False
        self._shared_stats: Optional[Dict[str, Any]] = (
            self._resolve_shared_stats(shared_stats, dataset_name)
            if shared_stats is not None
            else None
        )
        self.image_tx = SiglipImageTransform(size=C.SIGLIP_IMAGE_SIZE, training=False)
        self.dinov2_image_tx = DINOv2ImageTransform(size=C.SIGLIP_IMAGE_SIZE)

    @staticmethod
    def _resolve_shared_stats(
        shared_stats: Union[Dict[str, Any], str, Path],
        dataset_name: str,
    ) -> Dict[str, Any]:
        """Same accept-shapes as RLDSLiberoDataset: unwrapped {action,proprio,...},
        single-key wrapper, or path to a JSON file containing either form."""
        if isinstance(shared_stats, (str, Path)):
            with open(str(shared_stats), "r") as f:
                payload = json.load(f)
        elif isinstance(shared_stats, dict):
            payload = shared_stats
        else:
            raise TypeError(
                f"shared_stats must be dict | str | Path, got {type(shared_stats).__name__}"
            )
        if "action" in payload and "proprio" in payload:
            return payload
        inner_keys = list(payload.keys())
        if len(inner_keys) == 0:
            raise ValueError("shared_stats payload is empty")
        if len(inner_keys) > 1:
            if dataset_name in payload:
                return payload[dataset_name]
            raise ValueError(
                f"shared_stats wrapper has multiple keys {inner_keys!r} and "
                f"none match dataset_name={dataset_name!r}; pass an unwrapped "
                f"dict or a single-key wrapper."
            )
        return payload[inner_keys[0]]

    def _build_rlds(self):
        from prismatic.vla.datasets.rlds.dataset import (
            apply_frame_transforms,
            apply_trajectory_transforms,
            make_dataset_from_rlds,
            make_interleaved_dataset,
        )
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

        # Single-source loader: bypass make_interleaved_dataset (which
        # complicates dataset_statistics override). This matches
        # rlds_libero_dataset.py:147.
        if self._shared_stats is not None:
            dk = dict(per_dataset_kwargs[0])
            dk.pop("dataset_statistics", None)
            ds_raw, _stats = make_dataset_from_rlds(
                **dk, train=self.train, dataset_statistics=self._shared_stats,
            )
        else:
            dk = dict(per_dataset_kwargs[0])
            ds_raw, _stats = make_dataset_from_rlds(**dk, train=self.train)

        # Filter trajectories shorter than action_chunk_len. Without this,
        # apply_trajectory_transforms's chunker produces a range op with
        # limit = T - (chunk_len - 1) < 0 when T < chunk_len, raising
        # "InvalidArgumentError: Requires start <= limit when delta > 0".
        # OXE datasets have variable trajectory lengths; LIBERO has uniform
        # 50-200 steps so this filter is a no-op there.
        import tensorflow as tf  # local import — only used inside _build_rlds
        _chunk_len = self.action_chunk_len
        ds_raw = ds_raw.filter(
            lambda traj: tf.shape(traj["action"])[0] >= _chunk_len
        )
        # ``goal_relabeling_strategy=None``: skip the "uniform" relabeling
        # because we are not a goal-conditioned policy. With "uniform" set,
        # ``goal_relabeling.py`` copies every observation image into
        # ``task["image_*"]``, which doubles the encoded JPEG payload per
        # element (primary+wrist × 2 = 4 images). This payload is what was
        # blowing up rank-0 host RAM through the long-running shuffle —
        # removing it cuts the per-element baseline roughly in half.
        ds_t = apply_trajectory_transforms(
            ds_raw.repeat(),
            window_size=1,
            future_action_window_size=self.action_chunk_len - 1,
            skip_unlabeled=True,
            goal_relabeling_strategy=None,
            num_parallel_calls=1,
            train=self.train,
        ).flatten(num_parallel_calls=1)
        # Decode + resize FIRST, then shuffle uniform-byte uint8 frames.
        # Old order (shuffle → frame_transforms) holds variable-size encoded
        # JPEGs in the shuffle buffer: as samples cycle through, the resident
        # bytes drift toward the worst-case max element size, producing the
        # observed ~12 MB/step host-RAM creep that survived ``with_ram_budget(1)``
        # and ``MALLOC_ARENA_MAX=2`` (gdb-malloc_trim showed 99.9% of resident
        # bytes were genuinely live data, not glibc fragmentation). Frames
        # post-decode are 224×224×3 uint8 (primary + wrist) ≈ 300 KB each,
        # uniform across the dataset, so the shuffle buffer's resident byte
        # count is bounded at ``shuffle_buffer_size × 300 KB`` instead.
        dataset = apply_frame_transforms(
            ds_t,
            resize_size=(IMAGE_SIZE, IMAGE_SIZE),
            num_parallel_calls=16,
            train=self.train,
        )
        dataset = dataset.shuffle(self.shuffle_buffer_size)
        # Cap tf.data autotune RAM (covers prefetch / interleave). Does not
        # constrain the explicit shuffle(N) buffer above — that's bounded by
        # the now-uniform per-element size, not by autotune.
        dataset = dataset.with_ram_budget(1)
        return dataset

    @staticmethod
    def _hwc_uint8_to_chw_float01(img: np.ndarray) -> torch.Tensor:
        t = torch.from_numpy(img).permute(2, 0, 1).contiguous().float() / 255.0
        return t

    def _check_contract(self, raw: Dict) -> None:
        """First-sample contract check (cheap, run only on first iteration):
        - action_chunk_len match, action dim 7, proprio dim 8 (OXE EEF_POS +
          OXE-standardized 8-dim state).
        - image rank/dtype: HWC uint8 with 3 channels. Catches mis-decoded
          images (e.g., depth leaking into image_*) before they hit
          ``_hwc_uint8_to_chw_float01``'s permute(2,0,1) and produce silent
          channel-mixing.
        - wrist_mask scalar bool after window-axis indexing.
        """
        action_shape = np.asarray(raw["action"]).shape
        # _to_batch_item runs proprio padding BEFORE this check, so we relax
        # the strict ==PROPRIO_DIM check to <=PROPRIO_DIM (the wrapper
        # zero-pads up to 8; >8 should never happen for OXE single-arm but
        # is rejected defensively).
        proprio_shape = np.asarray(raw["observation"]["proprio"][0]).shape
        if action_shape[-1] != C.ACTION_DIM:
            raise ValueError(
                f"[{self.dataset_name}] action dim {action_shape[-1]} != "
                f"expected {C.ACTION_DIM} (EEF_POS = delta XYZ + RPY + gripper)"
            )
        if proprio_shape[-1] > C.PROPRIO_DIM:
            raise ValueError(
                f"[{self.dataset_name}] proprio dim {proprio_shape[-1]} > "
                f"PROPRIO_DIM={C.PROPRIO_DIM} (single-arm OXE expected ≤8)"
            )
        if action_shape[0] != self.action_chunk_len:
            raise ValueError(
                f"[{self.dataset_name}] action chunk len {action_shape[0]} != "
                f"expected {self.action_chunk_len}"
            )
        for img_key in ("image_primary", "image_wrist"):
            img = np.asarray(raw["observation"][img_key][0])
            if img.dtype != np.uint8:
                raise ValueError(
                    f"[{self.dataset_name}] {img_key} dtype {img.dtype} != uint8 "
                    f"(RLDS obs_transforms decode produces uint8 HWC)"
                )
            if img.ndim != 3 or img.shape[-1] != 3:
                raise ValueError(
                    f"[{self.dataset_name}] {img_key} shape {img.shape} != (H,W,3)"
                )
            if img.shape[0] != C.SIGLIP_IMAGE_SIZE or img.shape[1] != C.SIGLIP_IMAGE_SIZE:
                raise ValueError(
                    f"[{self.dataset_name}] {img_key} resize mismatch: "
                    f"{img.shape[:2]} != ({C.SIGLIP_IMAGE_SIZE},{C.SIGLIP_IMAGE_SIZE})"
                )
        wrist_mask_arr = np.asarray(raw["observation"]["pad_mask_dict"]["image_wrist"])
        if wrist_mask_arr.dtype != np.bool_:
            raise ValueError(
                f"[{self.dataset_name}] pad_mask_dict[image_wrist] dtype "
                f"{wrist_mask_arr.dtype} != bool"
            )
        if wrist_mask_arr.shape != (1,):
            raise ValueError(
                f"[{self.dataset_name}] pad_mask_dict[image_wrist] shape "
                f"{wrist_mask_arr.shape} != (1,) (window_size=1)"
            )

    def _to_batch_item(self, raw: Dict) -> Dict[str, torch.Tensor]:
        scene_img = raw["observation"]["image_primary"][0]   # (H, W, 3) uint8
        wrist_img = raw["observation"]["image_wrist"][0]     # zero-pad if missing
        proprio = np.asarray(raw["observation"]["proprio"][0], dtype=np.float32)
        action_chunk = np.asarray(raw["action"], dtype=np.float32)                # (H, 7)

        # Right-pad proprio with zeros to PROPRIO_DIM. OXE state_encoding
        # nominally normalizes to 8-dim (POS_EULER = 3+3+1+1, POS_QUAT = 3+4+1,
        # JOINT = 7+1) but several OXE configs.py entries omit padding entries
        # in state_obs_keys and yield 7-dim (e.g. stanford_hydra emits
        # [EEF_state(6), gripper_state(1)] = 7 instead of [EEF_state(6),
        # PAD(1), gripper_state(1)] = 8). Zero-pad keeps PROPRIO_DIM constant
        # across the batch; the per-domain DA proprio_proj absorbs the
        # constant-zero column without harm. Truncate (not pad) only if
        # somehow >8 — unexpected but defensive.
        if proprio.shape[-1] < C.PROPRIO_DIM:
            pad = np.zeros((C.PROPRIO_DIM - proprio.shape[-1],), dtype=np.float32)
            proprio = np.concatenate([proprio, pad], axis=-1)
        elif proprio.shape[-1] > C.PROPRIO_DIM:
            proprio = proprio[: C.PROPRIO_DIM]

        if self.check_contract and not self._contract_checked:
            self._check_contract(raw)
            self._contract_checked = True

        # wrist_mask from RLDS pad_mask_dict (True=real wrist, False=padded zero).
        # add_pad_mask_dict at traj level (traj_transforms.py:70) sets
        # pad_mask_dict[image_wrist] = (string-length != 0) for image keys, so
        # padded entries (which are stored as empty strings before decoding) read
        # False. The string-vs-decoded-image timing is irrelevant by the time we
        # iterate: the bool mask was computed before image decoding.
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
            "domain_id": torch.tensor(self.domain_id, dtype=torch.long),
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
