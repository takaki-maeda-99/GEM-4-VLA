"""Single-graph multi-domain RLDS loader (v40 redesign).

Replaces the v37/v39 ``WeightedMultiDataset`` pattern that wrapped N
independent ``RLDSOxeDataset`` / ``RLDSLiberoDataset`` children. That
pattern eagerly constructed N RLDS pipelines per worker (and per rank),
which on dl50's 16-CPU NUMA0 saturated tf.data's AUTOTUNE thread
allocator and prevented the 13-source v39 pretrain mix from ever
reaching its first training step.

This loader builds a SINGLE RLDS graph for all sources via a minor
re-implementation of prismatic's ``make_interleaved_dataset``. Each child
dataset is mapped with a constant ``_domain_id`` field before
interleaving, so per-sample DA-row routing is preserved without an
extra Python-side lookup. Thread budgets (``traj_transform_threads``,
``traj_read_threads``, ``frame_transform_threads``) are exposed at
config so they can be tuned to the host CPU count.

LIBERO suites must be reachable from the OXE registry root (we symlink
``modified_libero_rlds/<suite>`` into ``stage3_openx/<suite>``); their
metadata is already in ``OXE_DATASET_CONFIGS``
(``src/prismatic/vla/datasets/rlds/oxe/configs.py:657-684``).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Union

import numpy as np
import torch
from torch.utils.data import IterableDataset

from vla_project.data import constants as C
from vla_project.data.transforms.image import DINOv2ImageTransform, SiglipImageTransform
from vla_project.data.transforms.language import GemmaPromptTokenizer


class SourceSpec:
    """Per-source spec consumed by RLDSInterleavedMultidomain.

    Mirrors the field shape of the YAML ``data.sources`` entries. We
    deliberately don't accept a per-source ``data_dir`` override here —
    ``make_interleaved_dataset`` requires a single root, and we resolve
    LIBERO suites by symlinking them into the OXE root.
    """

    __slots__ = ("dataset_name", "domain_id", "weight")

    def __init__(self, dataset_name: str, domain_id: int, weight: float = 1.0) -> None:
        self.dataset_name = str(dataset_name)
        self.domain_id = int(domain_id)
        self.weight = float(weight)


class RLDSInterleavedMultidomain(IterableDataset):
    """Yield per-frame Batch items from N OXE-registry datasets in one graph.

    Args:
        data_dir: OXE root containing all sources (LIBERO suites must be
            symlinked in if they live elsewhere).
        sources: list of SourceSpec or dicts with ``dataset_name``,
            ``domain_id``, ``weight``. All ``dataset_name`` values must
            be present in ``OXE_DATASET_CONFIGS``.
        tokenizer: shared prompt tokenizer (one instance, not per-source).
        action_chunk_len: target action chunk length (default from
            constants).
        shuffle_buffer_size: post-interleave shuffle buffer (in frames).
        train: pass-through to RLDS.
        seed: per-source seed offset is implicit via interleave ordering;
            we don't rely on the dlimp seed= argument.
        traj_transform_threads / traj_read_threads / frame_transform_threads:
            global thread budgets (allocated proportionally to weights
            inside the RLDS graph). Override the AUTOTUNE default that
            otherwise spawns 60+ threads per worker on this host.
        include_scene_dinov2 / include_wrist_dinov2: optional extra
            channels.
    """

    def __init__(
        self,
        data_dir: Union[str, Path],
        sources: Sequence[Union[SourceSpec, Dict[str, Any]]],
        tokenizer: GemmaPromptTokenizer,
        action_chunk_len: int = C.ACTION_CHUNK_LEN,
        shuffle_buffer_size: int = 65536,
        train: bool = True,
        seed: int = 42,
        traj_transform_threads: int = 13,
        traj_read_threads: int = 13,
        frame_transform_threads: int = 16,
        include_scene_dinov2: bool = False,
        include_wrist_dinov2: bool = False,
    ) -> None:
        super().__init__()
        if not sources:
            raise ValueError("sources is empty")
        # Normalize source list to SourceSpec
        norm: List[SourceSpec] = []
        for src in sources:
            if isinstance(src, SourceSpec):
                norm.append(src)
            elif isinstance(src, dict):
                norm.append(SourceSpec(
                    dataset_name=src["dataset_name"],
                    domain_id=int(src["domain_id"]),
                    weight=float(src.get("weight", 1.0)),
                ))
            else:
                # OmegaConf DictConfig falls through here; pull via attr access
                norm.append(SourceSpec(
                    dataset_name=getattr(src, "dataset_name"),
                    domain_id=int(getattr(src, "domain_id")),
                    weight=float(getattr(src, "weight", 1.0)),
                ))
        names = [s.dataset_name for s in norm]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate dataset_name: {names!r}")
        self.data_dir = str(data_dir)
        self.sources: List[SourceSpec] = norm
        self.tokenizer = tokenizer
        self.action_chunk_len = int(action_chunk_len)
        self.shuffle_buffer_size = int(shuffle_buffer_size)
        self.train = bool(train)
        self.seed = int(seed)
        self.traj_transform_threads = int(traj_transform_threads)
        self.traj_read_threads = int(traj_read_threads)
        self.frame_transform_threads = int(frame_transform_threads)
        self.include_scene_dinov2 = bool(include_scene_dinov2)
        self.include_wrist_dinov2 = bool(include_wrist_dinov2)
        self.image_tx = SiglipImageTransform(size=C.SIGLIP_IMAGE_SIZE, training=False)
        self.dinov2_image_tx = DINOv2ImageTransform(size=C.SIGLIP_IMAGE_SIZE)

    def _build_rlds(self):
        # Local TF imports — only used inside the iterator-construction path.
        import tensorflow as tf
        import dlimp as dl
        from prismatic.vla.datasets.rlds.dataset import (
            apply_frame_transforms,
            apply_trajectory_transforms,
            make_dataset_from_rlds,
        )
        from prismatic.vla.datasets.rlds.utils.data_utils import (
            allocate_threads,
            NormalizationType,
        )
        from prismatic.vla.datasets.rlds.oxe import get_oxe_dataset_kwargs_and_weights

        IMAGE_SIZE = C.SIGLIP_IMAGE_SIZE

        mixture_spec = [(s.dataset_name, s.weight) for s in self.sources]
        per_dataset_kwargs, weights = get_oxe_dataset_kwargs_and_weights(
            self.data_dir,
            mixture_spec,
            load_camera_views=("primary", "wrist"),
            load_depth=False,
            load_proprio=True,
            load_language=True,
            action_proprio_normalization_type=NormalizationType.BOUNDS_Q99,
        )
        # Reorder name_to_domain in registry order (some sources may have
        # been filtered/re-ordered by get_oxe_dataset_kwargs_and_weights).
        name_to_domain = {s.dataset_name: s.domain_id for s in self.sources}

        # Allocate global thread budgets across sources by weight (mirrors
        # ``make_interleaved_dataset`` lines 558-560).
        threads_per = allocate_threads(self.traj_transform_threads, np.asarray(weights))
        reads_per = allocate_threads(self.traj_read_threads, np.asarray(weights))

        children = []
        domain_ids_per_child: List[int] = []
        for dataset_kwargs, threads, reads in zip(per_dataset_kwargs, threads_per, reads_per):
            name = dataset_kwargs["name"]
            domain_id = name_to_domain[name]
            domain_ids_per_child.append(domain_id)

            dk = dict(dataset_kwargs)
            ds_raw, _stats = make_dataset_from_rlds(
                **dk,
                train=self.train,
                num_parallel_calls=int(threads),
                num_parallel_reads=int(reads),
            )

            # Filter trajectories shorter than action_chunk_len (matches
            # rlds_oxe_dataset.py:158-160 — OXE datasets have variable
            # length, LIBERO is uniform 50-200).
            _chunk_len = self.action_chunk_len
            ds_raw = ds_raw.filter(
                lambda traj: tf.shape(traj["action"])[0] >= _chunk_len
            )

            ds_t = apply_trajectory_transforms(
                ds_raw.repeat(),
                window_size=1,
                future_action_window_size=self.action_chunk_len - 1,
                skip_unlabeled=True,
                # MUST be None: "uniform" duplicates every obs image into
                # task["image_*"], doubling JPEG payload (primary+wrist × 2
                # = 4 images per element). OXE wrapper already disables it
                # (rlds_oxe_dataset.py:165); LIBERO wrapper retained
                # "uniform" by accident (rlds_libero_dataset.py:194).
                goal_relabeling_strategy=None,
                num_parallel_calls=int(threads),
                train=self.train,
            ).flatten(num_parallel_calls=int(threads))

            # Inject constant domain_id into each frame BEFORE interleave.
            # This is the v40 mechanism that lets us recover per-sample
            # routing after sample_from_datasets shuffles children. We use
            # ``_domain_id`` (underscore prefix) to mark it as our internal
            # field, separate from any RLDS native fields.
            _did = tf.constant(int(domain_id), dtype=tf.int64)

            def _add_domain_id(frame, _did=_did):
                # Note: avoid {**frame, ...} — RLDS frames are nested dicts
                # which dict-spread doesn't deep-copy. Use a flat update on
                # a shallow copy.
                out = dict(frame)
                out["_domain_id"] = _did
                return out

            ds_t = ds_t.map(_add_domain_id, num_parallel_calls=int(threads))
            children.append(ds_t)

        # Interleave at the frame level. ``sample_from_datasets`` does NOT
        # expose the choice index in its output, so per-sample domain_id
        # MUST come from the constant injected above.
        ds: dl.DLataset = dl.DLataset.sample_from_datasets(
            children, np.asarray(weights, dtype=np.float64) / float(np.sum(weights))
        )

        # Shuffle once after interleave (matches make_interleaved_dataset).
        ds = ds.shuffle(self.shuffle_buffer_size)

        # Apply frame transforms (decode + resize). 16 parallel calls is
        # the prismatic default; we expose it as a config knob.
        ds = apply_frame_transforms(
            ds,
            resize_size=(IMAGE_SIZE, IMAGE_SIZE),
            num_parallel_calls=int(self.frame_transform_threads),
            train=self.train,
        )

        # Cap tf.data autotune RAM (matches rlds_oxe_dataset.py:194).
        ds = ds.with_ram_budget(1)
        return ds

    @staticmethod
    def _hwc_uint8_to_chw_float01(img: np.ndarray) -> torch.Tensor:
        t = torch.from_numpy(img).permute(2, 0, 1).contiguous().float() / 255.0
        return t

    def _to_batch_item(self, raw: Dict) -> Dict[str, torch.Tensor]:
        scene_img = raw["observation"]["image_primary"][0]
        wrist_img = raw["observation"]["image_wrist"][0]
        proprio = np.asarray(raw["observation"]["proprio"][0], dtype=np.float32)
        action_chunk = np.asarray(raw["action"], dtype=np.float32)

        # Right-pad proprio with zeros to PROPRIO_DIM. Several OXE configs
        # yield <8-dim proprio (e.g. stanford_hydra = 7-dim, libero =
        # EEF_state+gripper variants); without padding the collate stack
        # fails on a mixed batch. Mirrors rlds_oxe_dataset.py:279-283.
        if proprio.shape[-1] < C.PROPRIO_DIM:
            pad = np.zeros((C.PROPRIO_DIM - proprio.shape[-1],), dtype=np.float32)
            proprio = np.concatenate([proprio, pad], axis=-1)
        elif proprio.shape[-1] > C.PROPRIO_DIM:
            proprio = proprio[: C.PROPRIO_DIM]

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

        # Wrist mask from RLDS pad_mask_dict (matches rlds_oxe_dataset.py:295).
        # OXE datasets without wrist (fractal20220817_data, kuka,
        # nyu_franka_play, bridge_orig) emit pad_mask_dict["image_wrist"] =
        # False; LIBERO suites always have wrist so mask=True.
        try:
            wm = bool(raw["observation"]["pad_mask_dict"]["image_wrist"][0])
        except (KeyError, IndexError, TypeError):
            wm = True
        # Per-sample domain_id was injected into the TF graph as
        # ``_domain_id`` constant (see _build_rlds).
        domain_id_val = int(raw["_domain_id"])

        item = {
            "domain_id": torch.tensor(domain_id_val, dtype=torch.long),
            "scene_image": scene,
            "wrist_image": wrist,
            "prompt_input_ids": prompt["input_ids"],
            "prompt_attention_mask": prompt["attention_mask"],
            "proprio": torch.from_numpy(proprio),
            "last_action_chunk": last_action,
            "target_action": target_action,
            "action_mask": torch.ones(self.action_chunk_len, dtype=torch.bool),
            "wrist_mask": torch.tensor(wm, dtype=torch.bool),
        }
        if self.include_wrist_dinov2:
            item["wrist_image_dinov2"] = self.dinov2_image_tx(wrist_raw)
        if self.include_scene_dinov2:
            item["scene_image_dinov2"] = self.dinov2_image_tx(scene_raw)
        return item

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        ds = self._build_rlds()
        for raw in ds.as_numpy_iterator():
            yield self._to_batch_item(raw)

    @staticmethod
    def collate_fn(samples):
        """Same shape contract as RLDSOxeDataset.collate_fn / RLDSLiberoDataset.collate_fn."""
        from vla_project.data.datasets.rlds_oxe_dataset import RLDSOxeDataset
        return RLDSOxeDataset.collate_fn(samples)
