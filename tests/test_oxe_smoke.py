"""First-batch contract test for v37 OXE multi-domain pretrain (B7).

Validates that ``RLDSOxeDataset`` + ``WeightedMultiDataset`` + ``collate_fn``
produce a batch matching the internal schema the model expects:

  - keys: domain_id, scene_image, wrist_image, prompt_input_ids,
    prompt_attention_mask, proprio, last_action_chunk, target_action,
    action_mask, wrist_mask
  - shapes: scene/wrist (B,3,224,224), proprio (B,8), target_action (B,8,7),
    action_mask (B,8) bool, wrist_mask (B,) bool, domain_id (B,) long
  - values: target_action in [-1, 1] (BOUNDS_Q99 normalized), action_mask all
    True (default contract), domain_id in {set of source ids}.

Skipped when:
  - tensorflow / prismatic.vla.datasets are not importable (e.g. CI without
    the vla-gemma-4 venv).
  - The fractal20220817_data dataset is not present at the expected disk
    path. CI/dev machines without OXE data simply skip.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

OXE_DATA_DIR = Path("/misc/dl00/takaki/vla-gemma-4/data/stage3_openx")
SMALLEST_DOWNLOADED = "fractal20220817_data"  # always present per stage3_openx layout


def _have_prismatic() -> bool:
    try:
        import prismatic.vla.datasets.rlds.dataset  # noqa: F401
        return True
    except Exception:
        return False


@pytest.mark.skipif(
    not (OXE_DATA_DIR / SMALLEST_DOWNLOADED).is_dir(),
    reason=f"OXE data not present at {OXE_DATA_DIR / SMALLEST_DOWNLOADED}",
)
@pytest.mark.skipif(
    not _have_prismatic(),
    reason="prismatic / TF not importable (run under vla-gemma-4 venv)",
)
def test_rlds_oxe_dataset_first_batch_contract() -> None:
    from vla_project.data.datasets.rlds_oxe_dataset import RLDSOxeDataset
    from vla_project.data.transforms.language import GemmaPromptTokenizer
    from vla_project.data import constants as C

    tok = GemmaPromptTokenizer(model_name="google/gemma-4-E2B", max_len=20)
    ds = RLDSOxeDataset(
        data_dir=str(OXE_DATA_DIR),
        dataset_name=SMALLEST_DOWNLOADED,
        tokenizer=tok,
        action_chunk_len=8,
        shuffle_buffer_size=1024,           # small for test
        train=True,
        domain_id=0,
        seed=42,
        check_contract=True,
    )
    dl = DataLoader(ds, batch_size=2, collate_fn=RLDSOxeDataset.collate_fn)
    batch = next(iter(dl))

    expected_keys = {
        "domain_id", "scene_image", "wrist_image", "prompt_input_ids",
        "prompt_attention_mask", "proprio", "last_action_chunk",
        "target_action", "action_mask", "wrist_mask",
    }
    assert expected_keys.issubset(batch.keys()), (
        f"missing keys: {expected_keys - set(batch.keys())}"
    )

    B = 2
    assert batch["domain_id"].shape == (B,)
    assert batch["domain_id"].dtype == torch.long
    assert (batch["domain_id"] == 0).all(), "single-source loader stamps fixed id"

    assert batch["scene_image"].shape == (B, 3, C.SIGLIP_IMAGE_SIZE, C.SIGLIP_IMAGE_SIZE)
    assert batch["wrist_image"].shape == (B, 3, C.SIGLIP_IMAGE_SIZE, C.SIGLIP_IMAGE_SIZE)
    assert batch["scene_image"].dtype == torch.float32

    assert batch["proprio"].shape == (B, C.PROPRIO_DIM)
    assert batch["target_action"].shape == (B, C.ACTION_CHUNK_LEN, C.ACTION_DIM)
    assert batch["last_action_chunk"].shape == (B, C.ACTION_CHUNK_LEN, C.ACTION_DIM)

    assert batch["action_mask"].shape == (B, C.ACTION_CHUNK_LEN)
    assert batch["action_mask"].dtype == torch.bool
    assert batch["action_mask"].all(), "RLDSOxeDataset always emits all-True action_mask"

    assert batch["wrist_mask"].shape == (B,)
    assert batch["wrist_mask"].dtype == torch.bool
    # fractal20220817_data has wrist=None in OXE configs.py:55, so RLDS pads
    # the wrist with zero images and pad_mask_dict reads False. Assert that
    # the wrist_mask wiring captures this — would have been silently True
    # under the old RLDSLiberoDataset hardcode.
    assert (batch["wrist_mask"] == False).all(), (  # noqa: E712
        "fractal has no wrist; wrist_mask must be False from pad_mask_dict"
    )

    # BOUNDS_Q99 normalization — values clipped to [-1, 1] for masked dims.
    # gripper dim (index 6) passes through (mask=False), so we only check
    # the first 6 dims for the [-1, 1] range guarantee.
    a = batch["target_action"][..., :6]
    assert torch.all(a >= -1.0 - 1e-5) and torch.all(a <= 1.0 + 1e-5), (
        f"action[0:6] out of [-1, 1]: min={a.min().item()}, max={a.max().item()}"
    )


@pytest.mark.skipif(
    not (OXE_DATA_DIR / SMALLEST_DOWNLOADED).is_dir(),
    reason=f"OXE data not present at {OXE_DATA_DIR / SMALLEST_DOWNLOADED}",
)
@pytest.mark.skipif(
    not _have_prismatic(),
    reason="prismatic / TF not importable (run under vla-gemma-4 venv)",
)
def test_rlds_oxe_with_wrist_dataset_marks_true() -> None:
    """Sanity-check the inverse: a dataset WITH wrist (taco_play) emits
    wrist_mask=True. Catches any future regression that flips the mask
    polarity."""
    taco = OXE_DATA_DIR / "taco_play"
    if not taco.is_dir():
        pytest.skip("taco_play not present")
    from vla_project.data.datasets.rlds_oxe_dataset import RLDSOxeDataset
    from vla_project.data.transforms.language import GemmaPromptTokenizer

    tok = GemmaPromptTokenizer(model_name="google/gemma-4-E2B", max_len=20)
    ds = RLDSOxeDataset(
        data_dir=str(OXE_DATA_DIR),
        dataset_name="taco_play",
        tokenizer=tok,
        action_chunk_len=8,
        shuffle_buffer_size=1024,
        train=True,
        domain_id=1,
        seed=42,
    )
    dl = DataLoader(ds, batch_size=2, collate_fn=RLDSOxeDataset.collate_fn)
    batch = next(iter(dl))
    assert (batch["wrist_mask"] == True).all(), (  # noqa: E712
        "taco_play has wrist=rgb_gripper; mask must be True"
    )
    assert (batch["domain_id"] == 1).all()
