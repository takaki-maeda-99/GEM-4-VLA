"""Offline test for LeRobotLiberoDataset.

We stub `lerobot.datasets.lerobot_dataset.LeRobotDataset` so the test runs
without network or HF cache access. The stub yields tensors with the same
shapes / dtypes the real loader returns. The test asserts the dataset
emits batches that satisfy `data/schema.py::validate_batch`.
"""
from pathlib import Path
from typing import Dict, List

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from vla_project.data import constants as C
from vla_project.data.schema import validate_batch
from vla_project.data.transforms.language import GemmaPromptTokenizer


class _StubMeta:
    def __init__(self) -> None:
        # Mimic the dict form the real loader's resolver supports.
        self.tasks = {0: "pick the red block"}


class _StubLeRobotDataset:
    def __init__(self, *_, **__) -> None:
        self.meta = _StubMeta()

    def __len__(self) -> int:
        return 4

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "observation.images.image":       torch.rand(3, 256, 256),
            "observation.images.wrist_image": torch.rand(3, 256, 256),
            "observation.state":              torch.randn(C.PROPRIO_DIM),
            "action":                         torch.randn(C.ACTION_CHUNK_LEN, C.ACTION_DIM),
            "task_index":                     torch.tensor(0, dtype=torch.long),
        }


class _StubTokenizer:
    """Avoids the network on `AutoTokenizer.from_pretrained`."""

    pad_token_id = 0
    eos_token = "<eos>"
    pad_token = "<pad>"

    def __call__(self, text, **kw):
        L = kw.get("max_length", C.DEFAULT_PROMPT_MAX_LEN)
        if isinstance(text, str):
            ids = torch.zeros(1, L, dtype=torch.long)
            mask = torch.zeros(1, L, dtype=torch.long)
            mask[0, : min(len(text.split()), L)] = 1
            return {"input_ids": ids, "attention_mask": mask}
        # batch
        out_ids = torch.zeros(len(text), L, dtype=torch.long)
        out_mask = torch.zeros(len(text), L, dtype=torch.long)
        for i, t in enumerate(text):
            out_mask[i, : min(len(t.split()), L)] = 1
        return {"input_ids": out_ids, "attention_mask": out_mask}


@pytest.fixture
def stats_path(tmp_path: Path) -> Path:
    import json
    payload = {
        "libero_spatial_no_noops": {
            "action": {
                "q01": [-1.0] * C.ACTION_DIM,
                "q99": [ 1.0] * C.ACTION_DIM,
                "mask": [True, True, True, True, True, True, False],
            }
        }
    }
    p = tmp_path / "stats.json"
    p.write_text(json.dumps(payload))
    return p


def test_yields_valid_batch(monkeypatch, stats_path: Path) -> None:
    from vla_project.data.datasets import lerobot_libero_dataset as M

    monkeypatch.setattr(M, "_LeRobotDatasetCls", _StubLeRobotDataset)

    tok = GemmaPromptTokenizer(model_name=None, _tokenizer=_StubTokenizer())
    ds = M.LeRobotLiberoDataset(
        repo_id="lerobot/libero_spatial_image",
        stats_path=str(stats_path),
        unnorm_key="libero_spatial_no_noops",
        fps=10,
        tokenizer=tok,
        episodes=[0],
        download_videos=False,
        domain_id=0,
        max_samples=4,
    )
    dl = DataLoader(ds, batch_size=2, collate_fn=M.LeRobotLiberoDataset.collate_fn)
    batch = next(iter(dl))
    validate_batch(batch)
    assert batch["domain_id"].shape == (2,)
    assert batch["scene_image"].shape == (2, 3, C.SIGLIP_IMAGE_SIZE, C.SIGLIP_IMAGE_SIZE)
    assert batch["wrist_image"].shape == (2, 3, C.SIGLIP_IMAGE_SIZE, C.SIGLIP_IMAGE_SIZE)
    assert batch["proprio"].shape == (2, C.PROPRIO_DIM)
    assert batch["last_action_chunk"].shape == (2, C.ACTION_CHUNK_LEN, C.ACTION_DIM)
    assert batch["target_action"].shape == (2, C.ACTION_CHUNK_LEN, C.ACTION_DIM)
    assert batch["action_mask"].shape == (2, C.ACTION_CHUNK_LEN)
    assert batch["prompt_input_ids"].shape == (2, C.DEFAULT_PROMPT_MAX_LEN)
    # Cold-start convention: last_action_chunk is zeros at training time.
    assert torch.all(batch["last_action_chunk"] == 0.0)
    # Target actions clipped to [-1, 1] (mask=True dims).
    assert batch["target_action"][:, :, :6].abs().max().item() <= 1.0 + 1e-6
