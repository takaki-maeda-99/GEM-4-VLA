"""Round-trip tests for save_checkpoint / load_checkpoint."""
import json
import subprocess
from pathlib import Path
from typing import Dict

import pytest
import torch
import torch.nn as nn

from vla_project.training.checkpoint import (
    load_checkpoint,
    save_checkpoint,
)


class _Toy(nn.Module):
    """Tiny module so checkpoint round-trip is fast and deterministic."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(3, 4)
        self.fc2 = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.relu(self.fc(x)))


def _state_dicts_equal(a: Dict[str, torch.Tensor], b: Dict[str, torch.Tensor]) -> bool:
    if set(a.keys()) != set(b.keys()):
        return False
    for k in a:
        if not torch.equal(a[k], b[k]):
            return False
    return True


def test_save_creates_files(tmp_path: Path) -> None:
    m = _Toy()
    cfg = {"train": {"lr": 1e-4, "batch_size": 1}, "model": {"num_blocks": 35}}
    out = tmp_path / "step_42"
    save_checkpoint(out, m, step=42, cfg=cfg)
    assert (out / "model.pt").is_file()
    assert (out / "meta.json").is_file()
    # No optimizer was passed, so optimizer.pt must not exist.
    assert not (out / "optimizer.pt").exists()


def test_load_restores_state_dict(tmp_path: Path) -> None:
    m1 = _Toy()
    m1.fc.weight.data.fill_(0.5)
    m1.fc2.weight.data.fill_(-0.25)
    out = tmp_path / "step_1"
    save_checkpoint(out, m1, step=1, cfg={"a": 1})

    m2 = _Toy()  # fresh random init
    assert not _state_dicts_equal(m1.state_dict(), m2.state_dict())
    meta = load_checkpoint(out, m2)
    assert _state_dicts_equal(m1.state_dict(), m2.state_dict())
    assert meta["step"] == 1
    assert meta["cfg"] == {"a": 1}


def test_save_records_norm_stats(tmp_path: Path) -> None:
    m = _Toy()
    norm_stats = {
        "libero_spatial_no_noops": {
            "action": {
                "q01": [-1.0] * 7,
                "q99": [ 1.0] * 7,
                "mask": [True, True, True, True, True, True, False],
            }
        }
    }
    out = tmp_path / "step_2"
    save_checkpoint(out, m, step=2, cfg={}, norm_stats=norm_stats)
    meta = json.loads((out / "meta.json").read_text())
    assert meta["norm_stats"] == norm_stats


def test_save_records_git_commit(tmp_path: Path) -> None:
    """git_commit should be a 40-char hex string OR 'unknown' if not in a repo."""
    m = _Toy()
    out = tmp_path / "step_3"
    save_checkpoint(out, m, step=3, cfg={})
    meta = json.loads((out / "meta.json").read_text())
    gc = meta["git_commit"]
    assert isinstance(gc, str)
    # Either 40-hex or 'unknown' or '<hex>-dirty'
    assert (len(gc) == 40 and all(c in "0123456789abcdef" for c in gc)) or gc == "unknown" or gc.endswith("-dirty")


def test_save_records_tokenizer_settings(tmp_path: Path) -> None:
    m = _Toy()
    out = tmp_path / "step_4"
    save_checkpoint(
        out, m, step=4, cfg={},
        tokenizer_settings={"model_name": "google/gemma-4-E2B", "max_len": 50},
    )
    meta = json.loads((out / "meta.json").read_text())
    assert meta["tokenizer_settings"] == {"model_name": "google/gemma-4-E2B", "max_len": 50}


def test_save_with_optimizer_round_trip(tmp_path: Path) -> None:
    m = _Toy()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    # Take one step so the optimizer has non-trivial state (Adam moments).
    m(torch.randn(2, 3)).sum().backward()
    opt.step()
    opt.zero_grad()
    state_before = opt.state_dict()

    out = tmp_path / "step_5"
    save_checkpoint(out, m, step=5, cfg={}, optimizer=opt)
    assert (out / "optimizer.pt").is_file()

    m2 = _Toy()
    opt2 = torch.optim.AdamW(m2.parameters(), lr=1e-3)
    load_checkpoint(out, m2, optimizer=opt2)
    state_after = opt2.state_dict()

    # Compare optimizer state — Adam stores `exp_avg`, `exp_avg_sq`, `step`.
    assert state_before["state"].keys() == state_after["state"].keys()
    for pid in state_before["state"]:
        for k, v in state_before["state"][pid].items():
            v2 = state_after["state"][pid][k]
            if torch.is_tensor(v):
                assert torch.equal(v, v2), f"opt state mismatch on {pid}/{k}"
            else:
                assert v == v2


def test_load_missing_dir_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_checkpoint(tmp_path / "does_not_exist", _Toy())


def test_load_missing_model_pt_raises(tmp_path: Path) -> None:
    out = tmp_path / "step_6"
    out.mkdir()
    (out / "meta.json").write_text("{}")  # only meta.json
    with pytest.raises(FileNotFoundError):
        load_checkpoint(out, _Toy())


def test_save_atomic_rename(tmp_path: Path) -> None:
    """The final out_dir name must not exist until the write completes."""
    m = _Toy()
    out = tmp_path / "step_7"
    save_checkpoint(out, m, step=7, cfg={})
    # After save, the dir exists and looks normal.
    assert out.is_dir()
    # Sibling temp dir from atomic rename must not linger (we check there's no
    # ``step_7.tmp`` or similar leftover sibling).
    siblings = [p.name for p in tmp_path.iterdir()]
    assert siblings == ["step_7"]


def test_save_overwrites_existing(tmp_path: Path) -> None:
    """Re-saving to the same dir must replace the previous checkpoint atomically."""
    out = tmp_path / "step_8"
    save_checkpoint(out, _Toy(), step=8, cfg={"v": 1})
    save_checkpoint(out, _Toy(), step=8, cfg={"v": 2})
    meta = json.loads((out / "meta.json").read_text())
    assert meta["cfg"] == {"v": 2}


def test_resolved_omegaconf_serializes(tmp_path: Path) -> None:
    """OmegaConf DictConfig must be serializable via OmegaConf.to_container()."""
    from omegaconf import OmegaConf
    cfg = OmegaConf.create({
        "train": {"lr": 1e-4, "batch_size": 1},
        "data": {"type": "libero_synthetic"},
    })
    out = tmp_path / "step_9"
    save_checkpoint(out, _Toy(), step=9, cfg=cfg)
    meta = json.loads((out / "meta.json").read_text())
    assert meta["cfg"] == {
        "train": {"lr": 1e-4, "batch_size": 1},
        "data": {"type": "libero_synthetic"},
    }
