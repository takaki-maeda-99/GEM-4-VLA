"""Offline test for LeRobotLiberoDataset's EE6D action_format (Plan 11 part B).

We stub LeRobotDataset to return ``observation.state`` chunks (shape
[len(offsets), 8]) when the wrapper requests state via
``delta_timestamps``. The proprio pos[0] component encodes the requested
offset so the test can verify the EE6D pack/unpack round trip recovers the
right anchor times.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import pytest
import torch
from torch.utils.data import DataLoader

from vla_project.data import constants as C
from vla_project.data.datasets import lerobot_libero_dataset as M
from vla_project.data.transforms.action_alignment import action20_to_ee_pose
from vla_project.data.transforms.language import GemmaPromptTokenizer


class _StubMeta:
    def __init__(self) -> None:
        self.tasks = {0: "pick the red block"}


class _StubTokenizer:
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
        out_ids = torch.zeros(len(text), L, dtype=torch.long)
        out_mask = torch.zeros(len(text), L, dtype=torch.long)
        for i, t in enumerate(text):
            out_mask[i, : min(len(t.split()), L)] = 1
        return {"input_ids": out_ids, "attention_mask": out_mask}


class _StubLeRobotDSEE6D:
    """LeRobot stub that respects delta_timestamps on observation.state.

    Encodes the requested offset into state[..., 0] and the frame index into
    state[..., 1] so tests can recover what was fetched. Identity quat at
    state[..., 3:7]; gripper=0.5 at state[..., 7].
    """

    def __init__(self, *args, delta_timestamps: Optional[Dict[str, List[float]]] = None,
                 **kwargs) -> None:
        self.meta = _StubMeta()
        self.delta_timestamps = delta_timestamps or {}

    def __len__(self) -> int:
        return 8

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        offsets = self.delta_timestamps.get("observation.state")
        if offsets is not None:
            T = len(offsets)
            state = torch.zeros(T, C.PROPRIO_DIM)
            for t, off in enumerate(offsets):
                state[t, 0] = float(off)
                state[t, 1] = float(idx)
                state[t, 3:7] = torch.tensor([0.0, 0.0, 0.0, 1.0])  # identity quat
                state[t, 7] = 0.5
        else:
            state = torch.zeros(C.PROPRIO_DIM)
            state[3:7] = torch.tensor([0.0, 0.0, 0.0, 1.0])
        return {
            "observation.images.image":       torch.rand(3, 256, 256),
            "observation.images.wrist_image": torch.rand(3, 256, 256),
            "observation.state":              state,
            "task_index":                     torch.tensor(0, dtype=torch.long),
            "frame_index":                    torch.tensor(idx, dtype=torch.long),
        }


@pytest.fixture
def stats_path_native(tmp_path: Path) -> Path:
    """Q99 stats file. Not used in ee6d mode but the dataset still loads it."""
    payload = {
        "libero_spatial_no_noops": {
            "action": {
                "q01":  [-1.0] * C.ACTION_DIM,
                "q99":  [ 1.0] * C.ACTION_DIM,
                "mask": [True, True, True, True, True, True, False],
            }
        }
    }
    p = tmp_path / "stats.json"
    p.write_text(json.dumps(payload))
    return p


def _make_dataset(stats_path: Path, *, mode: str, num_anchors: int = 30,
                  window_s: float = 4.0) -> M.LeRobotLiberoDataset:
    tok = GemmaPromptTokenizer(model_name=None, _tokenizer=_StubTokenizer())
    return M.LeRobotLiberoDataset(
        repo_id="lerobot/libero_spatial_image",
        stats_path=str(stats_path),
        unnorm_key="libero_spatial_no_noops",
        fps=10,
        tokenizer=tok,
        download_videos=False,
        action_format="ee6d",
        action_chunk_len=num_anchors,
        anchor_window_s=window_s,
        last_action_chunk_mode=mode,
        max_samples=4,
    )


def test_ee6d_zero_mode_shapes_and_padding(monkeypatch, stats_path_native: Path) -> None:
    monkeypatch.setattr(M, "_LeRobotDatasetCls", _StubLeRobotDSEE6D)
    N = 30
    ds = _make_dataset(stats_path_native, mode="zero", num_anchors=N)
    dl = DataLoader(ds, batch_size=2, collate_fn=M.LeRobotLiberoDataset.collate_fn)
    batch = next(iter(dl))
    assert batch["target_action"].shape == (2, N, 20)
    assert batch["last_action_chunk"].shape == (2, N, 20)
    # zero mode: past chunk is all zeros.
    assert torch.all(batch["last_action_chunk"] == 0.0)
    # bimanual padding [10:20] is zero in single-arm.
    assert torch.all(batch["target_action"][..., 10:20] == 0.0)
    assert batch["proprio"].shape == (2, C.PROPRIO_DIM)
    assert batch["action_mask"].shape == (2, N)
    assert batch["action_mask"].dtype == torch.bool


def test_ee6d_real_mode_recovers_anchor_offsets(monkeypatch, stats_path_native: Path) -> None:
    """Round-trip target_action through action20_to_ee_pose and check pos[..., 0]
    holds the requested anchor offsets (encoded by the stub)."""
    monkeypatch.setattr(M, "_LeRobotDatasetCls", _StubLeRobotDSEE6D)
    N = 30
    window = 4.0
    ds = _make_dataset(stats_path_native, mode="real", num_anchors=N, window_s=window)
    dl = DataLoader(ds, batch_size=1, collate_fn=M.LeRobotLiberoDataset.collate_fn)
    batch = next(iter(dl))
    pos, _, _ = action20_to_ee_pose(batch["target_action"][0])
    spacing = window / (N - 1)
    # Future anchors: offsets [0, sp, 2sp, ..., (N-1)*sp] = [0, ..., window].
    expected = torch.tensor([k * spacing for k in range(N)])
    assert torch.allclose(pos[:, 0], expected, atol=1e-4)
    # Past anchors (last_action_chunk): offsets [-N*sp, ..., -sp] (matches the
    # action-style convention: 2*N entries with future at the end).
    pos_past, _, _ = action20_to_ee_pose(batch["last_action_chunk"][0])
    expected_past = torch.tensor([(k - N) * spacing for k in range(N)])
    assert torch.allclose(pos_past[:, 0], expected_past, atol=1e-4)


def test_ee6d_proprio_is_t0_state(monkeypatch, stats_path_native: Path) -> None:
    """Current-state proprio fed to the model should be the t=0 anchor
    (first future), not some past timestep."""
    monkeypatch.setattr(M, "_LeRobotDatasetCls", _StubLeRobotDSEE6D)
    N = 30
    ds = _make_dataset(stats_path_native, mode="real", num_anchors=N)
    dl = DataLoader(ds, batch_size=1, collate_fn=M.LeRobotLiberoDataset.collate_fn)
    batch = next(iter(dl))
    # Stub encodes pos[0] = offset; t=0 anchor has offset = 0.
    assert abs(float(batch["proprio"][0, 0])) < 1e-6
    # gripper-qpos slot = 0.5 in the stub.
    assert abs(float(batch["proprio"][0, 7]) - 0.5) < 1e-6


def test_ee6d_rejects_invalid_window_or_anchor_count(stats_path_native: Path) -> None:
    tok = GemmaPromptTokenizer(model_name=None, _tokenizer=_StubTokenizer())
    with pytest.raises(ValueError):
        M.LeRobotLiberoDataset(
            repo_id="x", stats_path=str(stats_path_native),
            unnorm_key="libero_spatial_no_noops", fps=10, tokenizer=tok,
            download_videos=False, action_format="ee6d",
            action_chunk_len=1,  # < 2 → invalid
            anchor_window_s=4.0,
        )
    with pytest.raises(ValueError):
        M.LeRobotLiberoDataset(
            repo_id="x", stats_path=str(stats_path_native),
            unnorm_key="libero_spatial_no_noops", fps=10, tokenizer=tok,
            download_videos=False, action_format="ee6d",
            action_chunk_len=30, anchor_window_s=0.0,  # window must be > 0
        )


def test_ee6d_rejects_unknown_action_format(stats_path_native: Path) -> None:
    tok = GemmaPromptTokenizer(model_name=None, _tokenizer=_StubTokenizer())
    with pytest.raises(ValueError):
        M.LeRobotLiberoDataset(
            repo_id="x", stats_path=str(stats_path_native),
            unnorm_key="libero_spatial_no_noops", fps=10, tokenizer=tok,
            download_videos=False, action_format="bogus",
        )
