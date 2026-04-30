"""last_action_chunk_mode='real' fetches prior chunk + zero-pads at episode start."""
import json
from pathlib import Path
from typing import Dict
from unittest.mock import patch

import pytest
import torch
from torch.utils.data import DataLoader

from vla_project.data import constants as C
from vla_project.data.datasets import lerobot_libero_dataset as M
from vla_project.data.transforms.language import GemmaPromptTokenizer


class _StubMeta:
    def __init__(self) -> None:
        self.tasks = {0: "pick the red block"}


class _StubTokenizer:
    pad_token_id = 0
    eos_token = "<eos>"
    pad_token = "<pad>"
    padding_side = "right"

    def __call__(self, text, **kw):
        L = kw.get("max_length", C.DEFAULT_PROMPT_MAX_LEN)
        if isinstance(text, str):
            return {
                "input_ids": torch.zeros(1, L, dtype=torch.long),
                "attention_mask": torch.zeros(1, L, dtype=torch.long),
            }
        return {
            "input_ids": torch.zeros(len(text), L, dtype=torch.long),
            "attention_mask": torch.zeros(len(text), L, dtype=torch.long),
        }


def _make_stub_lerobot_dataset(action_len: int, frame_index_at: Dict[int, int]):
    """Factory: returns a stub class whose __getitem__ returns `action_len`-step
    action chunk + a per-index frame_index from `frame_index_at` (defaults to idx
    if missing)."""

    class _StubDataset:
        def __init__(self, *_, **kw) -> None:
            self.meta = _StubMeta()
            # Capture the requested delta_timestamps so the test can verify
            # the dataset asked for the right number of offsets.
            self.delta_timestamps = kw.get("delta_timestamps")

        def __len__(self) -> int:
            return 12

        def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
            # Action: the i-th past entry holds value `i`, current is action_len
            # so we can verify the past-vs-future split downstream.
            arange = torch.arange(action_len, dtype=torch.float32).unsqueeze(-1)
            action = arange.repeat(1, C.ACTION_DIM)
            return {
                "observation.images.image":       torch.rand(3, 256, 256),
                "observation.images.wrist_image": torch.rand(3, 256, 256),
                "observation.state":              torch.zeros(C.PROPRIO_DIM),
                "action":                         action,
                "task_index":                     torch.tensor(0, dtype=torch.long),
                "frame_index":                    torch.tensor(frame_index_at.get(idx, idx), dtype=torch.long),
            }

    return _StubDataset


@pytest.fixture
def stats_path(tmp_path: Path) -> Path:
    payload = {
        "libero_test": {
            "action": {
                "q01": [-100.0] * C.ACTION_DIM,
                "q99": [ 100.0] * C.ACTION_DIM,
                "mask": [True] * (C.ACTION_DIM - 1) + [False],
            }
        }
    }
    p = tmp_path / "stats.json"
    p.write_text(json.dumps(payload))
    return p


def test_zero_mode_default_unchanged(stats_path: Path) -> None:
    """Default mode='zero': 1H delta, last_action_chunk all zeros."""
    Stub = _make_stub_lerobot_dataset(action_len=C.ACTION_CHUNK_LEN, frame_index_at={})
    with patch.object(M, "_LeRobotDatasetCls", Stub):
        ds = M.LeRobotLiberoDataset(
            repo_id="x", stats_path=str(stats_path), unnorm_key="libero_test",
            fps=10, tokenizer=GemmaPromptTokenizer(model_name=None, _tokenizer=_StubTokenizer()),
            episodes=None, download_videos=False, domain_id=0, max_samples=2,
        )
        # Verify we asked for 1H offsets, not 2H
        assert len(ds.ds.delta_timestamps["action"]) == C.ACTION_CHUNK_LEN
        dl = DataLoader(ds, batch_size=1, collate_fn=M.LeRobotLiberoDataset.collate_fn)
        batch = next(iter(dl))
        assert torch.all(batch["last_action_chunk"] == 0.0)


def test_real_mode_uses_2h_delta(stats_path: Path) -> None:
    """mode='real' requests 2H entries spanning [-H, H-1]."""
    Stub = _make_stub_lerobot_dataset(action_len=2 * C.ACTION_CHUNK_LEN, frame_index_at={})
    with patch.object(M, "_LeRobotDatasetCls", Stub):
        ds = M.LeRobotLiberoDataset(
            repo_id="x", stats_path=str(stats_path), unnorm_key="libero_test",
            fps=10, tokenizer=GemmaPromptTokenizer(model_name=None, _tokenizer=_StubTokenizer()),
            episodes=None, download_videos=False, domain_id=0, max_samples=2,
            last_action_chunk_mode="real",
        )
        assert len(ds.ds.delta_timestamps["action"]) == 2 * C.ACTION_CHUNK_LEN


def test_real_mode_mid_episode_returns_real_past(stats_path: Path) -> None:
    """At frame_index >= H, last_action_chunk holds the actual past values."""
    H = C.ACTION_CHUNK_LEN
    # Pin frame_index=H so no clamping is expected.
    Stub = _make_stub_lerobot_dataset(action_len=2 * H, frame_index_at={i: H + i for i in range(12)})
    with patch.object(M, "_LeRobotDatasetCls", Stub):
        ds = M.LeRobotLiberoDataset(
            repo_id="x", stats_path=str(stats_path), unnorm_key="libero_test",
            fps=10, tokenizer=GemmaPromptTokenizer(model_name=None, _tokenizer=_StubTokenizer()),
            episodes=None, download_videos=False, domain_id=0, max_samples=2,
            last_action_chunk_mode="real",
        )
        dl = DataLoader(ds, batch_size=1, collate_fn=M.LeRobotLiberoDataset.collate_fn)
        batch = next(iter(dl))
        last = batch["last_action_chunk"][0]  # [H, A]
        target = batch["target_action"][0]    # [H, A]
        # Past values 0..H-1, future H..2H-1 (after BOUNDS_Q99 with q01=-100, q99=100,
        # so values << q99 stay roughly proportional after the (2x - q01) / span - 1 maps).
        # Simpler invariant: past[i] strictly less than target[0] for the first 6 dims
        # (which are mask=True). For the gripper dim (mask=False), passthrough preserves
        # the raw arange value.
        assert (last[:, 6] < target[0, 6]).all() or (last[:, 6] <= target[0, 6]).all()
        # Past is strictly different from zero (real values were fetched).
        assert (last.abs().sum(dim=-1) > 0).any()


def test_real_mode_zero_pads_at_episode_start(stats_path: Path) -> None:
    """At frame_index < H, the first (H - frame_index) past entries are zero-padded."""
    H = C.ACTION_CHUNK_LEN
    # Pin frame_index=2 so we expect H-2 = 6 entries to be zero-padded (the rest real).
    Stub = _make_stub_lerobot_dataset(action_len=2 * H, frame_index_at={i: 2 for i in range(12)})
    with patch.object(M, "_LeRobotDatasetCls", Stub):
        ds = M.LeRobotLiberoDataset(
            repo_id="x", stats_path=str(stats_path), unnorm_key="libero_test",
            fps=10, tokenizer=GemmaPromptTokenizer(model_name=None, _tokenizer=_StubTokenizer()),
            episodes=None, download_videos=False, domain_id=0, max_samples=2,
            last_action_chunk_mode="real",
        )
        dl = DataLoader(ds, batch_size=1, collate_fn=M.LeRobotLiberoDataset.collate_fn)
        batch = next(iter(dl))
        last = batch["last_action_chunk"][0]  # [H, A]
        # First 6 past entries zero, last 2 not zero (after normalization).
        assert torch.all(last[:H - 2] == 0.0)
        # Position H-2 onwards has real (non-zero) data.
        assert (last[H - 2:].abs().sum() > 0).item()


def test_invalid_mode_raises(stats_path: Path) -> None:
    Stub = _make_stub_lerobot_dataset(action_len=C.ACTION_CHUNK_LEN, frame_index_at={})
    with patch.object(M, "_LeRobotDatasetCls", Stub):
        with pytest.raises(ValueError):
            M.LeRobotLiberoDataset(
                repo_id="x", stats_path=str(stats_path), unnorm_key="libero_test",
                fps=10, tokenizer=GemmaPromptTokenizer(model_name=None, _tokenizer=_StubTokenizer()),
                episodes=None, download_videos=False, domain_id=0,
                last_action_chunk_mode="bogus",
            )
