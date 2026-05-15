"""ModelRuntime carries is_local after _resolve_ckpt_dir.

This test does NOT load a real model — it just exercises _resolve_ckpt_dir
return type. The is_local=False (HF) branch is covered indirectly by the
inference_server integration test (Task 8) when a real HF ckpt is pulled.
"""
from __future__ import annotations

import pytest

from vla_project.deployment.runtime import _resolve_ckpt_dir


def test_resolve_local_existing_returns_local_true(tmp_path):
    (tmp_path / "meta.json").write_text("{}")
    resolved, is_local = _resolve_ckpt_dir(tmp_path)
    assert resolved == tmp_path
    assert is_local is True


def test_resolve_absolute_missing_raises(tmp_path):
    ghost = tmp_path / "no_such_dir"
    with pytest.raises(FileNotFoundError):
        _resolve_ckpt_dir(ghost)


def test_resolve_relative_with_dotdot_rejected():
    with pytest.raises(FileNotFoundError):
        _resolve_ckpt_dir("foo/../bar")
