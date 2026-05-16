"""post_process loader: file presence + trust gating cases per spec §6.

Six cases:
(a) local + valid file       → callable
(b) local + no file          → None
(c) local + no apply         → HardFailAssertion
(d) local + ImportError      → HardFailAssertion
(e) HF + valid + flag off    → None (with WARN log)
(f) HF + valid + flag on     → callable
"""
from __future__ import annotations

import textwrap
from pathlib import Path

import numpy as np
import pytest

from vla_project.deployment.post_process_loader import (
    HardFailAssertion,
    load_post_process,
)


def _write_pp(tmp: Path, body: str) -> Path:
    (tmp / "post_process.py").write_text(textwrap.dedent(body))
    return tmp


def _valid_body() -> str:
    return """\
        import numpy as np
        def apply(actions: np.ndarray, meta: dict) -> np.ndarray:
            actions[..., -1] = 0.5
            return actions
    """


def test_local_valid_returns_callable(tmp_path):
    _write_pp(tmp_path, _valid_body())
    fn = load_post_process(tmp_path, is_local=True, trust_checkpoint_code=False)
    assert callable(fn)
    out = fn(np.zeros((2, 3), dtype=np.float32), meta={})
    assert out[0, -1] == 0.5


def test_local_no_file_returns_none(tmp_path):
    fn = load_post_process(tmp_path, is_local=True, trust_checkpoint_code=False)
    assert fn is None


def test_local_missing_apply_raises(tmp_path):
    _write_pp(tmp_path, "x = 1\n")
    with pytest.raises(HardFailAssertion, match="apply"):
        load_post_process(tmp_path, is_local=True, trust_checkpoint_code=False)


def test_local_import_error_raises(tmp_path):
    _write_pp(tmp_path, "import nonexistent_module_xyz\n")
    with pytest.raises(HardFailAssertion):
        load_post_process(tmp_path, is_local=True, trust_checkpoint_code=False)


def test_hf_no_flag_skips_with_warn(tmp_path, caplog):
    _write_pp(tmp_path, _valid_body())
    with caplog.at_level("WARNING"):
        fn = load_post_process(tmp_path, is_local=False, trust_checkpoint_code=False)
    assert fn is None
    assert any("skipped" in rec.message for rec in caplog.records)


def test_hf_with_flag_returns_callable(tmp_path):
    _write_pp(tmp_path, _valid_body())
    fn = load_post_process(tmp_path, is_local=False, trust_checkpoint_code=True)
    assert callable(fn)
