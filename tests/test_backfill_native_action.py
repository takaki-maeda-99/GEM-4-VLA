"""tools.backfill_meta_native_action: rewrite local meta.json idempotently."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

# We import the main function (not via subprocess) so the test stays fast.
from tools.backfill_meta_native_action import backfill_local


def _write_meta(tmp_path: Path, contents: dict) -> Path:
    p = tmp_path / "meta.json"
    p.write_text(json.dumps(contents))
    return p


def test_adds_native_action_block(tmp_path):
    meta_p = _write_meta(tmp_path, {"step": 1, "cfg": {}})
    backfill_local(
        ckpt_dir=tmp_path,
        units="meter_axisangle_rad",
        frame="world",
        gripper_kind="absolute",
        gripper_units="normalized_0_1",
        gripper_closed=0.0,
        gripper_open=1.0,
    )
    m = json.loads(meta_p.read_text())
    assert m["native_action"]["frame"] == "world"
    assert m["native_action"]["gripper"]["sign"] == {"closed": 0.0, "open": 1.0}


def test_idempotent(tmp_path):
    meta_p = _write_meta(tmp_path, {"step": 1, "cfg": {}})
    args = dict(
        units="meter_axisangle_rad", frame="world",
        gripper_kind="absolute", gripper_units="normalized_0_1",
        gripper_closed=0.0, gripper_open=1.0,
    )
    backfill_local(ckpt_dir=tmp_path, **args)
    first = json.loads(meta_p.read_text())
    backfill_local(ckpt_dir=tmp_path, **args)
    second = json.loads(meta_p.read_text())
    assert first == second


def test_rejects_missing_meta_json(tmp_path):
    with pytest.raises(FileNotFoundError):
        backfill_local(
            ckpt_dir=tmp_path,
            units="meter_axisangle_rad", frame="world",
            gripper_kind="absolute", gripper_units="normalized_0_1",
            gripper_closed=0.0, gripper_open=1.0,
        )
