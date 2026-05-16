"""checkpoint.save writes meta.native_action when cfg.data.native_action is set."""
from __future__ import annotations

from vla_project.training.checkpoint import build_meta_dict


def test_build_meta_dict_includes_native_action_when_present():
    cfg = {
        "data": {
            "unnorm_key": "x",
            "native_action": {
                "units": "meter_axisangle_rad",
                "frame": "world",
                "gripper": {
                    "kind": "absolute",
                    "units": "normalized_0_1",
                    "sign": {"closed": 0, "open": 1},
                },
            },
        },
        "model": {},
    }
    out = build_meta_dict(step=1, cfg=cfg, norm_stats={}, git_commit="x")
    assert "native_action" in out
    assert out["native_action"]["frame"] == "world"


def test_build_meta_dict_omits_native_action_when_absent():
    cfg = {"data": {"unnorm_key": "x"}, "model": {}}
    out = build_meta_dict(step=1, cfg=cfg, norm_stats={}, git_commit="x")
    assert "native_action" not in out
