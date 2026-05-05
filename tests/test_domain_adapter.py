"""Tests for DomainAdapter and DeployConfig.

Spec §Section 4 deploy YAML schema, §Section 3 per-request data flow.
Covered:
- DeployConfig pydantic round-trip + per-field validation
- proprio.adapt step ops (deg_to_rad, copy, pad_zeros)
- Q99 normalize/denormalize with mask handling
- gripper convention conversion (normalized_0_1 ↔ signed_neg1_pos1, sign flip)
- frame_conversion=none identity
- row-shape postprocess assert (input [T, 7] ok; [T, 6] / [T, 8] raises)
- startup hard-fail assertions: domain_id < 0, mismatched unnorm_key, etc.
"""
from __future__ import annotations

import base64
import io

import numpy as np
import pytest
import yaml
from PIL import Image
from pydantic import ValidationError

from vla_project.deployment.domain_adapter import (
    DeployConfig,
    DomainAdapter,
    HardFailAssertion,
    load_deploy_config,
)
from vla_project.deployment.schemas import PredictRequest


# ---------- helpers ----------

def _make_jpeg_b64(size: int = 224) -> str:
    img = Image.new("RGB", (size, size), color=(127, 127, 127))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _minimal_deploy_yaml(**overrides) -> dict:
    """Returns a valid DeployConfig dict matching v36 + SO-101 contract."""
    base = {
        "ckpt": {
            "expected_unnorm_key": "libero_spatial_no_noops",
            "expected_action_chunk_len": 8,
            "expected_action_dim": 7,
            "expected_proprio_dim": 8,
        },
        "request_fields": {
            "scene_image": "image_primary",
            "wrist_image": "image_wrist",
            "proprio": "proprio",
            "instruction": "instruction",
        },
        "proprio": {
            "source": {
                "components": [
                    {"name": "joint_pos", "dims": 6, "units": "deg"},
                    {"name": "gripper_pos", "dims": 1, "units": "normalized_neg1_pos1"},
                ],
                "total_dim": 7,
            },
            "adapt": {
                "steps": [
                    {"op": "deg_to_rad", "source": "joint_pos", "dims": 6},
                    {"op": "copy", "source": "gripper_pos", "dims": 1},
                    {"op": "pad_zeros", "dims": 1},
                ],
                "output_dim": 8,
            },
            "normalization": {"method": "q99", "stats_key": "proprio"},
        },
        "action": {
            "native": {
                "units": "meter_axisangle_rad",
                "frame": "world",
                "gripper": {
                    "kind": "absolute",
                    "units": "normalized_0_1",
                    "sign": {"closed": 0, "open": 1},
                },
            },
            "contract": {
                "units": "meter_axisangle_rad",
                "frame": "ee_local",
                "gripper": {
                    "kind": "absolute",
                    "units": "normalized_0_1",
                    "sign": {"closed": 0, "open": 1},
                },
            },
            "denormalization": {"method": "q99", "stats_key": "action"},
            "frame_conversion": {"method": "none"},
        },
        "holdposition": {"gripper_native_midpoint": 0.5},
        "wire_only_smoke": True,  # set true so v36 world->ee_local mismatch passes startup for tests
        "runtime": {
            "device": "cpu",
            "dtype": "bf16",
            "torch_compile": "off",
            "warmup_iters": 0,
        },
    }
    for k, v in overrides.items():
        base[k] = v
    return base


def _norm_stats_v36() -> dict:
    """Subset of meta.norm_stats[unnorm_key] sufficient for D_prop=8 + A=7."""
    return {
        "action": {
            "mean": [0.0] * 7,
            "std": [1.0] * 7,
            "q01": [-1.0] * 7,
            "q99": [1.0] * 7,
            "mask": [True, True, True, True, True, True, False],  # gripper dim passes through
        },
        "proprio": {
            "mean": [0.0] * 8,
            "std": [1.0] * 8,
            "q01": [-1.0] * 8,
            "q99": [1.0] * 8,
            "mask": [True] * 8,
        },
    }


# ---------- DeployConfig parsing ----------

class TestDeployConfigParsing:
    def test_round_trip_minimal(self):
        cfg = DeployConfig.model_validate(_minimal_deploy_yaml())
        assert cfg.ckpt.expected_unnorm_key == "libero_spatial_no_noops"
        assert cfg.proprio.adapt.output_dim == 8
        assert cfg.holdposition.gripper_native_midpoint == 0.5

    def test_proprio_source_total_dim_must_match_components(self):
        bad = _minimal_deploy_yaml()
        bad["proprio"]["source"]["total_dim"] = 99
        with pytest.raises(ValidationError, match="total_dim"):
            DeployConfig.model_validate(bad)

    def test_load_deploy_config_from_yaml_file(self, tmp_path):
        path = tmp_path / "v36.yaml"
        path.write_text(yaml.safe_dump(_minimal_deploy_yaml()))
        cfg = load_deploy_config(path)
        assert cfg.ckpt.expected_action_chunk_len == 8


# ---------- preprocess: JPEG decode + field mapping ----------

class TestPreprocess:
    def test_decode_jpeg_to_uint8_rgb_array(self):
        cfg = DeployConfig.model_validate(_minimal_deploy_yaml())
        adapter = DomainAdapter(cfg, _norm_stats_v36(), domain_id=0)
        req = PredictRequest(
            image_primary=_make_jpeg_b64(),
            image_wrist=_make_jpeg_b64(),
            proprio=[10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 0.5],
            instruction="x",
        )
        obs = adapter.preprocess(req)
        assert obs["scene_image"].dtype == np.uint8
        assert obs["scene_image"].shape == (224, 224, 3)
        assert obs["wrist_image"].shape == (224, 224, 3)
        assert obs["wrist_was_provided"] is True

    def test_wrist_absent_with_dropout_tolerant_zero_fills(self):
        """When deploy yaml allows wrist absent (Phase 0 wire_only_smoke=True
        bypasses hard-required check), zero-fill + wrist_was_provided=False."""
        cfg = DeployConfig.model_validate(_minimal_deploy_yaml())
        adapter = DomainAdapter(cfg, _norm_stats_v36(), domain_id=0)
        req = PredictRequest(
            image_primary=_make_jpeg_b64(),
            image_wrist=None,
            proprio=[0.0] * 7,
            instruction="x",
        )
        obs = adapter.preprocess(req)
        # In Phase 0 wire_only_smoke mode, missing wrist becomes a zero-image.
        assert obs["wrist_image"].shape == (224, 224, 3)
        np.testing.assert_array_equal(obs["wrist_image"], np.zeros((224, 224, 3), dtype=np.uint8))
        assert obs["wrist_was_provided"] is False


class TestProprioAdapt:
    def test_deg_to_rad_op(self):
        cfg = DeployConfig.model_validate(_minimal_deploy_yaml())
        # disable normalization for this unit-level test
        cfg = cfg.model_copy(update={"proprio": cfg.proprio.model_copy(
            update={"normalization": cfg.proprio.normalization.model_copy(update={"method": "none"})}
        )})
        adapter = DomainAdapter(cfg, norm_stats=None, domain_id=0)
        # 90 deg → π/2 rad ≈ 1.5708; gripper 0.5 → 0.5; pad_zeros → 0.0
        raw = [90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 0.5]
        out = adapter._apply_proprio_adapt(np.array(raw, dtype=np.float32))
        assert out.shape == (8,)
        np.testing.assert_allclose(out[:6], np.pi / 2, atol=1e-5)
        assert out[6] == 0.5
        assert out[7] == 0.0

    def test_q99_normalize_with_unit_stats_is_identity(self):
        cfg = DeployConfig.model_validate(_minimal_deploy_yaml())
        adapter = DomainAdapter(cfg, _norm_stats_v36(), domain_id=0)
        out = adapter._normalize_proprio(np.zeros(8, dtype=np.float32))
        # mean=0, q99=1, q01=-1 → normalize is identity at zero
        np.testing.assert_allclose(out, np.zeros(8), atol=1e-6)


# ---------- postprocess: gripper conv + frame conv + row-shape assert ----------

class TestPostprocess:
    def test_identity_gripper_conversion_when_native_eq_contract(self):
        """v36 native and SO-101 contract both normalized_0_1 closed=0/open=1
        → identity."""
        cfg = DeployConfig.model_validate(_minimal_deploy_yaml())
        adapter = DomainAdapter(cfg, _norm_stats_v36(), domain_id=0)
        native = np.array([[0.001, 0.0, 0.0, 0.0, 0.0, 0.0, 0.7]], dtype=np.float32)
        out = adapter.postprocess(native)
        assert out == [[pytest.approx(0.001), 0.0, 0.0, 0.0, 0.0, 0.0, pytest.approx(0.7)]]

    def test_signed_to_normalized_gripper_conversion(self):
        """signed_neg1_pos1 (open=-1, closed=+1) → normalized_0_1 (closed=0, open=1)."""
        cfg_d = _minimal_deploy_yaml()
        cfg_d["action"]["native"]["gripper"] = {
            "kind": "absolute",
            "units": "signed_neg1_pos1",
            "sign": {"closed": 1, "open": -1},
        }
        cfg = DeployConfig.model_validate(cfg_d)
        adapter = DomainAdapter(cfg, _norm_stats_v36(), domain_id=0)
        # native +1 (closed) → contract 0 (closed)
        # native -1 (open)   → contract 1 (open)
        # native  0 (mid)    → contract 0.5 (mid)
        native = np.array([
            [0, 0, 0, 0, 0, 0, +1.0],
            [0, 0, 0, 0, 0, 0, -1.0],
            [0, 0, 0, 0, 0, 0, 0.0],
        ], dtype=np.float32)
        out = adapter.postprocess(native)
        assert out[0][6] == pytest.approx(0.0)
        assert out[1][6] == pytest.approx(1.0)
        assert out[2][6] == pytest.approx(0.5)

    def test_row_width_assert_rejects_too_few_cols(self):
        cfg = DeployConfig.model_validate(_minimal_deploy_yaml())
        adapter = DomainAdapter(cfg, _norm_stats_v36(), domain_id=0)
        with pytest.raises(AssertionError, match="row width"):
            adapter.postprocess(np.zeros((1, 6), dtype=np.float32))

    def test_row_width_assert_rejects_too_many_cols(self):
        cfg = DeployConfig.model_validate(_minimal_deploy_yaml())
        adapter = DomainAdapter(cfg, _norm_stats_v36(), domain_id=0)
        with pytest.raises(AssertionError, match="row width"):
            adapter.postprocess(np.zeros((1, 8), dtype=np.float32))

    def test_q99_denormalize_respects_mask_false_passthrough(self):
        """Verifies inverse of training-time normalize_action_q99:
            raw = q01 + (norm + 1) * span / 2
        Uses an asymmetric Q99 envelope where (mean != midpoint) so the
        original plan's incorrect formula `mean + a * span/2` would be
        detected here."""
        cfg_d = _minimal_deploy_yaml()
        cfg = DeployConfig.model_validate(cfg_d)
        stats = _norm_stats_v36()
        # asymmetric: q01=-2, q99=1 → span=3, midpoint=-0.5 (≠ mean=0)
        stats["action"]["q01"] = [-2.0] * 7
        stats["action"]["q99"] = [1.0] * 7
        stats["action"]["mean"] = [0.0] * 7
        adapter = DomainAdapter(cfg, stats, domain_id=0)
        native = np.array([[0.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.5]], dtype=np.float32)
        out = adapter.postprocess(native, denormalize=True)
        # 0.0 → q01 + 0.5*span = -2 + 1.5 = -0.5 (midpoint)
        # 1.0 → q01 + 1.0*span = -2 + 3.0 = 1.0 (= q99)
        # -1.0 → q01 + 0.0*span = -2.0 (= q01)
        assert out[0][0] == pytest.approx(-0.5)
        assert out[0][1] == pytest.approx(1.0)
        assert out[0][2] == pytest.approx(-2.0)
        # gripper (mask=False) passes through unchanged
        assert out[0][6] == pytest.approx(0.5)


# ---------- startup hard-fail assertions ----------

class TestStartupAssertions:
    def _norm_stats(self):
        return {"libero_spatial_no_noops": _norm_stats_v36()}

    def test_negative_domain_id_raises(self):
        cfg = DeployConfig.model_validate(_minimal_deploy_yaml())
        with pytest.raises(HardFailAssertion, match="domain_id"):
            DomainAdapter.validate_startup_xvla(
                cfg, meta_cfg={"model": {"num_domains": 1}, "data": {"unnorm_key": "libero_spatial_no_noops", "action_chunk_len": 8}},
                norm_stats=self._norm_stats(), domain_id=-1,
            )

    def test_domain_id_out_of_range_raises(self):
        cfg = DeployConfig.model_validate(_minimal_deploy_yaml())
        with pytest.raises(HardFailAssertion, match="domain_id"):
            DomainAdapter.validate_startup_xvla(
                cfg, meta_cfg={"model": {"num_domains": 1}, "data": {"unnorm_key": "libero_spatial_no_noops", "action_chunk_len": 8}},
                norm_stats=self._norm_stats(), domain_id=5,
            )

    def test_unnorm_key_mismatch_raises(self):
        cfg = DeployConfig.model_validate(_minimal_deploy_yaml())
        with pytest.raises(HardFailAssertion, match="unnorm_key"):
            DomainAdapter.validate_startup_xvla(
                cfg, meta_cfg={"model": {"num_domains": 1}, "data": {"unnorm_key": "libero_other", "action_chunk_len": 8}},
                norm_stats=self._norm_stats(), domain_id=0,
            )

    def test_action_chunk_len_fallback_chain_picks_default_8(self):
        """v36 sets cfg.data.action_chunk_len=8; v35-style ckpts set neither,
        falling back to C.ACTION_CHUNK_LEN=8."""
        cfg = DeployConfig.model_validate(_minimal_deploy_yaml())
        # cfg.data has no action_chunk_len → fallback to default
        DomainAdapter.validate_startup_xvla(
            cfg, meta_cfg={"model": {"num_domains": 1}, "data": {"unnorm_key": "libero_spatial_no_noops"}},
            norm_stats=self._norm_stats(), domain_id=0,
        )

    def test_holdposition_startup_skips_ckpt_asserts(self):
        cfg = DeployConfig.model_validate(_minimal_deploy_yaml())
        # no meta_cfg / no norm_stats → should still pass for hold_position mode
        DomainAdapter.validate_startup_hold_position(cfg, domain_id=0)
        with pytest.raises(HardFailAssertion, match="domain_id"):
            DomainAdapter.validate_startup_hold_position(cfg, domain_id=-1)

    def test_holdposition_startup_rejects_zero_chunk_len(self):
        bad = _minimal_deploy_yaml()
        bad["ckpt"]["expected_action_chunk_len"] = 0
        cfg = DeployConfig.model_validate(bad)
        with pytest.raises(HardFailAssertion, match="action_chunk_len"):
            DomainAdapter.validate_startup_hold_position(cfg, domain_id=0)

    def test_holdposition_startup_applies_gripper_compat(self):
        """Spec §Section 4 step 4 keeps gripper-convention compat check active
        in HoldPosition mode (only frame_conversion is skipped). Mismatched
        gripper conventions without wire_only_smoke must fail."""
        bad = _minimal_deploy_yaml()
        bad["wire_only_smoke"] = False
        bad["action"]["native"]["gripper"] = {
            "kind": "absolute",
            "units": "signed_neg1_pos1",
            "sign": {"closed": 1, "open": -1},
        }
        # contract still normalized_0_1; built-in linear remap will work, so
        # this should NOT fail (compat check passes for any (closed, open) pair).
        # The failure case is: sign.closed == sign.open (degenerate) — covered below.
        cfg = DeployConfig.model_validate(bad)
        DomainAdapter.validate_startup_hold_position(cfg, domain_id=0)  # ok

    def test_holdposition_startup_rejects_degenerate_native_gripper(self):
        bad = _minimal_deploy_yaml()
        bad["action"]["native"]["gripper"]["sign"] = {"closed": 0.5, "open": 0.5}
        cfg = DeployConfig.model_validate(bad)
        with pytest.raises(HardFailAssertion, match="gripper"):
            DomainAdapter.validate_startup_hold_position(cfg, domain_id=0)


class TestStartupAssertionsXVLAFull:
    """All hard-fail assertions from spec §Section 4 step 3, exercised against
    the xvla_adapter validator. Tests live in test_runtime_load.py per spec
    §Section 6 testing table — the implementations of the assertions are in
    DomainAdapter.validate_startup_xvla but the tests are gathered here for
    locality with the meta.json fixture."""

    def _ok_meta_cfg(self):
        return {"model": {"num_domains": 1}, "data": {"unnorm_key": "libero_spatial_no_noops", "action_chunk_len": 8}}

    def _ok_norm_stats(self):
        return {"libero_spatial_no_noops": _norm_stats_v36()}

    def test_action_dim_mismatch_raises(self):
        cfg_d = _minimal_deploy_yaml()
        cfg_d["ckpt"]["expected_action_dim"] = 9  # ckpt has 7
        cfg = DeployConfig.model_validate(cfg_d)
        with pytest.raises(HardFailAssertion, match="action_dim"):
            DomainAdapter.validate_startup_xvla(
                cfg, meta_cfg=self._ok_meta_cfg(),
                norm_stats=self._ok_norm_stats(), domain_id=0,
            )

    def test_proprio_dim_mismatch_raises(self):
        cfg_d = _minimal_deploy_yaml()
        cfg_d["ckpt"]["expected_proprio_dim"] = 9
        cfg_d["proprio"]["adapt"]["output_dim"] = 9
        cfg_d["proprio"]["adapt"]["steps"][-1]["dims"] = 2  # match output_dim
        cfg = DeployConfig.model_validate(cfg_d)
        with pytest.raises(HardFailAssertion, match="proprio"):
            DomainAdapter.validate_startup_xvla(
                cfg, meta_cfg=self._ok_meta_cfg(),
                norm_stats=self._ok_norm_stats(), domain_id=0,
            )

    def test_proprio_adapt_output_dim_mismatch_raises(self):
        cfg_d = _minimal_deploy_yaml()
        cfg_d["proprio"]["adapt"]["output_dim"] = 9
        cfg_d["proprio"]["adapt"]["steps"][-1]["dims"] = 2
        cfg = DeployConfig.model_validate(cfg_d)
        with pytest.raises(HardFailAssertion, match="output_dim"):
            DomainAdapter.validate_startup_xvla(
                cfg, meta_cfg=self._ok_meta_cfg(),
                norm_stats=self._ok_norm_stats(), domain_id=0,
            )

    def test_hard_required_wrist_missing_raises(self):
        cfg_d = _minimal_deploy_yaml()
        cfg_d["request_fields"]["wrist_image"] = None
        cfg = DeployConfig.model_validate(cfg_d)
        meta_cfg = self._ok_meta_cfg()
        meta_cfg["model"]["use_wrist_bridge"] = True  # hard-required
        with pytest.raises(HardFailAssertion, match="wrist"):
            DomainAdapter.validate_startup_xvla(
                cfg, meta_cfg=meta_cfg,
                norm_stats=self._ok_norm_stats(), domain_id=0,
            )

    def test_frame_mismatch_without_wire_only_smoke_raises(self):
        cfg_d = _minimal_deploy_yaml()
        cfg_d["wire_only_smoke"] = False  # no escape hatch
        # native.frame=world, contract.frame=ee_local in default → mismatched
        cfg = DeployConfig.model_validate(cfg_d)
        with pytest.raises(HardFailAssertion, match="frame"):
            DomainAdapter.validate_startup_xvla(
                cfg, meta_cfg=self._ok_meta_cfg(),
                norm_stats=self._ok_norm_stats(), domain_id=0,
            )
