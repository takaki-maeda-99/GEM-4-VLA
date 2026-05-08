"""F5: GET /admin/schema returns the wire contract introspection payload.

Spec: docs/superpowers/specs/2026-05-08-server-request-validation-design.md §F5.
Returns 9 top-level keys (predictor, ckpt, wrist_hard_required, request_fields,
proprio, image, instruction, prompt, proprio_ood). prompt.max_tokens is null in
both Phase 0 modes (Phase 0 deferral — server doesn't tokenize yet).
"""
import pytest
from fastapi.testclient import TestClient

from tests.conftest import write_synthetic_ckpt

from vla_project.deployment.domain_adapter import (
    IMAGE_MAX_SIDE,
    IMAGE_MIN_SIDE,
    PROPRIO_OOD_HARD_ABS,
    PROPRIO_OOD_WARN_ABS,
)
from vla_project.deployment.inference_server import build_app
from vla_project.deployment.schemas import INSTRUCTION_MAX_BYTES

DEPLOY_YAML = "configs/deploy/v36_libero_spatial.yaml"


@pytest.fixture
def hold_position_client():
    app = build_app(
        predictor_kind="hold_position",
        checkpoint=None,
        deploy_config_path=DEPLOY_YAML,
        domain_id=0,
    )
    return TestClient(app)


def test_admin_schema_route_exists(hold_position_client):
    resp = hold_position_client.get("/admin/schema")
    assert resp.status_code == 200


def test_admin_schema_response_has_expected_top_level_keys(hold_position_client):
    resp = hold_position_client.get("/admin/schema")
    data = resp.json()
    expected_keys = {
        "predictor",
        "ckpt",
        "wrist_hard_required",
        "request_fields",
        "proprio",
        "image",
        "instruction",
        "prompt",
        "proprio_ood",
    }
    assert set(data.keys()) == expected_keys


def test_admin_schema_predictor_is_hold_position_class_name(hold_position_client):
    data = hold_position_client.get("/admin/schema").json()
    assert data["predictor"] == "HoldPositionChunkPredictor"


def test_admin_schema_ckpt_subfields(hold_position_client):
    data = hold_position_client.get("/admin/schema").json()
    ckpt = data["ckpt"]
    assert "expected_unnorm_key" in ckpt
    assert "expected_action_chunk_len" in ckpt
    assert "expected_action_dim" in ckpt
    assert "expected_proprio_dim" in ckpt
    assert ckpt["expected_action_dim"] == 7  # MimicRec MVP


def test_admin_schema_image_bounds_match_constants(hold_position_client):
    data = hold_position_client.get("/admin/schema").json()
    assert data["image"]["min_side"] == IMAGE_MIN_SIDE
    assert data["image"]["max_side"] == IMAGE_MAX_SIDE


def test_admin_schema_instruction_max_bytes_matches_constant(hold_position_client):
    data = hold_position_client.get("/admin/schema").json()
    assert data["instruction"]["max_bytes"] == INSTRUCTION_MAX_BYTES


def test_admin_schema_proprio_ood_thresholds_match_constants(hold_position_client):
    data = hold_position_client.get("/admin/schema").json()
    assert data["proprio_ood"]["warn_threshold"] == PROPRIO_OOD_WARN_ABS
    assert data["proprio_ood"]["hard_threshold"] == PROPRIO_OOD_HARD_ABS


def test_admin_schema_prompt_max_tokens_null_in_phase_0(hold_position_client):
    """Phase 0 deferral: server does not tokenize, so reporting max_tokens
    would be misleading. Should be null in both predictor modes."""
    data = hold_position_client.get("/admin/schema").json()
    assert data["prompt"]["max_tokens"] is None


def test_admin_schema_request_fields_match_deploy_yaml(hold_position_client):
    """Names come from configs/deploy/v36_libero_spatial.yaml's request_fields."""
    data = hold_position_client.get("/admin/schema").json()
    rf = data["request_fields"]
    assert rf["scene_image"] == "image_primary"
    assert rf["wrist_image"] == "image_wrist"
    assert rf["proprio"] == "proprio"
    assert rf["instruction"] == "instruction"


def test_admin_schema_proprio_source_components_present(hold_position_client):
    data = hold_position_client.get("/admin/schema").json()
    src = data["proprio"]["source"]
    assert "components" in src
    assert "total_dim" in src
    assert isinstance(src["components"], list)
    assert all("name" in c and "dims" in c and "units" in c for c in src["components"])
    assert sum(c["dims"] for c in src["components"]) == src["total_dim"]


def test_admin_schema_wrist_hard_required_is_bool(hold_position_client):
    data = hold_position_client.get("/admin/schema").json()
    assert isinstance(data["wrist_hard_required"], bool)
    # hold_position mode never hard-requires wrist (no ckpt cfg → False)
    assert data["wrist_hard_required"] is False


# ----- xvla_adapter mode (synthetic ckpt) -----


@pytest.fixture
def xvla_adapter_client(tmp_path):
    ckpt_dir = write_synthetic_ckpt(tmp_path, use_wrist_bridge=False)
    app = build_app(
        predictor_kind="xvla_adapter",
        checkpoint=ckpt_dir,
        deploy_config_path=DEPLOY_YAML,
        domain_id=0,
    )
    return TestClient(app)


@pytest.fixture
def xvla_adapter_wrist_required_client(tmp_path):
    ckpt_dir = write_synthetic_ckpt(tmp_path, use_wrist_bridge=True)
    app = build_app(
        predictor_kind="xvla_adapter",
        checkpoint=ckpt_dir,
        deploy_config_path=DEPLOY_YAML,
        domain_id=0,
    )
    return TestClient(app)


def test_admin_schema_predictor_is_xvla_adapter_class_name(xvla_adapter_client):
    data = xvla_adapter_client.get("/admin/schema").json()
    assert data["predictor"] == "XVLAAdapterChunkPredictor"


def test_admin_schema_prompt_max_tokens_null_in_xvla_adapter_phase_0(xvla_adapter_client):
    """Phase 0 deferral: same null contract regardless of predictor mode."""
    data = xvla_adapter_client.get("/admin/schema").json()
    assert data["prompt"]["max_tokens"] is None


def test_admin_schema_wrist_hard_required_true_when_ckpt_uses_wrist_bridge(
    xvla_adapter_wrist_required_client,
):
    data = xvla_adapter_wrist_required_client.get("/admin/schema").json()
    assert data["wrist_hard_required"] is True
