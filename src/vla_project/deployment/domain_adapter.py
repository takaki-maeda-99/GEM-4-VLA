"""DomainAdapter — per-domain in/out conversion + DeployConfig pydantic.

Loads `configs/deploy/<robot>_<model>.yaml` into a typed DeployConfig and
provides:
  - preprocess(req): JPEG decode + field-name mapping + proprio adapt
    (deg_to_rad, copy, pad_zeros) + Q99 normalize → Obs dict
  - postprocess(native_chunk): gripper-convention conversion + frame
    conversion (none in Phase 0) + row-shape assert → list[list[float]]
  - validate_startup_xvla / validate_startup_hold_position: hard-fail asserts
    per spec §Section 4 startup validation flow

Phase 0 implements the contract for HoldPosition path; xvla_adapter mode
runs all the same code paths but the predictor itself is stubbed.
"""
from __future__ import annotations

import base64
import io
import json
import logging
from pathlib import Path
from typing import Any, Literal

import numpy as np
import yaml
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, model_validator

# C.ACTION_CHUNK_LEN — single source of truth for the v33/v35 default
from vla_project.data import constants as C

from vla_project.deployment.schemas import PredictRequest

logger = logging.getLogger("vla_project.deployment.domain_adapter")

# F1 (per docs/superpowers/specs/2026-05-08-server-request-validation-design.md §F1):
# Image side sanity bounds. Catches replay corruption (1×1) and abusive payloads
# (100k×100k) before the JPEG decoder is asked to allocate pixel buffers.
IMAGE_MIN_SIDE: int = 64
IMAGE_MAX_SIDE: int = 4096

# F3 (per docs/superpowers/specs/2026-05-08-server-request-validation-design.md §F3):
# Proprio out-of-distribution thresholds. Computed against the normalized values
# (after q01/q99 mapping). >WARN absorbed by clip + WARNING log; >HARD raises.
# 10.0 is wide enough to admit legitimate startup poses outside training
# support but still catches deg/rad swap (rad ≈ 0.5 → deg = 30 → ~30x q-range).
PROPRIO_OOD_WARN_ABS: float = 1.0
PROPRIO_OOD_HARD_ABS: float = 10.0


class HardFailAssertion(Exception):
    """Raised at startup if deploy yaml ↔ ckpt metadata ↔ args are inconsistent."""


# ---------- DeployConfig pydantic schema ----------

class _ProprioComponent(BaseModel):
    name: str
    dims: int
    units: str


class _ProprioSource(BaseModel):
    components: list[_ProprioComponent]
    total_dim: int

    @model_validator(mode="after")
    def _check_total(self) -> "_ProprioSource":
        if sum(c.dims for c in self.components) != self.total_dim:
            raise ValueError("proprio.source.total_dim must equal sum(components.dims)")
        return self


class _ProprioStep(BaseModel):
    model_config = ConfigDict(extra="forbid")
    op: Literal["deg_to_rad", "rad_to_deg", "copy", "pad_zeros", "scale", "constant"]
    source: str | None = None
    dims: int = 1
    value: float | None = None  # for "constant"
    factor: float | None = None  # for "scale"


class _ProprioAdapt(BaseModel):
    steps: list[_ProprioStep]
    output_dim: int


class _ProprioNormalization(BaseModel):
    method: Literal["none", "q99"] = "q99"
    stats_key: str = "proprio"


class _Proprio(BaseModel):
    source: _ProprioSource
    adapt: _ProprioAdapt
    normalization: _ProprioNormalization


class _GripperSign(BaseModel):
    closed: float
    open: float


class _Gripper(BaseModel):
    kind: Literal["absolute", "delta", "binary"]
    units: Literal["normalized_0_1", "signed_neg1_pos1", "binary_threshold_0p5"]
    sign: _GripperSign


class _ActionSide(BaseModel):
    units: Literal["meter_axisangle_rad"] = "meter_axisangle_rad"
    frame: Literal["ee_local", "world"]
    gripper: _Gripper


class _Denormalization(BaseModel):
    method: Literal["none", "q99", "mean_std"] = "q99"
    stats_key: str = "action"


class _FrameConversion(BaseModel):
    method: Literal["none", "world_to_ee_local", "ee_local_to_world"] = "none"


class _Action(BaseModel):
    native: _ActionSide
    contract: _ActionSide
    denormalization: _Denormalization
    frame_conversion: _FrameConversion


class _CkptIdentity(BaseModel):
    expected_unnorm_key: str
    expected_action_chunk_len: int
    expected_action_dim: int
    expected_proprio_dim: int


class _RequestFields(BaseModel):
    scene_image: str
    wrist_image: str | None = None
    proprio: str = "proprio"
    instruction: str = "instruction"


class _HoldPosition(BaseModel):
    gripper_native_midpoint: float = 0.5


class _Runtime(BaseModel):
    device: str = "cuda:0"
    dtype: Literal["bf16", "fp32"] = "bf16"
    torch_compile: Literal["off", "reduce-overhead", "default"] = "off"
    warmup_iters: int = 1


class DeployConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ckpt: _CkptIdentity
    request_fields: _RequestFields
    proprio: _Proprio
    action: _Action
    holdposition: _HoldPosition = Field(default_factory=_HoldPosition)
    wire_only_smoke: bool = False
    runtime: _Runtime = Field(default_factory=_Runtime)


def load_deploy_config(path: str | Path) -> DeployConfig:
    return DeployConfig.model_validate(yaml.safe_load(Path(path).read_text()))


# ---------- DomainAdapter ----------

class DomainAdapter:
    def __init__(
        self,
        cfg: DeployConfig,
        norm_stats: dict | None,
        domain_id: int,
        *,
        wrist_hard_required: bool = False,
    ) -> None:
        self.cfg = cfg
        self.norm_stats = norm_stats
        self.domain_id = int(domain_id)
        self.wrist_hard_required = bool(wrist_hard_required)

    # ----- preprocess -----

    def preprocess(self, req: PredictRequest) -> dict[str, Any]:
        scene = self._decode_jpeg_b64(req.image_primary)
        wrist_b64 = req.image_wrist
        if wrist_b64 is not None:
            wrist = self._decode_jpeg_b64(wrist_b64)
            wrist_was_provided = True
        else:
            if self.wrist_hard_required:
                raise ValueError(
                    "checkpoint requires wrist_image (use_wrist_bridge or "
                    "DINOv2 path) but request omitted it"
                )
            wrist = np.zeros((224, 224, 3), dtype=np.uint8)
            wrist_was_provided = False
        proprio_raw = np.asarray(req.proprio, dtype=np.float32)
        # F3a: non-finite proprio is unconditionally invalid. Catches NaN/inf
        # from upstream sensor faults or test fixtures; also short-circuits any
        # downstream normalize/clip that would silently swallow the signal.
        if not np.isfinite(proprio_raw).all():
            bad_dims = np.where(~np.isfinite(proprio_raw))[0].tolist()
            raise ValueError(
                f"proprio contains non-finite values at dims {bad_dims}"
            )
        if proprio_raw.shape[0] != self.cfg.proprio.source.total_dim:
            raise ValueError(
                f"proprio length {proprio_raw.shape[0]} != "
                f"deploy.proprio.source.total_dim {self.cfg.proprio.source.total_dim}"
            )
        proprio_adapted = self._apply_proprio_adapt(proprio_raw)
        proprio_normalized = self._normalize_proprio(proprio_adapted)
        return {
            "scene_image": scene,
            "wrist_image": wrist,
            "wrist_was_provided": wrist_was_provided,
            "proprio": proprio_normalized,
            "language": req.instruction,
        }

    @staticmethod
    def _decode_jpeg_b64(b64_str: str) -> np.ndarray:
        raw = base64.b64decode(b64_str)
        # F1: header-parse-first. Image.open() reads only the JPEG header
        # (no pixel decode); .size returns (W, H) from the header. We bound
        # the dimensions before convert("RGB") forces full pixel decode,
        # so an attacker / corrupt payload can't allocate gigabytes via
        # an oversized header.
        img = Image.open(io.BytesIO(raw))
        w, h = img.size
        if min(w, h) < IMAGE_MIN_SIDE or max(w, h) > IMAGE_MAX_SIDE:
            raise ValueError(
                f"image side ({w}, {h}) out of sanity bound "
                f"[{IMAGE_MIN_SIDE}, {IMAGE_MAX_SIDE}]"
            )
        img = img.convert("RGB")
        return np.asarray(img, dtype=np.uint8)

    def _apply_proprio_adapt(self, raw: np.ndarray) -> np.ndarray:
        # Index source components by name for "source: <name>" lookups.
        offsets: dict[str, tuple[int, int]] = {}
        i = 0
        for c in self.cfg.proprio.source.components:
            offsets[c.name] = (i, i + c.dims)
            i += c.dims
        out_parts: list[np.ndarray] = []
        for step in self.cfg.proprio.adapt.steps:
            if step.op == "deg_to_rad":
                lo, hi = offsets[step.source]  # type: ignore[index]
                out_parts.append(raw[lo:hi] * np.float32(np.pi / 180.0))
            elif step.op == "rad_to_deg":
                lo, hi = offsets[step.source]  # type: ignore[index]
                out_parts.append(raw[lo:hi] * np.float32(180.0 / np.pi))
            elif step.op == "copy":
                lo, hi = offsets[step.source]  # type: ignore[index]
                out_parts.append(raw[lo:hi].copy())
            elif step.op == "pad_zeros":
                out_parts.append(np.zeros(step.dims, dtype=np.float32))
            elif step.op == "scale":
                lo, hi = offsets[step.source]  # type: ignore[index]
                out_parts.append(raw[lo:hi] * np.float32(step.factor or 1.0))
            elif step.op == "constant":
                out_parts.append(np.full(step.dims, step.value or 0.0, dtype=np.float32))
            else:
                raise ValueError(f"unknown proprio.adapt op: {step.op}")
        out = np.concatenate(out_parts, axis=0).astype(np.float32)
        if out.shape[0] != self.cfg.proprio.adapt.output_dim:
            raise ValueError(
                f"proprio.adapt produced {out.shape[0]} dims, expected "
                f"output_dim={self.cfg.proprio.adapt.output_dim}"
            )
        return out

    def _normalize_proprio(self, x: np.ndarray) -> np.ndarray:
        if self.cfg.proprio.normalization.method == "none" or self.norm_stats is None:
            return x
        # Q99: normalize each dim into [-1, +1] using (q01, q99) with mask.
        stats = self.norm_stats[self.cfg.proprio.normalization.stats_key]
        q01 = np.asarray(stats["q01"], dtype=np.float32)
        q99 = np.asarray(stats["q99"], dtype=np.float32)
        mask = np.asarray(stats.get("mask", [True] * len(q01)), dtype=bool)
        span = q99 - q01
        span = np.where(span == 0, 1.0, span)
        normed = 2.0 * (x - q01) / span - 1.0
        # F3b: OOD detection happens BEFORE clip. Mask=False dims are not
        # subject to OOD checks (the model receives the raw value for them).
        abs_normed = np.abs(normed)
        # F3b hard: |normed| > PROPRIO_OOD_HARD_ABS → 422. Runs first so the
        # warn line below is skipped on hard reject (single invalid_request
        # log is sufficient — see spec §F3 "warn-vs-raise ordering").
        hard_violations = (abs_normed > PROPRIO_OOD_HARD_ABS) & mask
        if hard_violations.any():
            hard_dims = np.where(hard_violations)[0].tolist()
            max_excess = float((abs_normed * mask).max() - 1.0)
            raise ValueError(
                f"proprio normalized |x|>{PROPRIO_OOD_HARD_ABS} at dims "
                f"{hard_dims} (max excess {max_excess:.2f}); likely unit "
                f"mismatch (deg/rad swap or wrong proprio_key)"
            )
        # F3b warn: |normed| > PROPRIO_OOD_WARN_ABS (and ≤ HARD) → log + clip.
        warn_violations = (abs_normed > PROPRIO_OOD_WARN_ABS) & mask
        if warn_violations.any():
            ood_dims = np.where(warn_violations)[0].tolist()
            max_excess = float((abs_normed * mask).max() - 1.0)
            logger.warning(json.dumps({
                "event": "proprio_ood",
                "ood_dim_count": len(ood_dims),
                "ood_max_excess": round(max_excess, 3),
                "ood_dims": ood_dims,
            }))
        # Clamp to [-1, +1] (training-time q99 clipping convention).
        normed = np.clip(normed, -1.0, 1.0)
        return np.where(mask, normed, x).astype(np.float32)

    # ----- postprocess -----

    def postprocess(self, native_chunk: np.ndarray, *, denormalize: bool = False) -> list[list[float]]:
        a = np.asarray(native_chunk, dtype=np.float32)
        assert a.ndim == 2, f"native_chunk must be 2-D; got {a.ndim}-D"
        T, A = a.shape
        # row-shape assert (per spec §Section 6 test_domain_adapter expectations)
        if A != self.cfg.ckpt.expected_action_dim:
            raise AssertionError(f"row width {A} != expected_action_dim {self.cfg.ckpt.expected_action_dim}")

        # Optional denorm (used when called from XVLAAdapter path; HoldPosition skips).
        if denormalize and self.cfg.action.denormalization.method == "q99" and self.norm_stats is not None:
            a = self._q99_denorm_action(a)

        # Frame conversion: Phase 0 only supports "none". Implementation deferred to Phase 1.
        if self.cfg.action.frame_conversion.method != "none":
            raise NotImplementedError(
                f"frame_conversion.method={self.cfg.action.frame_conversion.method} "
                "is Phase 1 work"
            )

        # Gripper conversion (linear remap based on (closed, open) in native + contract).
        a = self._convert_gripper(a)
        return a.tolist()

    def _q99_denorm_action(self, a: np.ndarray) -> np.ndarray:
        """Inverse of training-time normalize_action_q99.

        Mirrors data/normalization.py:denormalize_action_q99 exactly:
            raw = q01 + (a + 1) * span / 2     where span = q99 - q01
        For mask=False dims (e.g., gripper), pass through unchanged.

        NOTE: an earlier draft of the plan used `mean + a * span / 2`, which is
        only equivalent when mean == (q01 + q99) / 2 (symmetric distribution).
        Real LIBERO action stats are skewed, so that formula produced a
        per-dim offset proportional to (mean - mid). Fixed here.
        """
        stats = self.norm_stats[self.cfg.action.denormalization.stats_key]  # type: ignore[index]
        q01 = np.asarray(stats["q01"], dtype=np.float32)
        q99 = np.asarray(stats["q99"], dtype=np.float32)
        mask = np.asarray(stats.get("mask", [True] * a.shape[1]), dtype=bool)
        span = q99 - q01
        span = np.where(span == 0, 1.0, span)
        denormed = q01 + (a + 1.0) * 0.5 * span
        return np.where(mask, denormed, a).astype(np.float32)

    def _convert_gripper(self, a: np.ndarray) -> np.ndarray:
        n = self.cfg.action.native.gripper
        c = self.cfg.action.contract.gripper
        # Identity short-circuit
        if (n.units == c.units and n.sign.closed == c.sign.closed and n.sign.open == c.sign.open):
            return a
        # Linear remap: t in [0, 1] is the "openness" fraction.
        denom = (n.sign.open - n.sign.closed)
        if denom == 0:
            raise ValueError("native.gripper sign.open == sign.closed — cannot remap")
        g_native = a[:, -1]
        t = (g_native - n.sign.closed) / denom
        g_contract = c.sign.closed + t * (c.sign.open - c.sign.closed)
        out = a.copy()
        out[:, -1] = g_contract
        return out

    # ----- startup validators -----

    @staticmethod
    def validate_startup_xvla(
        cfg: DeployConfig,
        *,
        meta_cfg: dict,
        norm_stats: dict,
        domain_id: int,
    ) -> None:
        m_model = meta_cfg.get("model", {})
        m_data = meta_cfg.get("data", {})

        num_domains = int(m_model.get("num_domains", 0))
        if not (0 <= domain_id < num_domains):
            raise HardFailAssertion(
                f"domain_id={domain_id} out of range [0, {num_domains})"
            )
        if m_data.get("unnorm_key") != cfg.ckpt.expected_unnorm_key:
            raise HardFailAssertion(
                f"ckpt unnorm_key={m_data.get('unnorm_key')!r} != "
                f"expected {cfg.ckpt.expected_unnorm_key!r}"
            )
        # action_chunk_len fallback chain (per spec §Section 4 step 3)
        resolved_chunk_len = (
            m_model.get("action_chunk_len")
            or m_data.get("action_chunk_len")
            or C.ACTION_CHUNK_LEN
        )
        if resolved_chunk_len != cfg.ckpt.expected_action_chunk_len:
            raise HardFailAssertion(
                f"resolved action_chunk_len={resolved_chunk_len} != "
                f"expected {cfg.ckpt.expected_action_chunk_len}"
            )
        unk = cfg.ckpt.expected_unnorm_key
        action_mean = norm_stats[unk]["action"]["mean"]
        if len(action_mean) != cfg.ckpt.expected_action_dim:
            raise HardFailAssertion(
                f"len(norm_stats.action.mean)={len(action_mean)} != "
                f"expected_action_dim={cfg.ckpt.expected_action_dim}"
            )
        proprio_mean = norm_stats[unk]["proprio"]["mean"]
        if len(proprio_mean) != cfg.ckpt.expected_proprio_dim:
            raise HardFailAssertion(
                f"len(norm_stats.proprio.mean)={len(proprio_mean)} != "
                f"expected_proprio_dim={cfg.ckpt.expected_proprio_dim}"
            )
        if cfg.proprio.adapt.output_dim != cfg.ckpt.expected_proprio_dim:
            raise HardFailAssertion(
                f"deploy.proprio.adapt.output_dim={cfg.proprio.adapt.output_dim} != "
                f"expected_proprio_dim={cfg.ckpt.expected_proprio_dim}"
            )
        # Frame + gripper compatibility (hard-fail unless wire_only_smoke=True).
        # Spec §Section 4 step 4: frame mismatch + gripper degeneracy both
        # required to fire.
        if (
            cfg.action.native.frame != cfg.action.contract.frame
            and cfg.action.frame_conversion.method == "none"
            and not cfg.wire_only_smoke
        ):
            raise HardFailAssertion(
                f"native.frame={cfg.action.native.frame!r} != contract.frame="
                f"{cfg.action.contract.frame!r} with frame_conversion=none. "
                "Set wire_only_smoke=true to bypass for smoke testing."
            )
        n_g = cfg.action.native.gripper
        c_g = cfg.action.contract.gripper
        if n_g.sign.open == n_g.sign.closed and not cfg.wire_only_smoke:
            raise HardFailAssertion(
                f"native.gripper.sign.open == sign.closed ({n_g.sign.open}); "
                "gripper remap is degenerate"
            )
        if c_g.sign.open == c_g.sign.closed and not cfg.wire_only_smoke:
            raise HardFailAssertion(
                f"contract.gripper.sign.open == sign.closed ({c_g.sign.open}); "
                "gripper remap is degenerate"
            )
        # Wrist requirement (hard / soft path, per spec §Section 5 line 354)
        bridge_or_dinov2 = (
            m_model.get("use_wrist_bridge", False)
            or m_model.get("use_scene_wrist_dinov2_llm", False)
            or m_model.get("wrist_dinov2", False)
        )
        wrist_in_llm = m_model.get("wrist_in_llm", False)
        dropout = float(m_model.get("wrist_view_dropout_p") or 0.0)
        wrist_field = cfg.request_fields.wrist_image
        if bridge_or_dinov2 and not wrist_field:
            raise HardFailAssertion(
                "ckpt requires wrist (use_wrist_bridge or DINOv2 path); "
                "deploy.request_fields.wrist_image must be set"
            )
        if wrist_in_llm and dropout == 0.0 and not wrist_field:
            raise HardFailAssertion(
                "ckpt requires wrist (wrist_in_llm with no dropout); "
                "deploy.request_fields.wrist_image must be set"
            )

    @staticmethod
    def validate_startup_hold_position(
        cfg: DeployConfig,
        *,
        domain_id: int,
    ) -> None:
        if domain_id < 0:
            raise HardFailAssertion(
                f"domain_id={domain_id} must be >= 0 (upper bound only "
                "checkable in xvla_adapter mode)"
            )
        # deploy-yaml-internal asserts only; ckpt-derived asserts skipped.
        if cfg.proprio.source.total_dim != sum(c.dims for c in cfg.proprio.source.components):
            raise HardFailAssertion("proprio.source.total_dim != sum(components.dims)")
        if cfg.ckpt.expected_action_dim != 7:
            raise HardFailAssertion(
                f"expected_action_dim={cfg.ckpt.expected_action_dim} != 7 (MVP fixed)"
            )
        if cfg.ckpt.expected_action_chunk_len <= 0:
            raise HardFailAssertion(
                f"expected_action_chunk_len={cfg.ckpt.expected_action_chunk_len} must be > 0"
            )
        # Gripper compat (linear remap requires non-degenerate native sign).
        n = cfg.action.native.gripper
        if n.sign.open == n.sign.closed:
            raise HardFailAssertion(
                f"native.gripper.sign.open == sign.closed ({n.sign.open}); "
                "gripper remap is degenerate"
            )
