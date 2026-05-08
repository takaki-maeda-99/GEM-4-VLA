"""FastAPI app factory + /predict + /healthz + /admin/schema routes.

Reads deploy yaml + (optionally) ckpt meta.json, constructs DomainAdapter
and ChunkPredictor, mounts the FastAPI app. See spec §Section 6 for
HTTP code mapping, observability fields, and Phase 0 acceptance gate.
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from pathlib import Path
from typing import Literal

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from vla_project.deployment.domain_adapter import (
    DomainAdapter,
    IMAGE_MAX_SIDE,
    IMAGE_MIN_SIDE,
    PROPRIO_OOD_HARD_ABS,
    PROPRIO_OOD_WARN_ABS,
    load_deploy_config,
)
from vla_project.deployment.predictors.base import ChunkPredictor
from vla_project.deployment.predictors.hold_position import HoldPositionChunkPredictor
from vla_project.deployment.predictors.xvla_adapter import XVLAAdapterChunkPredictor
from vla_project.deployment.runtime import ModelRuntime
from vla_project.deployment.schemas import (
    INSTRUCTION_MAX_BYTES,
    PredictRequest,
    PredictResponse,
)


logger = logging.getLogger("vla_project.deployment")


_LATENCY_BUDGET_MS = 266.0  # spec §Section 3 latency budget breakdown


def build_app(
    *,
    predictor_kind: Literal["hold_position", "xvla_adapter"],
    checkpoint: str | Path | None,
    deploy_config_path: str | Path,
    domain_id: int,
    inject_sleep_s: float = 0.0,
) -> FastAPI:
    cfg = load_deploy_config(deploy_config_path)

    runtime: ModelRuntime | None = None
    norm_stats: dict | None = None

    if predictor_kind == "xvla_adapter":
        if checkpoint is None:
            raise ValueError("--checkpoint required when predictor_kind=xvla_adapter")
        runtime = ModelRuntime.from_export(
            checkpoint,
            device=cfg.runtime.device,
            dtype=cfg.runtime.dtype,
            torch_compile=cfg.runtime.torch_compile,
            warmup_iters=cfg.runtime.warmup_iters,
        )
        norm_stats = runtime.norm_stats
        DomainAdapter.validate_startup_xvla(
            cfg,
            meta_cfg=runtime.cfg,
            norm_stats=norm_stats,
            domain_id=domain_id,
        )
    else:
        DomainAdapter.validate_startup_hold_position(cfg, domain_id=domain_id)

    # Compute wrist_hard_required from the loaded ckpt cfg (None for hold_position).
    wrist_hard_required = False
    if predictor_kind == "xvla_adapter" and runtime is not None:
        m_model = runtime.cfg.get("model", {})
        wrist_hard_required = bool(
            m_model.get("use_wrist_bridge", False)
            or m_model.get("use_scene_wrist_dinov2_llm", False)
            or m_model.get("wrist_dinov2", False)
            or (m_model.get("wrist_in_llm", False) and float(m_model.get("wrist_view_dropout_p") or 0.0) == 0.0)
        )

    adapter = DomainAdapter(
        cfg=cfg,
        norm_stats=(norm_stats[cfg.ckpt.expected_unnorm_key] if norm_stats else None),
        domain_id=domain_id,
        wrist_hard_required=wrist_hard_required,
    )

    predictor: ChunkPredictor
    if predictor_kind == "hold_position":
        predictor = HoldPositionChunkPredictor(
            chunk_len=cfg.ckpt.expected_action_chunk_len,
            action_dim=cfg.ckpt.expected_action_dim,
            gripper_native_midpoint=cfg.holdposition.gripper_native_midpoint,
        )
    else:
        predictor = XVLAAdapterChunkPredictor(
            runtime=runtime,
            tokenizer=None,                 # Phase 1
            image_transform=None,           # Phase 1
            action_q99=norm_stats[cfg.ckpt.expected_unnorm_key]["action"] if norm_stats else None,
            action_chunk_len=cfg.ckpt.expected_action_chunk_len,
            action_dim=cfg.ckpt.expected_action_dim,
            domain_id=domain_id,
        )

    # ---- FastAPI app ----
    app = FastAPI(title="X-VLA-Adapter Inference Server")
    state_ready_at_ns = time.monotonic_ns()
    state = {
        "predictor_kind": predictor_kind,
        "predictor_class": type(predictor).__name__,
        "ready_at_ns": state_ready_at_ns,
        "inject_sleep_s": float(inject_sleep_s),
    }

    @app.get("/healthz")
    async def healthz() -> dict:
        return {
            "status": "ok",
            "predictor": state["predictor_class"],
            "ready_at_ns": state["ready_at_ns"],
        }

    @app.get("/admin/schema")
    async def admin_schema() -> dict:
        """F5: wire-contract introspection. Returns the minimum a client needs
        to construct valid requests + sanity-check shapes against this server's
        loaded config. See spec §F5 for field semantics.

        prompt.max_tokens is null in BOTH predictor modes for Phase 0 — the
        server does not tokenize yet, so reporting a max_tokens value would
        imply an enforced ceiling that does not exist. Phase 1 will populate
        the field when XVLAAdapterChunkPredictor.predict() actually
        tokenizes the instruction.
        """
        return {
            "predictor": state["predictor_class"],
            "ckpt": {
                "expected_unnorm_key": cfg.ckpt.expected_unnorm_key,
                "expected_action_chunk_len": cfg.ckpt.expected_action_chunk_len,
                "expected_action_dim": cfg.ckpt.expected_action_dim,
                "expected_proprio_dim": cfg.ckpt.expected_proprio_dim,
            },
            "wrist_hard_required": wrist_hard_required,
            "request_fields": {
                "scene_image": cfg.request_fields.scene_image,
                "wrist_image": cfg.request_fields.wrist_image,
                "proprio": cfg.request_fields.proprio,
                "instruction": cfg.request_fields.instruction,
            },
            "proprio": {
                "source": {
                    "components": [
                        {"name": c.name, "dims": c.dims, "units": c.units}
                        for c in cfg.proprio.source.components
                    ],
                    "total_dim": cfg.proprio.source.total_dim,
                },
            },
            "image": {"min_side": IMAGE_MIN_SIDE, "max_side": IMAGE_MAX_SIDE},
            "instruction": {"max_bytes": INSTRUCTION_MAX_BYTES},
            "prompt": {"max_tokens": None},  # Phase 0 deferral — see docstring
            "proprio_ood": {
                "warn_threshold": PROPRIO_OOD_WARN_ABS,
                "hard_threshold": PROPRIO_OOD_HARD_ABS,
            },
        }

    # Pydantic body-validation errors fire BEFORE the route body, so attach
    # an exception handler that logs them through our structured channel
    # rather than letting FastAPI's default 422 responder swallow the event.
    @app.exception_handler(RequestValidationError)
    async def _on_validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        # Pydantic v2 errors() includes ctx: {"error": ValueError(...)} when a
        # field_validator/model_validator raises ValueError. The exception
        # object is not JSON-serializable; jsonable_encoder converts it via
        # str(). Without this, F2/F4 validator failures would 500 instead of
        # returning a clean 422.
        errors = jsonable_encoder(exc.errors())
        request_id = request.headers.get("X-Request-Id") or uuid.uuid4().hex
        _log_request(
            request_id, elapsed_ms=0.0, state=state, domain_id=domain_id,
            outcome="invalid_request",
            error_class="RequestValidationError",
            error_msg=str(errors),
        )
        return JSONResponse(status_code=422, content={"detail": errors})

    @app.post("/predict", response_model=PredictResponse)
    async def predict(req: PredictRequest, request: Request) -> PredictResponse:
        request_id = request.headers.get("X-Request-Id") or uuid.uuid4().hex
        t0 = time.monotonic_ns()
        try:
            obs = adapter.preprocess(req)
        except Exception as e:  # noqa: BLE001 — invalid input is broad: ValueError, AssertionError,
            # binascii.Error (bad base64), PIL.UnidentifiedImageError (bad JPEG), etc.
            elapsed_ms = (time.monotonic_ns() - t0) / 1e6
            _log_request(
                request_id, elapsed_ms, state, domain_id,
                outcome="invalid_request",
                error_class=type(e).__name__,
                error_msg=str(e),
            )
            raise HTTPException(status_code=422, detail=str(e)) from e

        # Optional injected sleep for slow-path smoke (test-only).
        if state["inject_sleep_s"] > 0:
            import asyncio
            await asyncio.sleep(state["inject_sleep_s"])

        try:
            native = predictor.predict(obs)
        except Exception as e:  # noqa: BLE001 — NotImplementedError (Phase 0 stub) + torch errors
            elapsed_ms = (time.monotonic_ns() - t0) / 1e6
            _log_request(
                request_id, elapsed_ms, state, domain_id,
                outcome="predictor_error",
                error_class=type(e).__name__,
                error_msg=str(e),
                exc_info=True,
            )
            raise HTTPException(status_code=500, detail=str(e)) from e

        try:
            if np.isnan(native).any():
                raise ValueError("predictor emitted NaN before postprocess")
            actions = adapter.postprocess(native)
            # Spec-mandated NaN guard on the FINAL response actions
            # (postprocess can introduce NaN via gripper conversion / denorm).
            if any(any(_v != _v for _v in row) for row in actions):  # NaN check via != self
                raise ValueError("postprocess emitted NaN in final actions")
        except Exception as e:  # noqa: BLE001
            elapsed_ms = (time.monotonic_ns() - t0) / 1e6
            _log_request(
                request_id, elapsed_ms, state, domain_id,
                outcome="postprocess_error",
                error_class=type(e).__name__,
                error_msg=str(e),
                exc_info=True,
            )
            raise HTTPException(status_code=500, detail=str(e)) from e

        elapsed_ms = (time.monotonic_ns() - t0) / 1e6
        _log_request(request_id, elapsed_ms, state, domain_id, outcome="ok",
                     error_class=None, error_msg=None)
        return PredictResponse(actions=actions)

    return app


def _log_request(
    request_id: str,
    elapsed_ms: float,
    state: dict,
    domain_id: int,
    outcome: str,
    error_class: str | None,
    error_msg: str | None,
    *,
    exc_info: bool = False,
) -> None:
    payload = {
        "ts_ns": time.monotonic_ns(),
        "request_id": request_id,
        "elapsed_ms": round(elapsed_ms, 3),
        "predictor": state["predictor_class"],
        "domain_id": domain_id,
        "outcome": outcome,
    }
    over_budget = elapsed_ms > _LATENCY_BUDGET_MS
    if over_budget:
        payload["latency_budget_ms"] = _LATENCY_BUDGET_MS
        payload["latency_budget_exceeded"] = True
    if error_class:
        payload["error_class"] = error_class
        payload["error_msg"] = error_msg
    msg = json.dumps(payload)
    if outcome in ("predictor_error", "postprocess_error"):
        # Spec §6: 500 paths get error-level + full traceback (exc_info=True
        # forwards the active exception to the handler).
        logger.error(msg, exc_info=exc_info)
    elif outcome == "invalid_request" or over_budget:
        logger.warning(msg)
    else:
        logger.info(msg)
