"""FastAPI app factory + /predict + /healthz routes.

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

from vla_project.deployment.domain_adapter import (
    DomainAdapter,
    load_deploy_config,
)
from vla_project.deployment.predictors.base import ChunkPredictor
from vla_project.deployment.predictors.hold_position import HoldPositionChunkPredictor
from vla_project.deployment.predictors.xvla_adapter import XVLAAdapterChunkPredictor
from vla_project.deployment.runtime import ModelRuntime
from vla_project.deployment.schemas import PredictRequest, PredictResponse


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

    @app.post("/predict", response_model=PredictResponse)
    async def predict(req: PredictRequest, request: Request) -> PredictResponse:
        request_id = request.headers.get("X-Request-Id") or uuid.uuid4().hex
        t0 = time.monotonic_ns()
        outcome: str = "ok"
        error_class: str | None = None
        error_msg: str | None = None
        try:
            obs = adapter.preprocess(req)
        except (ValueError, AssertionError) as e:
            outcome = "invalid_request"
            error_class = type(e).__name__
            error_msg = str(e)
            elapsed_ms = (time.monotonic_ns() - t0) / 1e6
            _log_request(request_id, elapsed_ms, state, outcome, error_class, error_msg)
            raise HTTPException(status_code=422, detail=str(e)) from e

        # Optional injected sleep for slow-path smoke (test-only).
        if state["inject_sleep_s"] > 0:
            import asyncio
            await asyncio.sleep(state["inject_sleep_s"])

        try:
            native = predictor.predict(obs)
        except NotImplementedError as e:
            outcome = "predictor_error"
            error_class = "NotImplementedError"
            error_msg = str(e)
            elapsed_ms = (time.monotonic_ns() - t0) / 1e6
            _log_request(request_id, elapsed_ms, state, outcome, error_class, error_msg)
            raise HTTPException(status_code=500, detail=str(e)) from e
        except Exception as e:  # noqa: BLE001
            outcome = "predictor_error"
            error_class = type(e).__name__
            error_msg = str(e)
            elapsed_ms = (time.monotonic_ns() - t0) / 1e6
            _log_request(request_id, elapsed_ms, state, outcome, error_class, error_msg)
            raise HTTPException(status_code=500, detail=str(e)) from e

        try:
            if np.isnan(native).any():
                raise ValueError("predictor emitted NaN")
            actions = adapter.postprocess(native)
        except (ValueError, AssertionError, NotImplementedError) as e:
            outcome = "postprocess_error"
            error_class = type(e).__name__
            error_msg = str(e)
            elapsed_ms = (time.monotonic_ns() - t0) / 1e6
            _log_request(request_id, elapsed_ms, state, outcome, error_class, error_msg)
            raise HTTPException(status_code=500, detail=str(e)) from e

        elapsed_ms = (time.monotonic_ns() - t0) / 1e6
        _log_request(request_id, elapsed_ms, state, outcome, None, None)
        return PredictResponse(actions=actions)

    return app


def _log_request(
    request_id: str,
    elapsed_ms: float,
    state: dict,
    outcome: str,
    error_class: str | None,
    error_msg: str | None,
) -> None:
    payload = {
        "ts_ns": time.monotonic_ns(),
        "request_id": request_id,
        "elapsed_ms": round(elapsed_ms, 3),
        "predictor": state["predictor_class"],
        "outcome": outcome,
    }
    if elapsed_ms > _LATENCY_BUDGET_MS:
        payload["latency_budget_ms"] = _LATENCY_BUDGET_MS
        payload["latency_budget_exceeded"] = True
    if error_class:
        payload["error_class"] = error_class
        payload["error_msg"] = error_msg
    if outcome != "ok":
        logger.warning(json.dumps(payload))
    else:
        logger.info(json.dumps(payload))
