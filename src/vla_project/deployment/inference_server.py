"""Yamlless inference HTTP server.

See docs/superpowers/specs/2026-05-16-yamlless-hf-deploy-design.md.
"""
from __future__ import annotations

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

from vla_project.data import constants as C
from vla_project.deployment.metadata import build_metadata_response
from vla_project.deployment.predictors.base import ChunkPredictor
from vla_project.deployment.predictors.hold_position import HoldPositionChunkPredictor
from vla_project.deployment.predictors.xvla_adapter import XVLAAdapterChunkPredictor
from vla_project.deployment.runtime import ModelRuntime
from vla_project.deployment.schemas import PredictRequest, PredictResponse
from vla_project.deployment.startup_validation import (
    HardFailAssertion,
    derive_wrist_hard_required,
    resolve_unnorm_key,
    validate_runtime,
)
from vla_project.deployment.wire_io import (
    decode_jpeg_b64,
    normalize_proprio_q99,
)

logger = logging.getLogger("vla_project.deployment")


def build_app(
    *,
    checkpoint: str | Path,
    predictor_kind: Literal["xvla_adapter", "hold_position"] = "xvla_adapter",
    domain_id: int | None = None,
    unnorm_key: str | None = None,
    trust_checkpoint_code: bool = False,
    device: str = "cuda:0",
    dtype: str = "bf16",
    torch_compile: str = "off",
    warmup_iters: int = 1,
    inject_sleep_s: float = 0.0,
) -> FastAPI:
    if predictor_kind == "xvla_adapter":
        runtime = ModelRuntime.from_export(
            checkpoint, device=device, dtype=dtype,
            torch_compile=torch_compile, warmup_iters=warmup_iters,
            trust_checkpoint_code=trust_checkpoint_code,
        )
    else:
        # hold_position: meta + post_process only, no model load
        runtime = ModelRuntime.from_meta_only(
            checkpoint, trust_checkpoint_code=trust_checkpoint_code,
        )

    meta = runtime.meta_raw
    resolved_unnorm_key = resolve_unnorm_key(meta, override=unnorm_key)
    if domain_id is None:
        domain_id = int(meta["cfg"]["data"]["domain_id"])

    action_dim = C.ACTION_DIM
    validate_runtime(
        meta, unnorm_key=resolved_unnorm_key,
        domain_id=domain_id, model_action_dim=action_dim,
    )
    wrist_hard_required = derive_wrist_hard_required(meta)

    action_chunk_len = int(
        meta["cfg"].get("data", {}).get("action_chunk_len")
        or meta["cfg"].get("model", {}).get("action_chunk_len")
        or C.ACTION_CHUNK_LEN
    )
    if predictor_kind == "hold_position":
        predictor: ChunkPredictor = HoldPositionChunkPredictor(
            chunk_len=action_chunk_len,
            action_dim=action_dim,
            gripper_native_midpoint=0.5,
        )
    else:
        predictor = XVLAAdapterChunkPredictor(
            runtime=runtime,
            tokenizer=runtime.tokenizer,
            image_transform=runtime.image_transform,
            action_q99=meta["norm_stats"][resolved_unnorm_key]["action"],
            action_chunk_len=action_chunk_len,
            action_dim=action_dim,
            domain_id=domain_id,
        )

    post_process_fn = runtime.post_process_fn
    post_process_path = runtime.post_process_path
    proprio_stats = meta["norm_stats"][resolved_unnorm_key]["proprio"]

    app = FastAPI(title="GEM-4-VLA Inference Server")
    state = {
        "predictor_class": type(predictor).__name__,
        "ready_at_ns": time.monotonic_ns(),
        "inject_sleep_s": float(inject_sleep_s),
    }

    @app.get("/healthz")
    async def healthz() -> dict:
        return {
            "status": "ok",
            "predictor": state["predictor_class"],
            "ready_at_ns": state["ready_at_ns"],
        }

    @app.get("/metadata")
    async def metadata_route() -> dict:
        return build_metadata_response(
            meta,
            unnorm_key=resolved_unnorm_key,
            domain_id=domain_id,
            has_post_process=post_process_fn is not None,
            post_process_path=post_process_path,
        )

    @app.exception_handler(RequestValidationError)
    async def _on_validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        errors = jsonable_encoder(exc.errors())
        # Surface validation errors in the server log so operators can debug
        # 422s without inspecting the client response body. Each entry has
        # `loc` (field path), `type` (error kind), and `msg` (human-readable).
        summary = "; ".join(
            f"{'.'.join(str(p) for p in e.get('loc', []))}: {e.get('type')} ({e.get('msg')})"
            for e in errors
        )
        logger.warning("validation_error path=%s errors=[%s]", request.url.path, summary)
        return JSONResponse(status_code=422, content={"detail": errors})

    @app.post("/predict", response_model=PredictResponse)
    async def predict(req: PredictRequest, request: Request) -> PredictResponse:
        request_id = request.headers.get("X-Request-Id") or uuid.uuid4().hex
        t0 = time.monotonic_ns()
        # Bound the per-request log size: proprio is normally 7-14 floats,
        # but Pydantic doesn't cap the list, so log length + truncated sample
        # instead of the full vector to keep the log line bounded even if a
        # caller submits an oversized payload (validation rejects it later).
        proprio_len = len(req.proprio)
        proprio_sample = list(req.proprio[:8])
        logger.info(
            "predict_request request_id=%s instruction=%r"
            " proprio_len=%d proprio_head=%s"
            " image_primary_b64_len=%d image_wrist_b64_len=%s model_version=%s",
            request_id,
            req.instruction,
            proprio_len,
            proprio_sample,
            len(req.image_primary),
            len(req.image_wrist) if req.image_wrist is not None else None,
            req.model_version,
        )
        try:
            scene = decode_jpeg_b64(req.image_primary)
            wrist_was_provided = req.image_wrist is not None
            if wrist_was_provided:
                wrist = decode_jpeg_b64(req.image_wrist)
            elif wrist_hard_required:
                raise ValueError(
                    "checkpoint requires wrist_image but request omitted it"
                )
            else:
                wrist = np.zeros((224, 224, 3), dtype=np.uint8)
            logger.info(
                "predict_decoded request_id=%s scene_shape=%s wrist_shape=%s"
                " wrist_provided=%s",
                request_id, scene.shape, wrist.shape, wrist_was_provided,
            )

            proprio_raw = np.asarray(req.proprio, dtype=np.float32)
            if len(proprio_raw) != len(proprio_stats["q99"]):
                raise ValueError(
                    f"proprio length {len(proprio_raw)} != expected "
                    f"{len(proprio_stats['q99'])}"
                )
            proprio_norm, _ = normalize_proprio_q99(proprio_raw, proprio_stats)

            obs = {
                "scene_image": scene,
                "wrist_image": wrist,
                "wrist_was_provided": wrist_was_provided,
                "proprio": proprio_norm,
                "language": req.instruction,
            }
        except Exception as e:
            elapsed_ms = (time.monotonic_ns() - t0) / 1e6
            logger.warning(
                f"request_invalid request_id={request_id} elapsed_ms={elapsed_ms:.1f}"
                f" error={type(e).__name__}: {e}"
            )
            raise HTTPException(status_code=422, detail=str(e)) from e

        if state["inject_sleep_s"] > 0:
            import asyncio
            await asyncio.sleep(state["inject_sleep_s"])

        try:
            native = predictor.predict(obs)
        except Exception as e:
            elapsed_ms = (time.monotonic_ns() - t0) / 1e6
            logger.error(
                f"predictor_error request_id={request_id} elapsed_ms={elapsed_ms:.1f}"
                f" error={type(e).__name__}: {e}",
                exc_info=True,
            )
            raise HTTPException(status_code=500, detail=str(e)) from e

        if np.isnan(native).any():
            elapsed_ms = (time.monotonic_ns() - t0) / 1e6
            logger.error(
                f"predictor_nan request_id={request_id} elapsed_ms={elapsed_ms:.1f}"
            )
            raise HTTPException(status_code=500, detail="predictor emitted NaN")

        if post_process_fn is not None:
            try:
                native = post_process_fn(native, meta)
            except Exception as e:
                elapsed_ms = (time.monotonic_ns() - t0) / 1e6
                logger.error(
                    f"postprocess_error request_id={request_id} elapsed_ms={elapsed_ms:.1f}"
                    f" error={type(e).__name__}: {e}",
                    exc_info=True,
                )
                raise HTTPException(status_code=500, detail=f"post_process: {e}") from e
            if not isinstance(native, np.ndarray) or native.shape[-1] != action_dim:
                elapsed_ms = (time.monotonic_ns() - t0) / 1e6
                logger.error(
                    f"postprocess_bad_shape request_id={request_id} elapsed_ms={elapsed_ms:.1f}"
                    f" shape={getattr(native, 'shape', None)}"
                )
                raise HTTPException(
                    status_code=500,
                    detail=f"post_process returned bad shape {getattr(native, 'shape', None)}",
                )
            if np.isnan(native).any():
                elapsed_ms = (time.monotonic_ns() - t0) / 1e6
                logger.error(
                    f"postprocess_nan request_id={request_id} elapsed_ms={elapsed_ms:.1f}"
                )
                raise HTTPException(status_code=500, detail="post_process emitted NaN")

        actions = native.astype(np.float32).tolist()
        elapsed_ms = (time.monotonic_ns() - t0) / 1e6
        logger.info(
            f"predict ok request_id={request_id} elapsed_ms={elapsed_ms:.1f}"
        )
        return PredictResponse(actions=actions)

    return app
