"""Entry point for the inference HTTP server.

Run:
  uv run python scripts/serve.py \\
    --predictor hold_position \\
    --deploy-config configs/deploy/v36_libero_spatial.yaml \\
    --domain-id 0 \\
    --port 8001

For xvla_adapter mode (Phase 1 — predict() returns 500 NotImplementedError today):
  uv run python scripts/serve.py \\
    --predictor xvla_adapter \\
    --checkpoint /path/to/v36_export \\
    --deploy-config configs/deploy/v36_libero_spatial.yaml \\
    --domain-id 0 \\
    --port 8001
"""
from __future__ import annotations

import argparse
import logging
import sys

import uvicorn

from vla_project.deployment.inference_server import build_app


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="X-VLA-Adapter inference HTTP server")
    ap.add_argument("--predictor", choices=["hold_position", "xvla_adapter"], required=True)
    ap.add_argument("--checkpoint", required=False, default=None,
                    help="ckpt export dir (required iff --predictor xvla_adapter)")
    ap.add_argument("--deploy-config", required=True)
    ap.add_argument("--domain-id", type=int, required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--inject-sleep", type=float, default=0.0,
                    help="test-only: sleep N seconds before predict to exercise the latency log path")
    args = ap.parse_args(argv)

    if args.predictor == "xvla_adapter" and args.checkpoint is None:
        ap.error("--checkpoint required when --predictor xvla_adapter")

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    app = build_app(
        predictor_kind=args.predictor,
        checkpoint=args.checkpoint,
        deploy_config_path=args.deploy_config,
        domain_id=args.domain_id,
        inject_sleep_s=args.inject_sleep,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    sys.exit(main())
