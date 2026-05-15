"""Entry point for the inference HTTP server (yaml-less).

Run with a Hugging Face checkpoint id:
  uv run python scripts/serve.py \\
    --checkpoint takaki99/GEM-4-FT-bottle \\
    --port 8001

Or a local checkpoint dir:
  uv run python scripts/serve.py \\
    --checkpoint outputs/run/checkpoints/step_2000 \\
    --port 8001

See docs/superpowers/specs/2026-05-16-yamlless-hf-deploy-design.md.
"""
from __future__ import annotations

import argparse
import logging
import sys

import uvicorn

from vla_project.deployment.inference_server import build_app


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="X-VLA-Adapter inference HTTP server")
    ap.add_argument("--checkpoint", required=True,
                    help="local ckpt dir, HF id 'org/repo', or 'org/repo/subfolder'")
    ap.add_argument("--predictor", choices=["hold_position", "xvla_adapter"],
                    default="xvla_adapter")
    ap.add_argument("--domain-id", type=int, default=None,
                    help="defaults to cfg.data.domain_id from meta.json")
    ap.add_argument("--unnorm-key", default=None,
                    help="required iff meta.norm_stats has >1 keys")
    ap.add_argument("--trust-checkpoint-code", action="store_true",
                    help="opt-in to load post_process.py from HF-resolved ckpts")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    ap.add_argument("--torch-compile", default="off",
                    choices=["off", "reduce-overhead", "default"])
    ap.add_argument("--warmup-iters", type=int, default=1)
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        app = build_app(
            checkpoint=args.checkpoint,
            predictor_kind=args.predictor,
            domain_id=args.domain_id,
            unnorm_key=args.unnorm_key,
            trust_checkpoint_code=args.trust_checkpoint_code,
            device=args.device,
            dtype=args.dtype,
            torch_compile=args.torch_compile,
            warmup_iters=args.warmup_iters,
        )
    except ValueError as e:
        ap.error(str(e))
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    sys.exit(main())
