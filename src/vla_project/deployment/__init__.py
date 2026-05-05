"""HTTP inference server for X-VLA-Adapter checkpoints.

See docs/superpowers/specs/2026-05-06-vla-inference-server-design.md for the
full design. This package implements the Phase 0 HoldPosition path; the
XVLAAdapterChunkPredictor full forward path is Phase 1.
"""

from vla_project.deployment.inference_server import build_app

__all__ = ["build_app"]
