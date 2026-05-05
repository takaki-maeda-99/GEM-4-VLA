"""HoldPositionChunkPredictor — wire-shape smoke / pre-model-trained sentinel.

NOT a production safety fallback (MimicRec's slow-stop ramp is the real
fallback). Emits zero ee_delta for cols 0..5 and `gripper_native_midpoint`
for col 6 (in MODEL NATIVE gripper units; postprocess converts to contract).

For v36 (and v33/v35) native = normalized_0_1 (closed=0, open=1), midpoint
0.5 lands on contract midpoint. For signed_neg1_pos1 native, set 0.0.

See docs/superpowers/specs/2026-05-06-vla-inference-server-design.md
§Section 5 (HoldPositionChunkPredictor).
"""
from __future__ import annotations

from typing import Any

import numpy as np

from vla_project.deployment.predictors.base import ChunkPredictor


class HoldPositionChunkPredictor(ChunkPredictor):
    def __init__(
        self,
        chunk_len: int,
        action_dim: int,
        gripper_native_midpoint: float = 0.5,
    ) -> None:
        self._T = int(chunk_len)
        self._A = int(action_dim)
        self._g = float(gripper_native_midpoint)

    @property
    def chunk_len(self) -> int:
        return self._T

    @property
    def action_dim(self) -> int:
        return self._A

    def predict(self, obs: dict[str, Any]) -> np.ndarray:  # noqa: ARG002 (obs unused)
        a = np.zeros((self._T, self._A), dtype=np.float32)
        a[:, -1] = self._g
        return a
