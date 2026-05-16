"""ChunkPredictor ABC.

A predictor takes a fully-prepped Obs dict (with JPEG-decoded images,
normalized proprio, and language instruction) and returns a (T, A) np.float32
chunk in MODEL NATIVE physical units. The caller handles frame / gripper-convention
conversion to MimicRec contract units.

For v36 (and v33/v35 RLDS-trained), native gripper is normalized_0_1
(closed=0, open=1), frame is LIBERO world frame, deltas are
meter+axisangle_rad. See docs/superpowers/specs/2026-05-06-vla-inference-server-design.md
§Section 5.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class ChunkPredictor(ABC):
    @abstractmethod
    def predict(self, obs: dict[str, Any]) -> np.ndarray:
        """Return one chunk in NATIVE units, shape (T, A) np.float32."""

    @property
    @abstractmethod
    def chunk_len(self) -> int: ...

    @property
    @abstractmethod
    def action_dim(self) -> int: ...
