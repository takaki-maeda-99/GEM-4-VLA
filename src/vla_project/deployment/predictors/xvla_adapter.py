"""XVLAAdapterChunkPredictor — Phase 0 typed shell.

Constructor signature is frozen per spec §Section 5 line 466 so Phase 1
cannot drift the public API. predict() raises NotImplementedError.

Phase 1 implementation will follow XVLAAdapterPolicy._refill_buffer for
the forward path (SigLIP transform + tokenize + batch build, including
DINOv2 conditional keys + wrist_was_provided plumbing per spec §Section 5).
"""
from __future__ import annotations

from typing import Any

import numpy as np

from vla_project.deployment.predictors.base import ChunkPredictor


class XVLAAdapterChunkPredictor(ChunkPredictor):
    def __init__(
        self,
        runtime: Any,                  # Phase 1: ModelRuntime
        tokenizer: Any,                # Phase 1: GemmaPromptTokenizer
        image_transform: Any,          # Phase 1: SiglipImageTransform
        action_q99: Any,               # Phase 1: Q99Stats from meta.norm_stats
        action_chunk_len: int,
        action_dim: int,
        domain_id: int,
    ) -> None:
        self._T = int(action_chunk_len)
        self._A = int(action_dim)
        self.runtime = runtime
        self.tokenizer = tokenizer
        self.image_transform = image_transform
        self.action_q99 = action_q99
        self.domain_id = int(domain_id)

    @property
    def chunk_len(self) -> int:
        return self._T

    @property
    def action_dim(self) -> int:
        return self._A

    def predict(self, obs: dict[str, Any]) -> np.ndarray:
        raise NotImplementedError(
            "XVLAAdapterChunkPredictor.predict() is Phase 1 work. "
            "See spec §Section 5 lines 478-504."
        )
