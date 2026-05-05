"""Drop-in projector modules matching vla-gemma-4 wristb_b16_v2 baseline.

The baseline (73% LIBERO, `libero_b_siglip_10k_wristb_b16_v2`) uses
non-domain-aware MLPs for scene and proprio projection — a structural
delta vs our default :class:`DomainAwareLinear` that materially differs
in capacity (3-MLP w/ 8192-dim intermediate vs single Linear). When a
config sets ``use_baseline_projectors=True`` we swap to these classes.

The forward signature accepts an optional ``domain_id`` for callsite
compatibility with :class:`DomainAwareLinear`; the value is ignored.
This lets ``VLAPolicy.forward`` keep a single call shape regardless of
projector backend.

Reference: ``vla-gemma-4/VLA-Adapter/prismatic/extern/hf/modeling_prismatic_gemma4.py``
``VisionProjector`` (lines 81-97) and ``ProprioProjector`` (lines 6-24
in ``projectors.py``).
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class BaselineSceneProjector(nn.Module):
    """3-layer MLP: vision_dim → init_proj_dim → llm_dim → llm_dim with GELU.

    Matches ``VisionProjector(vision_dim, llm_dim, initial_projection_dim=8192)``
    from vla-gemma-4. ~32 M params at the default 1152→8192→1536→1536.
    """

    def __init__(
        self,
        vision_dim: int,
        llm_dim: int,
        initial_projection_dim: int = 8192,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(vision_dim, initial_projection_dim, bias=True)
        self.fc2 = nn.Linear(initial_projection_dim, llm_dim, bias=True)
        self.fc3 = nn.Linear(llm_dim, llm_dim, bias=True)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor, domain_id: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.fc3(self.act(self.fc2(self.act(self.fc1(x)))))


class BaselineProprioProjector(nn.Module):
    """2-layer MLP: proprio_dim → llm_dim → llm_dim with one GELU between.

    Matches ``ProprioProjector(proprio_dim, llm_dim)`` from vla-gemma-4.
    No activation after fc2.
    """

    def __init__(self, proprio_dim: int, llm_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(proprio_dim, llm_dim, bias=True)
        self.fc2 = nn.Linear(llm_dim, llm_dim, bias=True)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor, domain_id: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))
