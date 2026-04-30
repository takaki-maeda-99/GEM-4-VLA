"""Abstract base for runtime policy wrappers.

A policy maps a raw observation dict to a single executable action. Subclasses
encapsulate model loading, observation preprocessing, language tokenization,
action chunking, and denormalization. CLAUDE.md "Policy Structure" lists the
full responsibilities; this base codifies only the call contract so concrete
wrappers can vary in implementation.

The internal ``forward`` of the underlying model is not part of this
interface — policies are runtime glue, not training code.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

import numpy as np


class BasePolicy(ABC):
    @abstractmethod
    def select_action(self, obs: Dict[str, Any]) -> np.ndarray:
        """Return one executable action.

        Args:
            obs: dict with keys (at minimum):
                - ``scene_image``: ``np.ndarray[H, W, 3]`` uint8
                - ``wrist_image``: ``np.ndarray[H, W, 3]`` uint8
                - ``proprio``: ``np.ndarray[D]`` float32
                - ``language``: ``str``

        Returns:
            ``np.ndarray[A]`` float32.
        """
        ...

    @abstractmethod
    def reset(self) -> None:
        """Reset any internal buffer (action chunk queue, episode state).

        Called by the rollout loop at the start of each episode.
        """
        ...
