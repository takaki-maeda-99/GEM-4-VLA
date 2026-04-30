"""Abstract base for robot wrappers.

All concrete subclasses (sim or real) expose the same four-method contract:

  - ``connect()``           — open whatever resource the wrapper needs
                              (sim env, ROS connection, hardware handle).
  - ``reset()``             — start a new episode. Return the first obs.
  - ``get_observation()``   — sample current obs without stepping.
  - ``send_action(action)`` — apply one action; return the next obs.
  - ``close()``             — release resources.

The observation dict has at minimum::

  {
    "scene_image": np.ndarray[H, W, 3]  uint8,
    "wrist_image": np.ndarray[H, W, 3]  uint8,
    "proprio":     np.ndarray[D]        float32,
    "language":    str,
  }

This matches the dataset-side schema closely so that ``XVLAAdapterPolicy``
can consume robot obs with no extra glue.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

import numpy as np


class BaseRobot(ABC):
    @abstractmethod
    def connect(self) -> None:
        """Open the underlying sim env / hardware handle."""
        ...

    @abstractmethod
    def reset(self) -> Dict[str, Any]:
        """Start a new episode. Return the initial observation."""
        ...

    @abstractmethod
    def get_observation(self) -> Dict[str, Any]:
        """Return the current observation without stepping the env."""
        ...

    @abstractmethod
    def send_action(self, action: np.ndarray) -> Dict[str, Any]:
        """Apply one action and return the resulting observation."""
        ...

    @abstractmethod
    def close(self) -> None:
        """Release any resources opened by ``connect()``."""
        ...
