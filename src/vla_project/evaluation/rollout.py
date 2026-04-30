"""Single-episode rollout glue for closed-loop evaluation.

Mirrors the OpenVLA / X-VLA reference convention:
  - ``num_steps_wait`` warm-up env steps with zero action while the env
    stabilizes (gripper settling, camera frame populating, etc.).
  - then up to ``max_steps`` env steps where each action comes from
    ``policy.select_action(obs)``.
  - the loop short-circuits as soon as ``robot.check_success()`` returns True.

The chunk buffer (open-loop action chunking) lives inside the policy, so the
rollout calls ``select_action`` once per env step regardless of chunk length.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from vla_project.policies.base_policy import BasePolicy
from vla_project.robots.base_robot import BaseRobot


@dataclass
class EpisodeResult:
    success: bool
    num_env_steps: int
    num_policy_calls: int
    elapsed_s: float
    frames: List[np.ndarray] = field(default_factory=list)
    info: Dict[str, Any] = field(default_factory=dict)


def _zero_action(dim: int = 7) -> np.ndarray:
    return np.zeros(dim, dtype=np.float32)


def run_episode(
    policy: BasePolicy,
    robot: BaseRobot,
    *,
    max_steps: int,
    num_steps_wait: int = 10,
    action_dim: int = 7,
    success_check: Optional[Callable[[BaseRobot, Dict[str, Any]], bool]] = None,
    capture_frames: bool = False,
) -> EpisodeResult:
    """Run one closed-loop episode.

    When ``capture_frames=True``, ``EpisodeResult.frames`` is populated with
    one ``obs["scene_image"]`` snapshot per env step (including warm-up).
    """
    t0 = time.perf_counter()
    check = success_check or (lambda r, _o: r.check_success())
    success = False
    num_policy_calls = 0
    num_env_steps = 0
    frames: List[np.ndarray] = []

    obs = robot.reset()
    policy.reset()

    def _maybe_capture(o: Dict[str, Any]) -> None:
        if not capture_frames:
            return
        img = o.get("scene_image")
        if img is not None:
            frames.append(np.asarray(img).copy())

    _maybe_capture(obs)

    for _ in range(num_steps_wait):
        obs = robot.send_action(_zero_action(action_dim))
        num_env_steps += 1
        _maybe_capture(obs)
        if check(robot, obs):
            success = True
            break

    if not success:
        for _ in range(max_steps):
            action = policy.select_action(obs)
            num_policy_calls += 1
            obs = robot.send_action(action)
            num_env_steps += 1
            _maybe_capture(obs)
            if check(robot, obs):
                success = True
                break

    return EpisodeResult(
        success=success,
        num_env_steps=num_env_steps,
        num_policy_calls=num_policy_calls,
        elapsed_s=time.perf_counter() - t0,
        frames=frames,
        info={},
    )
