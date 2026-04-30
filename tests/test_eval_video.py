"""Frame capture in run_episode + video writing in evaluate_libero."""
from pathlib import Path
from typing import Any, Dict

import numpy as np

from vla_project.evaluation.libero_eval import evaluate_libero
from vla_project.evaluation.rollout import EpisodeResult, run_episode
from vla_project.policies.base_policy import BasePolicy
from vla_project.robots.base_robot import BaseRobot


class _FakePolicy(BasePolicy):
    def select_action(self, obs: Dict[str, Any]) -> np.ndarray:
        return np.zeros(7, dtype=np.float32)

    def reset(self) -> None:
        pass


class _FakeRobot(BaseRobot):
    def __init__(self, *_a, succeed_at: int = -1, **_kw) -> None:
        self.t = 0
        self.connected = False
        self.succeed_at = succeed_at

    def _obs(self) -> Dict[str, Any]:
        # Distinct image per timestep so the captured frames are not all-zero.
        img = np.full((32, 32, 3), self.t % 256, dtype=np.uint8)
        return {
            "scene_image": img,
            "wrist_image": np.zeros((32, 32, 3), dtype=np.uint8),
            "proprio":     np.zeros(8, dtype=np.float32),
            "language":    "fake",
        }

    def connect(self) -> None:
        self.connected = True

    def reset(self) -> Dict[str, Any]:
        self.t = 0
        return self._obs()

    def get_observation(self) -> Dict[str, Any]:
        return self._obs()

    def send_action(self, action: np.ndarray) -> Dict[str, Any]:
        self.t += 1
        return self._obs()

    def check_success(self) -> bool:
        return 0 <= self.succeed_at <= self.t

    def close(self) -> None:
        self.connected = False


def test_run_episode_default_no_frames() -> None:
    p = _FakePolicy()
    r = _FakeRobot(succeed_at=-1)
    r.connect()
    out = run_episode(p, r, max_steps=5, num_steps_wait=2)
    r.close()
    assert isinstance(out, EpisodeResult)
    assert out.frames == []


def test_run_episode_captures_frames() -> None:
    p = _FakePolicy()
    r = _FakeRobot(succeed_at=-1)
    r.connect()
    out = run_episode(p, r, max_steps=5, num_steps_wait=2, capture_frames=True)
    r.close()
    # initial reset obs + 2 warmup + 5 policy steps = 8 frames
    assert len(out.frames) == 1 + 2 + 5
    assert all(f.shape == (32, 32, 3) and f.dtype == np.uint8 for f in out.frames)


def test_evaluate_libero_writes_gif(tmp_path: Path) -> None:
    p = _FakePolicy()
    out = evaluate_libero(
        policy=p,
        robot_factory=lambda ti: _FakeRobot(succeed_at=-1),
        task_idxs=[0, 1],
        num_episodes_per_task=2,
        max_steps=4,
        num_steps_wait=2,
        video_dir=tmp_path,
        video_fps=5,
    )
    files = sorted(p.name for p in tmp_path.iterdir())
    assert files == [
        "task_0_ep0.gif",
        "task_0_ep1.gif",
        "task_1_ep0.gif",
        "task_1_ep1.gif",
    ]
    for f in files:
        assert (tmp_path / f).stat().st_size > 0
    assert out["overall"]["num_episodes"] == 4


def test_evaluate_libero_no_video_dir_no_files(tmp_path: Path) -> None:
    p = _FakePolicy()
    evaluate_libero(
        policy=p,
        robot_factory=lambda ti: _FakeRobot(succeed_at=-1),
        task_idxs=[0],
        num_episodes_per_task=1,
        max_steps=3,
        num_steps_wait=1,
        video_dir=None,
    )
    assert sorted(p.name for p in tmp_path.iterdir()) == []
