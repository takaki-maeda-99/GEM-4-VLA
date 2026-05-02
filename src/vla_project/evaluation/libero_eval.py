"""Top-level closed-loop evaluation orchestrator for LIBERO."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np

from vla_project.evaluation.metrics import aggregate_episodes
from vla_project.evaluation.rollout import EpisodeResult, run_episode
from vla_project.policies.base_policy import BasePolicy
from vla_project.robots.base_robot import BaseRobot


def _write_video(path: Path, frames: List[np.ndarray], fps: int = 10) -> None:
    """Write frames to disk as a GIF (or MP4 if path ends in .mp4 and
    imageio-ffmpeg is available). Best-effort: missing imageio silently
    skips the write."""
    if not frames:
        return
    try:
        import imageio.v2 as imageio
    except ImportError:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".gif":
        # GIF wants duration in ms per frame; imageio's fps kwarg is
        # deprecated for the pillow-backed GIF writer.
        imageio.mimsave(str(path), frames, duration=int(round(1000 / max(fps, 1))))
    else:
        imageio.mimsave(str(path), frames, fps=fps)


def evaluate_libero(
    policy: BasePolicy,
    robot_factory: Callable[[int], BaseRobot],
    task_idxs: List[int],
    *,
    num_episodes_per_task: int,
    max_steps: int,
    num_steps_wait: int = 10,
    task_label_fn: Callable[[int], str] = (lambda i: f"task_{i}"),
    video_dir: Optional[Union[str, Path]] = None,
    video_fps: int = 10,
) -> Dict[str, Any]:
    """Run closed-loop evaluation across tasks and aggregate metrics.

    Args:
        video_dir: when set, captures rollout frames and writes one GIF per
            episode to ``<video_dir>/<task_label>_ep<N>.gif``. Best-effort:
            missing imageio installs silently skip writing.
    """
    records: List[Dict[str, Any]] = []
    capture = video_dir is not None
    out_dir = Path(video_dir) if video_dir is not None else None
    for ti in task_idxs:
        robot = robot_factory(ti)
        robot.connect()
        try:
            for ep in range(num_episodes_per_task):
                # Use the LIBERO benchmark's standard init_state for this
                # (task, episode) pair (no-op if the robot doesn't support
                # set_episode_idx). Rollouts without this run on random
                # scenes that are out-of-distribution for the trained model.
                if hasattr(robot, "set_episode_idx"):
                    robot.set_episode_idx(ep)
                result: EpisodeResult = run_episode(
                    policy=policy,
                    robot=robot,
                    max_steps=max_steps,
                    num_steps_wait=num_steps_wait,
                    capture_frames=capture,
                )
                label = task_label_fn(ti)
                records.append({"task": label, "result": result})
                if out_dir is not None and result.frames:
                    _write_video(
                        out_dir / f"{label}_ep{ep}.gif",
                        result.frames,
                        fps=video_fps,
                    )
        finally:
            robot.close()
    out = aggregate_episodes(records)
    out["records"] = records
    return out
