"""LIBERO sim wrapper exposing the BaseRobot interface.

Lazily imports ``libero.libero.envs.OffScreenRenderEnv`` inside ``connect``,
so module load works even on machines without LIBERO / mujoco. Construction
captures a BDDL task spec; ``connect()`` instantiates the env; ``reset`` /
``send_action`` translate the LIBERO raw obs dict into the project's
``scene_image`` / ``wrist_image`` / ``proprio`` / ``language`` shape.

Reference: ``vla-gemma-4/scripts/gemma4/eval_libero_gemma4.py:97-133``.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from vla_project.robots.base_robot import BaseRobot


# Default LIBERO Python package location on dl40. Override via the
# constructor's ``libero_path`` kwarg or the LIBERO_PATH env var.
_DEFAULT_LIBERO_PATH = "/misc/dl00/takaki/vla-gemma-4/LIBERO"


def _ensure_libero_on_path(libero_path: Optional[str]) -> str:
    p = libero_path or os.environ.get("LIBERO_PATH", _DEFAULT_LIBERO_PATH)
    if p not in sys.path:
        sys.path.insert(0, p)
    return p


class LIBEROSimRobot(BaseRobot):
    def __init__(
        self,
        bddl_path_root: str,
        task_suite: str,
        task_idx: int,
        *,
        image_size: int = 224,
        seed: int = 0,
        libero_path: Optional[str] = None,
    ) -> None:
        self.bddl_path_root = Path(bddl_path_root)
        self.task_suite = task_suite
        self.task_idx = int(task_idx)
        self.image_size = int(image_size)
        self.seed = int(seed)
        self.libero_path = _ensure_libero_on_path(libero_path)
        self._env = None
        self._language: str = ""
        # LIBERO benchmark protocol: each task has a list of pre-defined initial
        # scene states (one per episode_idx). Standard eval calls
        # ``env.set_init_state(initial_states[episode_idx])`` immediately after
        # reset. Without this, sim resets to a randomized scene that the model
        # has never seen and rollouts fail systematically. See upstream
        # VLA-Adapter run_libero_eval.py:229-242 + 405-419.
        self._libero_init_states: Optional[Any] = None
        # Episode index to apply on next reset(); set via ``set_episode_idx``.
        self._next_episode_idx: Optional[int] = None

    def _resolve_bddl_file(self) -> Path:
        # LIBERO's bddl files live as
        # bddl_files/<suite>/<task_index>_*.bddl. We pick the file whose name
        # starts with the integer task_idx.
        suite_dir = self.bddl_path_root / self.task_suite
        if not suite_dir.is_dir():
            raise FileNotFoundError(f"BDDL suite dir not found: {suite_dir}")
        candidates = sorted(suite_dir.glob("*.bddl"))
        if not candidates:
            raise FileNotFoundError(f"no .bddl files under {suite_dir}")
        if self.task_idx < 0 or self.task_idx >= len(candidates):
            raise IndexError(
                f"task_idx {self.task_idx} out of range [0, {len(candidates)})"
            )
        return candidates[self.task_idx]

    def connect(self) -> None:
        try:
            from libero.libero.envs import OffScreenRenderEnv  # type: ignore
        except ImportError as e:
            raise ImportError(
                f"libero not importable from {self.libero_path}; "
                f"set LIBERO_PATH env var or pass libero_path= to LIBEROSimRobot"
            ) from e
        bddl_file = self._resolve_bddl_file()
        # Pull the task language from the BDDL file (heuristic: first line that
        # contains a quoted natural-language goal). Fallback to the file stem.
        try:
            text = bddl_file.read_text(errors="ignore")
            self._language = self._language_from_bddl(text) or bddl_file.stem
        except OSError:
            self._language = bddl_file.stem

        env_args = dict(
            bddl_file_name=str(bddl_file),
            camera_heights=self.image_size,
            camera_widths=self.image_size,
        )
        self._env = OffScreenRenderEnv(**env_args)
        self._env.seed(self.seed)
        # Load this task's standard LIBERO init_states. Match the upstream
        # eval protocol so each rollout uses an in-distribution scene.
        # libero's ``get_task_init_states`` calls ``torch.load`` without
        # ``weights_only=False``; on PyTorch ≥ 2.6 this fails with
        # "Unsupported global numpy.core.multiarray._reconstruct" because
        # the init_states file contains pickled numpy arrays. Bypass
        # libero's loader and read the file directly.
        try:
            import os as _os
            import torch as _torch
            from libero.libero import benchmark as _libero_benchmark  # type: ignore
            from libero.libero import get_libero_path as _get_libero_path  # type: ignore
            benchmark_dict = _libero_benchmark.get_benchmark_dict()
            task_suite_obj = benchmark_dict[self.task_suite]()
            task_obj = task_suite_obj.tasks[self.task_idx]
            init_states_path = _os.path.join(
                _get_libero_path("init_states"),
                task_obj.problem_folder,
                task_obj.init_states_file,
            )
            self._libero_init_states = _torch.load(
                init_states_path, weights_only=False
            )
        except Exception as e:
            # If init_states aren't available (older libero or path issue),
            # fall back to plain env.reset() — random scene.
            print(f"[LIBEROSimRobot] init_states load failed: {e!r}; using random reset")
            self._libero_init_states = None

    @staticmethod
    def _language_from_bddl(text: str) -> str:
        """Heuristic: pull a natural-language description from a BDDL :goal.

        BDDL files don't have a standard 'language' field, so we just return
        the file stem fallback if no obvious phrase is found. The model
        consumes the string verbatim, so any consistent encoding works for
        smoke purposes; real eval can override per-task.
        """
        for line in text.splitlines():
            s = line.strip()
            if s.startswith(";") and len(s) > 1:
                return s[1:].strip()
        return ""

    def _wrap_obs(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        # LIBERO's OffScreenRenderEnv returns a dict where:
        #   "agentview_image"      -> (H, W, 3) uint8 (scene)
        #   "robot0_eye_in_hand_image" -> (H, W, 3) uint8 (wrist)
        #   "robot0_eef_pos" / "robot0_eef_quat" / "robot0_gripper_qpos"
        # ProprioVec convention (must match LeRobot LIBERO dataset's
        # observation.state schema, names=['x','y','z','rx','ry','rz','rw',
        # 'gripper']): 3 (xyz) + 3 (Euler-style rotation) + 2 (gripper qpos)
        # = 8. The dataset stores rotation NOT as a quaternion despite the
        # 'rw' name — the values look like Euler-style {roll≈π, pitch≈0,
        # yaw≈small} for LIBERO's arm-pointing-down configuration, with
        # 'rw' being a 2nd gripper finger qpos slot.
        # Verified by inspecting lerobot/libero_spatial_image first samples
        # (rx≈3.14 for arm-down; values not normalizable as a unit quat).
        scene = np.asarray(raw["agentview_image"], dtype=np.uint8)
        wrist = np.asarray(raw["robot0_eye_in_hand_image"], dtype=np.uint8)
        if scene.shape != (self.image_size, self.image_size, 3):
            raise ValueError(f"scene shape {scene.shape} != expected")
        if wrist.shape != (self.image_size, self.image_size, 3):
            raise ValueError(f"wrist shape {wrist.shape} != expected")
        # LIBERO renders with both axes flipped relative to the LeRobot
        # dataset orientation that the model was trained on. Mirror the
        # 180-degree rotation used in vla-gemma-4/scripts/gemma4/eval_libero_gemma4.py
        # (lines 143-154) so closed-loop obs match the dataset distribution.
        scene = scene[::-1, ::-1, :]
        wrist = wrist[::-1, ::-1, :]
        # Convert sim's eef_quat (xyzw) to axis-angle (3 dims) to match the
        # LeRobot LIBERO dataset's observation.state format. The dataset's
        # schema labels 8 dims as ['x','y','z','rx','ry','rz','rw','gripper']
        # but inspection of values + cross-reference with upstream
        # VLA-Adapter (experiments/robot/libero/libero_utils.py:63-87 +
        # run_libero_eval.py:260) shows the 4th-7th components are actually
        # 3 axis-angle + 2 gripper qpos, NOT a quaternion + 1 gripper. Use
        # the upstream's quat2axisangle implementation verbatim.
        quat_xyzw = np.asarray(raw["robot0_eef_quat"], dtype=np.float32)
        # quat2axisangle (from robosuite via VLA-Adapter)
        if quat_xyzw[3] > 1.0:
            quat_xyzw[3] = 1.0
        elif quat_xyzw[3] < -1.0:
            quat_xyzw[3] = -1.0
        den = float(np.sqrt(1.0 - quat_xyzw[3] * quat_xyzw[3]))
        import math as _math
        if _math.isclose(den, 0.0):
            axis_angle = np.zeros(3, dtype=np.float32)
        else:
            axis_angle = (
                quat_xyzw[:3].astype(np.float32) * 2.0 * _math.acos(float(quat_xyzw[3])) / den
            ).astype(np.float32)
        gripper = np.asarray(raw["robot0_gripper_qpos"], dtype=np.float32)
        if gripper.shape[0] < 2:
            # Some LIBERO versions report only one finger. Pad with the
            # negation, matching the symmetric two-finger pattern in dataset.
            gripper = np.array([gripper[0], -gripper[0]], dtype=np.float32)
        else:
            gripper = gripper[:2]
        proprio = np.concatenate([
            np.asarray(raw["robot0_eef_pos"], dtype=np.float32),
            axis_angle,
            gripper,
        ]).astype(np.float32)
        return {
            "scene_image": np.ascontiguousarray(scene),
            "wrist_image": np.ascontiguousarray(wrist),
            "proprio": proprio,
            "language": self._language,
            "_raw": raw,  # kept for debugging / future success-detection
        }

    def set_episode_idx(self, episode_idx: int) -> None:
        """Mark the next ``reset()`` call to use the LIBERO benchmark's
        standard init_state for ``episode_idx`` (0-indexed). Required for
        in-distribution rollouts; without it, eval scenes are randomized and
        learned policies fail.
        """
        self._next_episode_idx = int(episode_idx)

    def reset(self) -> Dict[str, Any]:
        if self._env is None:
            raise RuntimeError("LIBEROSimRobot not connected; call connect() first")
        raw = self._env.reset()
        if self._libero_init_states is not None and self._next_episode_idx is not None:
            ep = self._next_episode_idx
            n = len(self._libero_init_states)
            if 0 <= ep < n:
                # set_init_state returns the obs dict after applying the state
                raw = self._env.set_init_state(self._libero_init_states[ep])
            else:
                print(f"[LIBEROSimRobot] episode_idx={ep} out of range [0, {n}); "
                      f"using plain reset")
        return self._wrap_obs(raw)

    def get_observation(self) -> Dict[str, Any]:
        if self._env is None:
            raise RuntimeError("LIBEROSimRobot not connected; call connect() first")
        # OffScreenRenderEnv exposes the cached observation via ._obs / ._get_observations.
        # We call the public-ish _get_observations() if available; otherwise re-read
        # from the env's observation buffer.
        if hasattr(self._env, "_get_observations"):
            raw = self._env._get_observations()
        elif hasattr(self._env, "env") and hasattr(self._env.env, "_get_observations"):
            raw = self._env.env._get_observations()
        else:
            raise RuntimeError("LIBERO env exposes no observation accessor")
        return self._wrap_obs(raw)

    def send_action(self, action: np.ndarray) -> Dict[str, Any]:
        if self._env is None:
            raise RuntimeError("LIBEROSimRobot not connected; call connect() first")
        a = np.asarray(action, dtype=np.float32).reshape(-1)
        if a.shape[0] != 7:
            raise ValueError(f"action length {a.shape[0]} != 7")
        raw, _reward, _done, _info = self._env.step(a.tolist())
        return self._wrap_obs(raw)

    def check_success(self) -> bool:
        """Ask the underlying LIBERO env whether the current sim state
        satisfies the task's BDDL goal. Returns False if not connected."""
        if self._env is None:
            return False
        env = self._env
        # OffScreenRenderEnv exposes `_check_success()` via robosuite's
        # ManipulationEnv; LIBERO subclasses override it with the BDDL
        # success predicate.
        for accessor in ("_check_success", "check_success"):
            fn = getattr(env, accessor, None)
            if callable(fn):
                try:
                    return bool(fn())
                except Exception:
                    return False
        return False

    def close(self) -> None:
        if self._env is not None:
            try:
                self._env.close()
            except Exception:
                pass
            self._env = None
