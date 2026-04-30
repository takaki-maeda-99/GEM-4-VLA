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
        # ProprioVec convention: 3 (xyz) + 4 (quat) + 1 (gripper) = 8.
        scene = np.asarray(raw["agentview_image"], dtype=np.uint8)
        wrist = np.asarray(raw["robot0_eye_in_hand_image"], dtype=np.uint8)
        if scene.shape != (self.image_size, self.image_size, 3):
            raise ValueError(f"scene shape {scene.shape} != expected")
        if wrist.shape != (self.image_size, self.image_size, 3):
            raise ValueError(f"wrist shape {wrist.shape} != expected")
        # LIBERO renders with the world Y axis flipped relative to
        # eval_libero_gemma4's reference; mirror that flip here so the
        # observation matches what the model was trained on.
        scene = scene[::-1, :, :]
        wrist = wrist[::-1, :, :]
        proprio = np.concatenate([
            np.asarray(raw["robot0_eef_pos"], dtype=np.float32),
            np.asarray(raw["robot0_eef_quat"], dtype=np.float32),
            np.asarray(raw["robot0_gripper_qpos"], dtype=np.float32)[:1],
        ]).astype(np.float32)
        return {
            "scene_image": np.ascontiguousarray(scene),
            "wrist_image": np.ascontiguousarray(wrist),
            "proprio": proprio,
            "language": self._language,
            "_raw": raw,  # kept for debugging / future success-detection
        }

    def reset(self) -> Dict[str, Any]:
        if self._env is None:
            raise RuntimeError("LIBEROSimRobot not connected; call connect() first")
        raw = self._env.reset()
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
