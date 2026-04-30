# LIBERO sim_robot Implementation Plan (Plan 7 / 10)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the `robots/` layer per CLAUDE.md. Provide an abstract `BaseRobot` interface and a concrete `LIBEROSimRobot` that wraps `libero.libero.envs.OffScreenRenderEnv` (MuJoCo + robosuite under the hood) and exposes the four-method `connect / reset / get_observation / send_action / close` contract. End-to-end smoke verifies one episode reset + 10 dummy steps on a real LIBERO-Spatial task.

**Architecture:** Three layers:

1. `robots/base_robot.py`: `BaseRobot` abstract base — the connect/reset/get_observation/send_action/close interface and obs/action dtype contracts.
2. `robots/sim_robot.py`: `LIBEROSimRobot(BaseRobot)`. Lazily imports `libero.libero.envs.OffScreenRenderEnv` so the import is only required at `connect()` time. Construction takes a BDDL task spec + optional `bddl_path_root`, an image-render size, and a `seed`. The wrapper translates LIBERO's raw obs dict into the project's observation shape (`scene_image`, `wrist_image`, `proprio`, `language`) so it slots straight into `XVLAAdapterPolicy.select_action`.
3. Real-sim smoke test that does `reset()` + 10 `send_action(zeros)` steps on `libero_spatial::pick_up_the_alphabet_soup`. The smoke is gated by an importable `libero` module — if the dep isn't installed, the test SKIPs. CI / portable runs stay green.

The wrapper does NOT wrap policy logic. `XVLAAdapterPolicy` and `LIBEROSimRobot` interact only through the `obs` schema; the rollout loop (Plan 8) will compose them.

**Tech Stack:** Add `mujoco`, `robosuite`, `bddl` as runtime deps. The LIBERO Python package itself (`libero`) is an editable install pointing at `/misc/dl00/takaki/vla-gemma-4/LIBERO` so we don't ship a copy of the BDDL files / asset trees in our repo.

**Repo references:**
- `/misc/dl00/takaki/vla-gemma-4/LIBERO/libero/libero/envs/__init__.py` — source of `OffScreenRenderEnv`. We don't import privately; we use the public class.
- `/misc/dl00/takaki/vla-gemma-4/scripts/gemma4/eval_libero_gemma4.py:97-133` — canonical reference for instantiating `OffScreenRenderEnv` with `env_args`. Mirror that constructor pattern verbatim.
- `/misc/dl00/takaki/vla-gemma-4/.venv-gemma4/` — existing working environment that holds robosuite 1.4.1 + mujoco 3.7.0 + bddl 3.6.0 + LIBERO 0.1.0. Confirms the deps coexist with newer torch/transformers.
- `~/.libero/config.yaml` — already exists on dl40 with paths into the LIBERO repo.

**Hard constraints from CLAUDE.md:**
- Boundary: `robots/` does not import `models/`, `policies/`, or `data/datasets/`. Robot wrappers know about hardware/sim only.
- The robot's `get_observation()` return value should be CLOSE to the dataset observation schema (uint8 HWC images, float32 proprio, str language) — it is what the runtime policy receives.
- Fail-fast on missing deps: `connect()` raises a clear `ImportError` (not a generic ModuleNotFoundError mid-step) if `libero` / `mujoco` / `robosuite` are not importable.

---

## File Structure

**Create:**
- `src/vla_project/robots/__init__.py`
- `src/vla_project/robots/base_robot.py`
- `src/vla_project/robots/sim_robot.py`
- `tests/test_base_robot.py`
- `tests/test_libero_sim_robot.py`

**Modify:**
- `pyproject.toml` (add `mujoco`, `robosuite`, `bddl`; document LIBERO_PATH env var)

**Do not modify:** `models/`, `policies/`, `training/`, `data/`, `scripts/train.py`. The LIBERO source tree at `/misc/dl00/takaki/vla-gemma-4/LIBERO` is read-only; no edits.

---

## Task 1: `BaseRobot` interface + interface contract test

**Files:**
- Create: `src/vla_project/robots/__init__.py` (empty)
- Create: `src/vla_project/robots/base_robot.py`
- Create: `tests/test_base_robot.py`

- [ ] **Step 1: Write failing test**

`tests/test_base_robot.py`:

```python
"""Interface contract tests for BaseRobot.

We don't sim anything — just verify the abstract base enforces the four-
method protocol and that a tiny subclass can be instantiated and called.
"""
import pytest
from typing import Any, Dict

import numpy as np

from vla_project.robots.base_robot import BaseRobot


class _ToyRobot(BaseRobot):
    def __init__(self) -> None:
        self.connected = False
        self.last_action: np.ndarray | None = None

    def connect(self) -> None:
        self.connected = True

    def reset(self) -> Dict[str, Any]:
        return {
            "scene_image": np.zeros((4, 4, 3), dtype=np.uint8),
            "wrist_image": np.zeros((4, 4, 3), dtype=np.uint8),
            "proprio":     np.zeros(8, dtype=np.float32),
            "language":    "noop",
        }

    def get_observation(self) -> Dict[str, Any]:
        return self.reset()

    def send_action(self, action: np.ndarray) -> Dict[str, Any]:
        self.last_action = np.asarray(action, dtype=np.float32)
        return self.get_observation()

    def close(self) -> None:
        self.connected = False


def test_base_robot_is_abstract() -> None:
    with pytest.raises(TypeError):
        BaseRobot()  # type: ignore[abstract]


def test_subclass_round_trip() -> None:
    r = _ToyRobot()
    assert not r.connected
    r.connect()
    assert r.connected
    obs = r.reset()
    assert obs["scene_image"].shape == (4, 4, 3)
    assert obs["proprio"].shape == (8,)
    assert obs["language"] == "noop"

    r.send_action(np.array([0.1] * 7, dtype=np.float32))
    assert r.last_action is not None
    assert r.last_action.shape == (7,)

    r.close()
    assert not r.connected
```

- [ ] **Step 2: Run, expect FAIL**

```bash
PYTHONPATH="" uv run pytest tests/test_base_robot.py -v
```

Expected: `ModuleNotFoundError` on `vla_project.robots.base_robot`.

- [ ] **Step 3: Implement**

`src/vla_project/robots/__init__.py` — empty file.

`src/vla_project/robots/base_robot.py`:

```python
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
```

- [ ] **Step 4: Run, expect PASS**

```bash
PYTHONPATH="" uv run pytest tests/test_base_robot.py -v
PYTHONPATH="" uv run pytest -q
```

Expected: 2 new tests pass; full suite green (91 + 2 = 93).

- [ ] **Step 5: Commit**

```bash
git add src/vla_project/robots/__init__.py \
        src/vla_project/robots/base_robot.py \
        tests/test_base_robot.py
git commit -m "feat(robots): BaseRobot abstract interface"
```

---

## Task 2: Add sim deps (`mujoco` + `robosuite` + `bddl`)

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add via uv, with overrides to keep transformers >= 5.0**

Run:

```bash
PYTHONPATH="" uv add mujoco robosuite bddl
```

If uv complains about a transitive constraint (LIBERO's old `requirements.txt` lists `transformers==4.21.1` / `numpy==1.22.4`, but neither robosuite nor bddl pin those — they should be safe), inspect the conflict and STOP for guidance. Specifically:
- robosuite 1.4.x doesn't pin transformers.
- mujoco 3.x doesn't pin transformers.
- bddl 3.x doesn't pin transformers.
- We do NOT install the `libero` package itself via this command (Task 3 imports it lazily from `LIBERO_PATH`).

If versions resolve to robosuite >= 1.4.1, mujoco >= 3.0, bddl >= 3.0 (matches the working `.venv-gemma4` setup), we're good.

- [ ] **Step 2: Verify imports**

```bash
PYTHONPATH="" uv run python -c "
import mujoco, robosuite, bddl
print('mujoco', mujoco.__version__)
print('robosuite', robosuite.__version__)
print('bddl', bddl.__version__)
"
```

Expected: three version lines, no traceback. Suppress robosuite's `[robosuite WARNING]` macro warning if it appears — it's not an error.

- [ ] **Step 3: Verify LIBERO importable via `PYTHONPATH`**

```bash
PYTHONPATH=/misc/dl00/takaki/vla-gemma-4/LIBERO uv run python -c "
import sys; sys.path.insert(0, '/misc/dl00/takaki/vla-gemma-4/LIBERO')
import libero.libero.envs as E
print('OffScreenRenderEnv ok:', hasattr(E, 'OffScreenRenderEnv'))
"
```

Expected: `OffScreenRenderEnv ok: True`. We do NOT install LIBERO as an editable package — we sys.path-inject at import time inside `LIBEROSimRobot.connect()` (Task 3).

- [ ] **Step 4: Full pytest still green**

```bash
PYTHONPATH="" uv run pytest -q
```

Expected: 93 passed (Task 1 baseline).

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "feat(deps): add mujoco + robosuite + bddl for LIBERO sim"
```

---

## Task 3: `LIBEROSimRobot` + integration smoke

**Files:**
- Create: `src/vla_project/robots/sim_robot.py`
- Create: `tests/test_libero_sim_robot.py`

- [ ] **Step 1: Write failing test**

`tests/test_libero_sim_robot.py`:

```python
"""Integration smoke for LIBEROSimRobot.

This test boots a real LIBERO env (MuJoCo off-screen render). It is gated
by ``importlib.util.find_spec("libero")`` AFTER injecting LIBERO_PATH so
the test SKIPs cleanly on machines without the LIBERO repo on disk.

Wallclock: ~10–30 s on dl40 (env init dominates).
"""
import importlib.util
import os
import sys

import numpy as np
import pytest

from vla_project.robots.sim_robot import LIBEROSimRobot

# Resolve LIBERO_PATH and probe for libero importability before parametrizing.
_LIBERO_PATH = os.environ.get(
    "LIBERO_PATH", "/misc/dl00/takaki/vla-gemma-4/LIBERO"
)
if _LIBERO_PATH not in sys.path:
    sys.path.insert(0, _LIBERO_PATH)
_LIBERO_AVAILABLE = importlib.util.find_spec("libero") is not None


pytestmark = pytest.mark.skipif(
    not _LIBERO_AVAILABLE,
    reason=f"libero not importable from {_LIBERO_PATH}",
)


def test_connect_reset_step_close() -> None:
    """Reset + 10 dummy steps must yield finite obs every step."""
    robot = LIBEROSimRobot(
        bddl_path_root="/misc/dl00/takaki/vla-gemma-4/LIBERO/libero/libero/bddl_files",
        task_suite="libero_spatial",
        task_idx=0,
        image_size=224,
        seed=0,
        libero_path=_LIBERO_PATH,
    )
    robot.connect()
    try:
        obs = robot.reset()
        assert obs["scene_image"].shape == (224, 224, 3)
        assert obs["scene_image"].dtype == np.uint8
        assert obs["wrist_image"].shape == (224, 224, 3)
        assert obs["proprio"].shape == (8,)
        assert obs["proprio"].dtype == np.float32
        assert isinstance(obs["language"], str)
        assert obs["language"]  # non-empty

        for _ in range(10):
            obs = robot.send_action(np.zeros(7, dtype=np.float32))
            assert np.isfinite(obs["proprio"]).all()
            assert obs["scene_image"].shape == (224, 224, 3)
    finally:
        robot.close()


def test_get_observation_does_not_step() -> None:
    """Two consecutive get_observation() calls return data from the same
    sim time step (proprio identical)."""
    robot = LIBEROSimRobot(
        bddl_path_root="/misc/dl00/takaki/vla-gemma-4/LIBERO/libero/libero/bddl_files",
        task_suite="libero_spatial",
        task_idx=0,
        image_size=224,
        seed=0,
        libero_path=_LIBERO_PATH,
    )
    robot.connect()
    try:
        robot.reset()
        a = robot.get_observation()
        b = robot.get_observation()
        np.testing.assert_array_equal(a["proprio"], b["proprio"])
    finally:
        robot.close()
```

- [ ] **Step 2: Run, expect FAIL**

```bash
PYTHONPATH="" uv run pytest tests/test_libero_sim_robot.py -v
```

Expected: `ModuleNotFoundError` on `vla_project.robots.sim_robot`.

- [ ] **Step 3: Implement**

`src/vla_project/robots/sim_robot.py`:

```python
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

    def close(self) -> None:
        if self._env is not None:
            try:
                self._env.close()
            except Exception:
                pass
            self._env = None
```

- [ ] **Step 4: Run the integration test**

```bash
PYTHONPATH="" uv run pytest tests/test_libero_sim_robot.py -v
```

If LIBERO can't be imported (e.g., LIBERO_PATH points to a stale dir, or mujoco failed to install), the test SKIPs with a clear message — that's acceptable. If LIBERO IS importable, both tests must pass.

- [ ] **Step 5: Full pytest**

```bash
PYTHONPATH="" uv run pytest -q
```

Expected: 95 passed (93 + 2 new sim tests) when LIBERO is importable; 93 passed + 2 skipped otherwise.

- [ ] **Step 6: Commit**

```bash
git add src/vla_project/robots/sim_robot.py tests/test_libero_sim_robot.py
git commit -m "feat(robots): LIBEROSimRobot wrapping OffScreenRenderEnv"
```

---

## Task 4: Push branch + open PR

- [ ] **Step 1: Push**

```bash
git status -sb
git log --oneline feat/policies-runtime..HEAD
git push -u origin feat/libero-sim-robot
```

- [ ] **Step 2: PR**

PR base: `feat/policies-runtime` (rebase to `main` after Plans 1-6 merge).
Title: `feat(robots): BaseRobot + LIBEROSimRobot wrapping OffScreenRenderEnv`.
Body should note:
- Test count delta (91 → 95 expected when LIBERO importable; otherwise 91 → 93 + 2 skipped).
- The sim deps add ~80 MB to `.venv` (mujoco binaries dominate).
- LIBERO_PATH env var is honored; defaults to `/misc/dl00/takaki/vla-gemma-4/LIBERO` for dl40.

---

## Done criteria

- [ ] `uv run pytest -q` green (95 or 93+2-skipped depending on LIBERO availability).
- [ ] `LIBEROSimRobot.connect / reset / send_action / close` round-trips on a real LIBERO-Spatial task without exceptions.
- [ ] `get_observation()` does not step the sim.
- [ ] `BaseRobot` is abstract; concrete subclass works without LIBERO deps.
- [ ] No edits to `models/`, `policies/`, `training/`, `data/`, `scripts/train.py`.
- [ ] `robots/sim_robot.py` does not import from `models/`, `policies/`, or `training/`.

## Out of scope (later plans)

- Closed-loop rollout that wires `XVLAAdapterPolicy` + `LIBEROSimRobot` (Plan 8).
- Success-detection from LIBERO obs (Plan 8 imports the LIBERO benchmark's `check_success` helpers).
- Real robot wrappers (`ros2_robot.py`, `lerobot_robot.py`).
- Vector-env / parallel rollout (`venv` from LIBERO).
- Per-task language overrides (BDDL files lack consistent natural-language goals; Plan 8 may inject them from the LeRobot dataset's `meta.tasks` instead).
