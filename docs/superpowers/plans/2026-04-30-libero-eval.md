# LIBERO Closed-Loop Evaluation Plan (Plan 8 / 10)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Glue Plans 6 (`XVLAAdapterPolicy`) and 7 (`LIBEROSimRobot`) into a closed-loop rollout: load the model, build the policy, instantiate a LIBERO env, run N episodes, aggregate per-task success rate, write metrics. End-to-end smoke runs one episode of `libero_spatial::task_0` with a randomly-initialized real Gemma4-E2B + randomly-initialized head and verifies the loop completes without exceptions (success rate=0 is expected; we are testing wiring, not learning).

**Architecture:** Three new files in `evaluation/`:

1. `rollout.py`: `run_episode(policy, robot, max_steps, num_steps_wait=10, success_check=None) -> EpisodeResult`. Walks the standard X-VLA / OpenVLA pattern — `num_steps_wait` warm-up zero actions then one `policy.select_action(obs)` per sim step. The chunk buffer is owned inside the policy (Plan 6); the rollout doesn't reach into it. Success comes from a caller-supplied `success_check(robot, info) -> bool` (defaults to checking `info["success"]` and `done` from the env step). Returns episode metadata: success bool, env-step count, model-call count (counted via a hook), wallclock.

2. `metrics.py`: `aggregate_episodes(episodes) -> dict` — turns a list of `EpisodeResult`s into per-task and per-suite success rate, mean steps, etc.

3. `libero_eval.py`: `evaluate_libero(policy, suite, task_idxs, num_episodes_per_task, max_steps, ...) -> dict` — top-level orchestrator. For each `task_idx`: build a `LIBEROSimRobot`, connect, run `num_episodes_per_task` episodes (calling `policy.reset()` before each), tear down. Returns aggregate metrics + per-episode raw records.

A new `LIBEROSimRobot.check_success()` method asks the underlying LIBERO env for its task success bit (LIBERO env's `_check_success()` is public-ish; we wrap it).

A thin `scripts/eval.py` (CLI) + a config block let the user run from the shell. The smoke uses a fresh model (no checkpoint), proving the loop works even when no real training has been done.

**Tech Stack:** `LIBEROSimRobot` (Plan 7), `XVLAAdapterPolicy` (Plan 6), existing torch / OmegaConf. No new deps.

**Repo references:**
- `/misc/dl00/takaki/vla-gemma-4/scripts/gemma4/eval_libero_gemma4.py:425-525` — reference rollout loop (action queue, num_steps_wait, env.step → done detection). Mirror the structure.
- `src/vla_project/policies/xvla_adapter_policy.py` — `select_action` already chunks; the rollout calls it once per env step.
- `src/vla_project/robots/sim_robot.py` — `connect / reset / send_action / close` plus the new `check_success()` method.
- `CLAUDE.md` "Evaluation" section — "Evaluation should save: metrics, rollout videos, failure cases, used config, checkpoint reference." We save metrics + config + checkpoint reference now; videos are out of scope (deferred).

**Hard constraints from CLAUDE.md:**
- Boundary: `evaluation/` may import `policies/` and `robots/` (it's the runtime layer that composes both). It must NOT import from `models/` directly.
- Each task's success bit comes from the env, not from a heuristic on action magnitudes.
- The smoke success expectation is "loop completes without exception" — not "high success rate".

---

## File Structure

**Create:**
- `src/vla_project/evaluation/__init__.py`
- `src/vla_project/evaluation/rollout.py`
- `src/vla_project/evaluation/metrics.py`
- `src/vla_project/evaluation/libero_eval.py`
- `tests/test_rollout.py`
- `tests/test_metrics.py`
- `tests/test_libero_eval.py` (uses stub policy + stub robot — no MuJoCo)
- `scripts/eval.py` (CLI thin wrapper)
- `configs/eval/libero_smoke.yaml`

**Modify:**
- `src/vla_project/robots/sim_robot.py` (add `check_success()` method)

**Do not modify:** `policies/`, `models/`, `training/`, `data/`.

---

## Task 1: `LIBEROSimRobot.check_success` extension

**Files:**
- Modify: `src/vla_project/robots/sim_robot.py`

- [ ] **Step 1: Read existing file**

```bash
sed -n '160,180p' src/vla_project/robots/sim_robot.py
```

This shows the bottom of the class. We append a `check_success()` method.

- [ ] **Step 2: Add method**

Insert before the `close()` method definition:

```python
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
```

- [ ] **Step 3: Quick sanity check**

```bash
PYTHONPATH=/misc/dl00/takaki/vla-gemma-4/LIBERO uv run python -c "
import sys; sys.path.insert(0, '/misc/dl00/takaki/vla-gemma-4/LIBERO')
from vla_project.robots.sim_robot import LIBEROSimRobot
import numpy as np
r = LIBEROSimRobot('/misc/dl00/takaki/vla-gemma-4/LIBERO/libero/libero/bddl_files', 'libero_spatial', 0, image_size=224, seed=0)
r.connect()
r.reset()
print('check_success after reset:', r.check_success())
r.close()
"
```

Expected: `check_success after reset: False` (the goal isn't satisfied at episode start).

- [ ] **Step 4: Pytest still green**

```bash
PYTHONPATH="" uv run pytest -q
```

Expected: 95 passed.

- [ ] **Step 5: Commit**

```bash
git add src/vla_project/robots/sim_robot.py
git commit -m "feat(robots): LIBEROSimRobot.check_success accessor"
```

---

## Task 2: `rollout.run_episode` + unit tests

**Files:**
- Create: `src/vla_project/evaluation/__init__.py` (empty)
- Create: `src/vla_project/evaluation/rollout.py`
- Create: `tests/test_rollout.py`

- [ ] **Step 1: Write failing tests**

`tests/test_rollout.py`:

```python
"""Stub-based tests for rollout.run_episode.

We use a fake policy + fake robot — no MuJoCo, no model.
"""
from typing import Any, Dict

import numpy as np
import pytest

from vla_project.evaluation.rollout import EpisodeResult, run_episode
from vla_project.policies.base_policy import BasePolicy
from vla_project.robots.base_robot import BaseRobot


class _FakePolicy(BasePolicy):
    def __init__(self) -> None:
        self.calls = 0
        self.reset_calls = 0

    def select_action(self, obs: Dict[str, Any]) -> np.ndarray:
        self.calls += 1
        return np.zeros(7, dtype=np.float32)

    def reset(self) -> None:
        self.reset_calls += 1


class _FakeRobot(BaseRobot):
    """Reaches success after `succeed_at` env steps."""

    def __init__(self, succeed_at: int = -1) -> None:
        self.succeed_at = succeed_at
        self.t = 0
        self.connected = False
        self.last_action: np.ndarray | None = None

    def _obs(self) -> Dict[str, Any]:
        return {
            "scene_image": np.zeros((4, 4, 3), dtype=np.uint8),
            "wrist_image": np.zeros((4, 4, 3), dtype=np.uint8),
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
        self.last_action = np.asarray(action, dtype=np.float32)
        return self._obs()

    def check_success(self) -> bool:
        return 0 <= self.succeed_at <= self.t

    def close(self) -> None:
        self.connected = False


def test_loop_runs_full_length_when_no_success() -> None:
    p = _FakePolicy()
    r = _FakeRobot(succeed_at=-1)
    r.connect()
    out = run_episode(p, r, max_steps=20, num_steps_wait=5)
    r.close()
    assert isinstance(out, EpisodeResult)
    assert out.success is False
    # 5 warmup zero-action steps + 20 policy-action steps = 25 sim steps.
    assert out.num_env_steps == 25
    # Policy is called once per non-warmup step.
    assert p.calls == 20
    # Policy.reset() called once at the start of the episode.
    assert p.reset_calls == 1


def test_loop_short_circuits_on_success() -> None:
    p = _FakePolicy()
    r = _FakeRobot(succeed_at=8)  # success after step 8 (counted from sim 0)
    r.connect()
    out = run_episode(p, r, max_steps=100, num_steps_wait=5)
    r.close()
    assert out.success is True
    # 5 warmup + N policy steps where N is the smallest integer such that
    # warmup + N >= 8. So N = 3 → total 8 sim steps.
    assert out.num_env_steps == 8


def test_warmup_steps_zero_action() -> None:
    p = _FakePolicy()
    r = _FakeRobot(succeed_at=-1)
    r.connect()
    run_episode(p, r, max_steps=2, num_steps_wait=3)
    r.close()
    # Last action stored — after warmup it would be a policy zero (also zeros),
    # but we can verify the warmup itself: zero-action.
    assert r.last_action is not None
    assert np.all(r.last_action == 0)


def test_zero_max_steps_is_warmup_only() -> None:
    p = _FakePolicy()
    r = _FakeRobot(succeed_at=-1)
    r.connect()
    out = run_episode(p, r, max_steps=0, num_steps_wait=3)
    r.close()
    assert out.num_env_steps == 3
    assert p.calls == 0
```

- [ ] **Step 2: Run, expect FAIL**

```bash
PYTHONPATH="" uv run pytest tests/test_rollout.py -v
```

Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

`src/vla_project/evaluation/rollout.py`:

```python
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
from typing import Any, Callable, Dict, Optional

import numpy as np

from vla_project.policies.base_policy import BasePolicy
from vla_project.robots.base_robot import BaseRobot


@dataclass
class EpisodeResult:
    success: bool
    num_env_steps: int
    num_policy_calls: int
    elapsed_s: float
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
) -> EpisodeResult:
    """Run one closed-loop episode.

    Args:
        policy: BasePolicy. Must implement ``select_action`` and ``reset``.
        robot: BaseRobot already ``connect()``-ed. Must implement
            ``reset / send_action / close``. ``check_success`` is consulted
            after each step unless ``success_check`` is supplied.
        max_steps: number of policy-driven env steps after warm-up.
        num_steps_wait: warm-up env steps with zero action.
        action_dim: action vector length used for the warm-up zero action.
        success_check: optional callable(robot, info) -> bool overriding the
            default ``robot.check_success()`` call.

    Returns:
        EpisodeResult with success bit, env-step count, policy-call count,
        wallclock seconds, and any extra ``info``.
    """
    t0 = time.perf_counter()
    success = False
    num_policy_calls = 0

    obs = robot.reset()
    policy.reset()

    # Warm-up: zero actions, env settles. Success can already fire here in
    # rare cases (BDDL spec evaluating True at t=0); we honor it.
    for _ in range(num_steps_wait):
        obs = robot.send_action(_zero_action(action_dim))
        if (success_check or (lambda r, _o: r.check_success()))(robot, obs):
            success = True
            break

    if not success:
        for _ in range(max_steps):
            action = policy.select_action(obs)
            num_policy_calls += 1
            obs = robot.send_action(action)
            if (success_check or (lambda r, _o: r.check_success()))(robot, obs):
                success = True
                break

    elapsed_s = time.perf_counter() - t0
    # The robot subclass may not expose a step counter directly; we
    # reconstruct it from the loop's bookkeeping.
    num_env_steps = num_policy_calls + (
        num_steps_wait
        if success and num_policy_calls == 0
        # If success fired during warm-up, num_env_steps is the iteration
        # count up to (and including) that step. We approximate as min
        # (num_steps_wait, last warmup index + 1); easier: just count what
        # we sent. Use the simpler total = warmup_executed + policy_calls.
        else num_steps_wait
    ) if not success else (
        # When success fires, we want the index of the firing step.
        # Recompute below.
        0
    )
    # Cleaner reformulation: total sent actions == warmup_done + policy_calls.
    # Track those explicitly:
    return EpisodeResult(
        success=success,
        num_env_steps=_count_env_steps(robot),
        num_policy_calls=num_policy_calls,
        elapsed_s=elapsed_s,
        info={},
    )


def _count_env_steps(robot: BaseRobot) -> int:
    """Best-effort env-step count via attributes commonly carried by sim
    wrappers (``robot.t`` for our LIBEROSimRobot / fake robot)."""
    return int(getattr(robot, "t", 0))
```

Wait — I made the bookkeeping in `run_episode` overly tangled. Replace the body with this cleaner version (delete the messy `num_env_steps = ...` math at the bottom):

```python
def run_episode(
    policy: BasePolicy,
    robot: BaseRobot,
    *,
    max_steps: int,
    num_steps_wait: int = 10,
    action_dim: int = 7,
    success_check: Optional[Callable[[BaseRobot, Dict[str, Any]], bool]] = None,
) -> EpisodeResult:
    t0 = time.perf_counter()
    check = success_check or (lambda r, _o: r.check_success())
    success = False
    num_policy_calls = 0
    num_env_steps = 0

    obs = robot.reset()
    policy.reset()

    for _ in range(num_steps_wait):
        obs = robot.send_action(_zero_action(action_dim))
        num_env_steps += 1
        if check(robot, obs):
            success = True
            break

    if not success:
        for _ in range(max_steps):
            action = policy.select_action(obs)
            num_policy_calls += 1
            obs = robot.send_action(action)
            num_env_steps += 1
            if check(robot, obs):
                success = True
                break

    return EpisodeResult(
        success=success,
        num_env_steps=num_env_steps,
        num_policy_calls=num_policy_calls,
        elapsed_s=time.perf_counter() - t0,
        info={},
    )
```

(The `_count_env_steps` helper goes away; delete it. The cleaner version above is what should land.)

`src/vla_project/evaluation/__init__.py` — empty.

- [ ] **Step 4: Run, expect PASS**

```bash
PYTHONPATH="" uv run pytest tests/test_rollout.py -v
PYTHONPATH="" uv run pytest -q
```

Expected: 4 new tests pass; full suite green (95 + 4 = 99).

- [ ] **Step 5: Commit**

```bash
git add src/vla_project/evaluation/__init__.py \
        src/vla_project/evaluation/rollout.py \
        tests/test_rollout.py
git commit -m "feat(evaluation): run_episode with chunk-aware closed-loop"
```

---

## Task 3: `metrics.aggregate_episodes` + tests

**Files:**
- Create: `src/vla_project/evaluation/metrics.py`
- Create: `tests/test_metrics.py`

- [ ] **Step 1: Write failing tests**

`tests/test_metrics.py`:

```python
"""Aggregation tests for evaluation.metrics."""
from vla_project.evaluation.metrics import aggregate_episodes
from vla_project.evaluation.rollout import EpisodeResult


def _ep(task: str, success: bool, steps: int = 100) -> dict:
    return {
        "task": task,
        "result": EpisodeResult(
            success=success,
            num_env_steps=steps,
            num_policy_calls=steps // 2,
            elapsed_s=1.0,
        ),
    }


def test_aggregates_per_task_and_overall() -> None:
    eps = [
        _ep("t0", True, 80),
        _ep("t0", False, 200),
        _ep("t1", True, 50),
        _ep("t1", True, 60),
    ]
    out = aggregate_episodes(eps)
    assert out["overall"]["num_episodes"] == 4
    assert out["overall"]["num_success"] == 3
    assert out["overall"]["success_rate"] == 0.75
    assert out["per_task"]["t0"]["num_episodes"] == 2
    assert out["per_task"]["t0"]["success_rate"] == 0.5
    assert out["per_task"]["t1"]["success_rate"] == 1.0
    # mean_env_steps per task
    assert out["per_task"]["t0"]["mean_env_steps"] == 140.0
    assert out["per_task"]["t1"]["mean_env_steps"] == 55.0


def test_empty_list_returns_zeroed_overall() -> None:
    out = aggregate_episodes([])
    assert out["overall"] == {
        "num_episodes": 0,
        "num_success": 0,
        "success_rate": 0.0,
        "mean_env_steps": 0.0,
    }
    assert out["per_task"] == {}
```

- [ ] **Step 2: Run, expect FAIL**

```bash
PYTHONPATH="" uv run pytest tests/test_metrics.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement**

`src/vla_project/evaluation/metrics.py`:

```python
"""Aggregate per-episode results into per-task / overall summary dicts."""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Iterable, List

from vla_project.evaluation.rollout import EpisodeResult


def _summary(results: List[EpisodeResult]) -> Dict[str, Any]:
    n = len(results)
    if n == 0:
        return {"num_episodes": 0, "num_success": 0, "success_rate": 0.0, "mean_env_steps": 0.0}
    n_success = sum(1 for r in results if r.success)
    mean_steps = sum(r.num_env_steps for r in results) / n
    return {
        "num_episodes": n,
        "num_success": n_success,
        "success_rate": n_success / n,
        "mean_env_steps": float(mean_steps),
    }


def aggregate_episodes(records: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    """Group a flat list of ``{"task": str, "result": EpisodeResult}`` records
    into per-task summaries plus an overall summary.

    Returns a dict shaped:

      {
        "overall":  {num_episodes, num_success, success_rate, mean_env_steps},
        "per_task": {task_id: {<same fields>}, ...},
      }
    """
    records = list(records)
    overall = _summary([r["result"] for r in records])
    per_task: Dict[str, List[EpisodeResult]] = defaultdict(list)
    for r in records:
        per_task[r["task"]].append(r["result"])
    return {
        "overall": overall,
        "per_task": {task: _summary(rs) for task, rs in per_task.items()},
    }
```

- [ ] **Step 4: Run, expect PASS**

```bash
PYTHONPATH="" uv run pytest tests/test_metrics.py -v
PYTHONPATH="" uv run pytest -q
```

Expected: 2 new tests pass; full suite green (99 + 2 = 101).

- [ ] **Step 5: Commit**

```bash
git add src/vla_project/evaluation/metrics.py tests/test_metrics.py
git commit -m "feat(evaluation): aggregate_episodes per-task / overall summary"
```

---

## Task 4: `evaluate_libero` orchestrator + stub-based test

**Files:**
- Create: `src/vla_project/evaluation/libero_eval.py`
- Create: `tests/test_libero_eval.py`

- [ ] **Step 1: Write failing test**

`tests/test_libero_eval.py`:

```python
"""Stub-based test for evaluate_libero (no MuJoCo)."""
from typing import Any, Dict

import numpy as np

from vla_project.evaluation.libero_eval import evaluate_libero
from vla_project.policies.base_policy import BasePolicy
from vla_project.robots.base_robot import BaseRobot


class _FakePolicy(BasePolicy):
    def select_action(self, obs: Dict[str, Any]) -> np.ndarray:
        return np.zeros(7, dtype=np.float32)

    def reset(self) -> None:
        pass


class _FakeRobot(BaseRobot):
    """Reports success after step 5 deterministically."""

    def __init__(self, *_a, **_kw) -> None:
        self.t = 0
        self.connected = False

    def _obs(self):
        return {
            "scene_image": np.zeros((4, 4, 3), dtype=np.uint8),
            "wrist_image": np.zeros((4, 4, 3), dtype=np.uint8),
            "proprio":     np.zeros(8, dtype=np.float32),
            "language":    "fake",
        }

    def connect(self) -> None:
        self.connected = True

    def reset(self):
        self.t = 0
        return self._obs()

    def get_observation(self):
        return self._obs()

    def send_action(self, action):
        self.t += 1
        return self._obs()

    def check_success(self) -> bool:
        return self.t >= 5

    def close(self) -> None:
        self.connected = False


def test_evaluate_libero_runs_with_stub_robot_factory() -> None:
    p = _FakePolicy()

    def robot_factory(task_idx: int) -> BaseRobot:
        return _FakeRobot()

    out = evaluate_libero(
        policy=p,
        robot_factory=robot_factory,
        task_idxs=[0, 1, 2],
        num_episodes_per_task=2,
        max_steps=20,
        num_steps_wait=2,
        task_label_fn=lambda i: f"task_{i}",
    )
    assert out["overall"]["num_episodes"] == 6
    assert out["overall"]["num_success"] == 6  # every fake episode succeeds
    assert set(out["per_task"].keys()) == {"task_0", "task_1", "task_2"}
    for k in out["per_task"]:
        assert out["per_task"][k]["num_episodes"] == 2
        assert out["per_task"][k]["success_rate"] == 1.0
```

- [ ] **Step 2: Run, expect FAIL**

```bash
PYTHONPATH="" uv run pytest tests/test_libero_eval.py -v
```

Expected: ModuleNotFoundError.

- [ ] **Step 3: Implement**

`src/vla_project/evaluation/libero_eval.py`:

```python
"""Top-level closed-loop evaluation orchestrator for LIBERO.

For each ``task_idx`` in the supplied list, a freshly-built robot runs
``num_episodes_per_task`` episodes through the rollout. Aggregated metrics
are returned. The robot is constructed via a caller-supplied factory so
tests can inject stubs without importing LIBERO / MuJoCo.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List

from vla_project.evaluation.metrics import aggregate_episodes
from vla_project.evaluation.rollout import run_episode
from vla_project.policies.base_policy import BasePolicy
from vla_project.robots.base_robot import BaseRobot


def evaluate_libero(
    policy: BasePolicy,
    robot_factory: Callable[[int], BaseRobot],
    task_idxs: List[int],
    *,
    num_episodes_per_task: int,
    max_steps: int,
    num_steps_wait: int = 10,
    task_label_fn: Callable[[int], str] = (lambda i: f"task_{i}"),
) -> Dict[str, Any]:
    records: List[Dict[str, Any]] = []
    for ti in task_idxs:
        robot = robot_factory(ti)
        robot.connect()
        try:
            for _ep in range(num_episodes_per_task):
                result = run_episode(
                    policy=policy,
                    robot=robot,
                    max_steps=max_steps,
                    num_steps_wait=num_steps_wait,
                )
                records.append({"task": task_label_fn(ti), "result": result})
        finally:
            robot.close()
    out = aggregate_episodes(records)
    out["records"] = records
    return out
```

- [ ] **Step 4: Run, expect PASS**

```bash
PYTHONPATH="" uv run pytest tests/test_libero_eval.py -v
PYTHONPATH="" uv run pytest -q
```

Expected: 1 new test passes; full suite green (101 + 1 = 102).

- [ ] **Step 5: Commit**

```bash
git add src/vla_project/evaluation/libero_eval.py tests/test_libero_eval.py
git commit -m "feat(evaluation): evaluate_libero orchestrator over task list"
```

---

## Task 5: `scripts/eval.py` CLI + real-sim smoke

**Files:**
- Create: `scripts/eval.py`
- Create: `configs/eval/libero_smoke.yaml`

- [ ] **Step 1: Write the smoke config**

`configs/eval/libero_smoke.yaml`:

```yaml
# 1 task × 1 episode end-to-end smoke. Does NOT load a checkpoint — uses a
# freshly-initialized model; the goal is to verify the loop, not learning.
seed: 0
model:
  num_domains: 1
  hidden_dim: 1536
  num_blocks: 35
  use_grad_checkpoint: false
vision:
  model_name: google/siglip-so400m-patch14-224
language:
  model_name: google/gemma-4-E2B
data:
  unnorm_key: libero_spatial_no_noops
  stats_path: ${oc.env:LIBERO_STATS_PATH,data/norm_stats/libero_spatial.json}
robot:
  bddl_path_root: /misc/dl00/takaki/vla-gemma-4/LIBERO/libero/libero/bddl_files
  task_suite: libero_spatial
  image_size: 224
  libero_path: ${oc.env:LIBERO_PATH,/misc/dl00/takaki/vla-gemma-4/LIBERO}
eval:
  task_idxs: [0]
  num_episodes_per_task: 1
  max_steps: 30
  num_steps_wait: 5
```

- [ ] **Step 2: Write `scripts/eval.py`**

`scripts/eval.py`:

```python
"""Closed-loop LIBERO evaluation entrypoint.

Loads a fresh (or checkpointed) VLAPolicy, builds an XVLAAdapterPolicy,
runs evaluate_libero over the configured task list, and prints aggregated
metrics. Checkpoint loading is optional via cfg.checkpoint.path.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

from vla_project.data.normalization import load_q99_stats
from vla_project.data.transforms.image import SiglipImageTransform
from vla_project.data.transforms.language import GemmaPromptTokenizer
from vla_project.evaluation.libero_eval import evaluate_libero
from vla_project.models.language.gemma4_wrapper import Gemma4Wrapper
from vla_project.models.vision.siglip import SigLIPEncoder
from vla_project.models.vla_policy import VLAPolicy, VLAPolicyConfig
from vla_project.policies.xvla_adapter_policy import XVLAAdapterPolicy
from vla_project.robots.sim_robot import LIBEROSimRobot
from vla_project.training.checkpoint import load_checkpoint
from vla_project.utils.seed import set_seed


def main(cfg_path: str) -> None:
    cfg = OmegaConf.load(cfg_path)
    set_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    print(f"[eval] device={device} dtype={dtype}")

    model_dict = OmegaConf.to_container(cfg.model, resolve=True)
    lora_cfg = model_dict.pop("lora", None)
    policy_cfg = VLAPolicyConfig(**model_dict)

    vision = SigLIPEncoder(model_name=cfg.vision.model_name)
    gemma = Gemma4Wrapper(
        model_name=cfg.language.model_name, freeze=True, lora=lora_cfg
    )
    model = VLAPolicy(policy_cfg, vision, gemma).to(device).to(dtype)
    model.eval()

    if cfg.get("checkpoint", {}).get("path"):
        meta = load_checkpoint(cfg.checkpoint.path, model)
        print(f"[eval] loaded checkpoint step={meta.get('step')!r}")

    stats = load_q99_stats(cfg.data.stats_path, cfg.data.unnorm_key)
    tok = GemmaPromptTokenizer(
        model_name=cfg.language.model_name, max_len=policy_cfg.prompt_max_len
    )
    image_tx = SiglipImageTransform(size=policy_cfg.prompt_max_len, training=False) \
        if False else SiglipImageTransform(size=224, training=False)

    policy = XVLAAdapterPolicy(
        model=model, tokenizer=tok, image_transform=image_tx,
        norm_stats=stats, action_chunk_len=policy_cfg.action_chunk_len,
        domain_id=0,
    )

    def _make_robot(task_idx: int) -> LIBEROSimRobot:
        return LIBEROSimRobot(
            bddl_path_root=cfg.robot.bddl_path_root,
            task_suite=cfg.robot.task_suite,
            task_idx=task_idx,
            image_size=cfg.robot.image_size,
            seed=cfg.seed,
            libero_path=cfg.robot.libero_path,
        )

    metrics = evaluate_libero(
        policy=policy,
        robot_factory=_make_robot,
        task_idxs=list(cfg.eval.task_idxs),
        num_episodes_per_task=int(cfg.eval.num_episodes_per_task),
        max_steps=int(cfg.eval.max_steps),
        num_steps_wait=int(cfg.eval.num_steps_wait),
    )
    summary = {"overall": metrics["overall"], "per_task": metrics["per_task"]}
    print(f"[eval] metrics={json.dumps(summary, indent=2)}")


if __name__ == "__main__":
    main(sys.argv[1])
```

(Note: the `image_tx = SiglipImageTransform(size=224, training=False)` line ignores `policy_cfg.prompt_max_len` — the size hardcoded is the SigLIP image size (224), which is unrelated to prompt length. Drop the `if False else ...` cruft when typing this and just write a single line.)

- [ ] **Step 3: Run the smoke**

```bash
cd /misc/dl00/takaki/X-VLA-Adapter
CUDA_VISIBLE_DEVICES=3 PYTHONPATH=/misc/dl00/takaki/vla-gemma-4/LIBERO PYTHONPATH= uv run python scripts/eval.py configs/eval/libero_smoke.yaml 2>&1 | tee /tmp/eval_smoke.log | tail -20
```

Wait — that PYTHONPATH chain is wrong. Use:

```bash
CUDA_VISIBLE_DEVICES=3 PYTHONPATH="" LIBERO_PATH=/misc/dl00/takaki/vla-gemma-4/LIBERO \
    uv run python scripts/eval.py configs/eval/libero_smoke.yaml 2>&1 | tee /tmp/eval_smoke.log | tail -20
```

Expected output: `[eval] device=cuda dtype=...`, then progress through env init (~30 s), then a JSON `metrics=` block. The success rate will be 0/1 (model is randomly initialized — actions are noise; success is impossible in 30 steps).

If the loop completes with a JSON metrics print, the wiring is correct. If anything throws (NaN, missing key, sim hang) STOP and report.

- [ ] **Step 4: Commit**

```bash
git add scripts/eval.py configs/eval/libero_smoke.yaml
git commit -m "feat(scripts): eval.py CLI + libero_smoke config"
```

---

## Task 6: Push branch + open PR

- [ ] **Step 1: Push**

```bash
git status -sb
git log --oneline feat/libero-sim-robot..HEAD
git push -u origin feat/libero-eval
```

- [ ] **Step 2: PR**

PR base: `feat/libero-sim-robot` (rebase to `main` after Plans 1-7 merge).
Title: `feat(evaluation): closed-loop LIBERO rollout + metrics + CLI`.
Body should include:
- Test count delta (95 → 95+7 = 102 expected).
- The smoke metrics JSON (`overall.success_rate=0.0` is expected with random init).
- Note: video saving + per-task language overrides are deferred.

---

## Done criteria

- [ ] `uv run pytest -q` green (102 tests expected).
- [ ] `python scripts/eval.py configs/eval/libero_smoke.yaml` runs to completion and prints metrics JSON.
- [ ] `LIBEROSimRobot.check_success()` is callable.
- [ ] No edits to `models/`, `policies/`, `training/`, `data/`.
- [ ] `evaluation/` does not import from `models/` directly.

## Out of scope (later plans / follow-ups)

- Video / GIF saving from rollout frames (would need imageio).
- Per-task BDDL → natural-language override map (current heuristic is filename stem; closed-loop quality may need real prompts).
- Image-flip reconciliation (Plan 7's `[::-1, :, :]` vs reference's `[::-1, ::-1]`) — affects rollout success rate but not loop wiring; revisit if needed.
- Loading a real trained checkpoint (we don't have one yet — Plan 4's checkpoint format is wired but no full training has run).
- Vector-env / parallel rollouts via robosuite SubprocVectorEnv.
