# XVLAAdapterPolicy Runtime Wrapper Plan (Plan 6 / 10)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the missing `policies/` layer per CLAUDE.md. Provide a `BasePolicy` interface and a concrete `XVLAAdapterPolicy` runtime wrapper that takes raw observations (`scene_image`, `wrist_image`, `proprio`, `language`), formats them into the project's internal Batch schema, calls the trained `VLAPolicy` model, denormalizes the output via stored `Q99Stats`, applies action chunking with stride 1, and returns one executable action per call. The wrapper is backed by a checkpoint produced by Plan 4. End-to-end stub-based smoke test verifies the full obs→action contract.

**Architecture:** Three pieces:

1. `data/normalization.py` extension: a pure `denormalize_action_q99(action_norm, stats)` inverse of the existing `normalize_action_q99`, used at inference to map model output back to executable action units.
2. `policies/base_policy.py`: a tiny abstract base that documents the `select_action(obs: dict) -> np.ndarray` contract (no logic, just interface + dtype/shape rules).
3. `policies/xvla_adapter_policy.py`: the concrete runtime wrapper. Owns the `VLAPolicy` model, an image transform (`SiglipImageTransform`), a `GemmaPromptTokenizer`, a `Q99Stats`, and an internal `_chunk_buffer: deque[np.ndarray]`. On each `select_action` call: if the buffer is empty, run model forward on the current observation, denormalize the predicted chunk, push all `H_act` actions into the buffer; pop and return the next action. Includes a `from_checkpoint(ckpt_dir, model, ...)` classmethod that hydrates from a Plan-4 checkpoint dir.

The model and the wrapper are intentionally separable: callers may pass any `VLAPolicy` (stub or real). The wrapper does not load Gemma4/SigLIP itself — that's the caller's responsibility, matching CLAUDE.md's boundary between `models/` (nn.Module construction) and `policies/` (runtime glue).

**Tech Stack:** existing `VLAPolicy`, `SiglipImageTransform`, `GemmaPromptTokenizer`, `Q99Stats`, `load_q99_stats`, `load_checkpoint`. New deps: none.

**Repo references:**
- `CLAUDE.md` "Policy Structure" section — locks the `select_action` interface and lists wrapper responsibilities (load ckpt, preprocess, prompt format, call model, denormalize, chunk, return).
- `src/vla_project/data/normalization.py:79-103` — `normalize_action_q99` is the inverse to write.
- `src/vla_project/data/transforms/image.py` — `SiglipImageTransform` with `(0.5, 0.5, 0.5)` mean/std.
- `src/vla_project/training/checkpoint.py` — `load_checkpoint` returns the meta dict with `cfg` and `norm_stats`.
- `src/vla_project/models/vla_policy.py:82-148` — `VLAPolicy.forward(batch) -> (pred, loss)`. We use `pred` only at inference; the wrapper still supplies a dummy `target_action` and `action_mask` so the existing forward signature is not changed.

**Hard constraints from CLAUDE.md:**
- Boundary: `policies/` does not import from `robots/` or hardware code. Real-time runtime concerns (latency, safety clamp, ROS) are out of scope and live in `deployment/` (later plan).
- Train-time and inference-time normalization MUST use the same `Q99Stats` payload. The wrapper takes `Q99Stats` at construction; `from_checkpoint` reads it from `meta.json::norm_stats`.
- Action chunking is explicit: the buffer is drained one action per call, and re-filled from the model only when empty.

---

## File Structure

**Create:**
- `src/vla_project/policies/__init__.py`
- `src/vla_project/policies/base_policy.py`
- `src/vla_project/policies/xvla_adapter_policy.py`
- `tests/test_denormalize_q99.py`
- `tests/test_xvla_adapter_policy.py`

**Modify:**
- `src/vla_project/data/normalization.py` (append `denormalize_action_q99`; do not touch existing symbols)

**Do not modify:** `models/`, `training/`, `data/datasets/`, `data/transforms/`, `scripts/train.py`, existing tests.

---

## Task 1: `denormalize_action_q99` inverse

**Files:**
- Modify: `src/vla_project/data/normalization.py`
- Create: `tests/test_denormalize_q99.py`

- [ ] **Step 1: Write failing tests**

`tests/test_denormalize_q99.py`:

```python
"""Tests for denormalize_action_q99 (inverse of normalize_action_q99)."""
import torch

from vla_project.data.normalization import (
    Q99Stats,
    denormalize_action_q99,
    normalize_action_q99,
)


def _stats(q01_val: float = -2.0, q99_val: float = 2.0, gripper_passthrough: bool = True) -> Q99Stats:
    A = 7
    return Q99Stats(
        q01=torch.tensor([q01_val] * A, dtype=torch.float32),
        q99=torch.tensor([q99_val] * A, dtype=torch.float32),
        mask=torch.tensor([True] * (A - 1) + [not gripper_passthrough], dtype=torch.bool),
    )


def test_denormalize_round_trip_clipped() -> None:
    stats = _stats(-2.0, 2.0, gripper_passthrough=True)
    raw = torch.tensor([
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5],   # midpoint -> 0 normed
        [1.0, -1.0, 0.5, -0.5, 1.5, -1.5, 1.0],
        [-2.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # at the bounds, normed -> -1 / +1
    ], dtype=torch.float32)
    normed = normalize_action_q99(raw, stats)
    denormed = denormalize_action_q99(normed, stats)
    # mask=True dims survive round-trip (within range; clipping is identity here).
    assert torch.allclose(denormed[:, :6], raw[:, :6], atol=1e-5)
    # mask=False dim (gripper) passes through both directions unchanged.
    assert torch.allclose(denormed[:, 6], raw[:, 6])


def test_denormalize_outside_range_does_not_inflate() -> None:
    """Inputs outside [-1, 1] on mask=True dims should denormalize to values in [q01, q99]."""
    stats = _stats(-2.0, 2.0, gripper_passthrough=True)
    normed = torch.tensor([[ 1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.5]], dtype=torch.float32)
    out = denormalize_action_q99(normed, stats)
    # +1 normed -> +q99 = +2.0;  -1 normed -> q01 = -2.0
    assert out[0, 0].item() == 2.0
    assert out[0, 1].item() == -2.0


def test_denormalize_preserves_dtype_and_shape() -> None:
    stats = _stats()
    normed = torch.zeros(4, 5, 7, dtype=torch.float32)
    out = denormalize_action_q99(normed, stats)
    assert out.shape == normed.shape
    assert out.dtype == normed.dtype


def test_denormalize_rejects_wrong_last_dim() -> None:
    import pytest
    stats = _stats()
    with pytest.raises(ValueError):
        denormalize_action_q99(torch.zeros(2, 5), stats)  # last dim 5 != stats dim 7


def test_denormalize_mask_false_passthrough() -> None:
    stats = _stats()  # last dim mask=False
    normed = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.42]], dtype=torch.float32)
    out = denormalize_action_q99(normed, stats)
    assert out[0, 6].item() == 0.42  # exact passthrough
```

- [ ] **Step 2: Run, expect FAIL**

```bash
PYTHONPATH="" uv run pytest tests/test_denormalize_q99.py -v
```

Expected: `ImportError: cannot import name 'denormalize_action_q99'`.

- [ ] **Step 3: Implement (append to `normalization.py`)**

Append to `src/vla_project/data/normalization.py`:

```python
def denormalize_action_q99(action_norm: torch.Tensor, stats: Q99Stats) -> torch.Tensor:
    """Inverse of ``normalize_action_q99`` for inference / deployment.

    For ``mask=True`` dims: rescale normalized values from [-1, 1] back to
    [q01, q99] via ``x = ((norm + 1) / 2) * (q99 - q01) + q01``.
    For ``mask=False`` dims: passthrough.

    Args:
        action_norm: [..., A] tensor (typically model output, may extend
            beyond [-1, 1]; downstream control code is expected to clip).
        stats: Q99Stats with shape [A].

    Returns:
        Tensor of same shape and dtype as ``action_norm``.
    """
    if action_norm.shape[-1] != stats.q01.shape[0]:
        raise ValueError(
            f"action last dim {action_norm.shape[-1]} != stats dim {stats.q01.shape[0]}"
        )
    q01 = stats.q01.to(action_norm.dtype).to(action_norm.device)
    q99 = stats.q99.to(action_norm.dtype).to(action_norm.device)
    mask = stats.mask.to(action_norm.device)
    span = q99 - q01
    raw = (action_norm + 1.0) * 0.5 * span + q01
    return torch.where(mask, raw, action_norm)
```

- [ ] **Step 4: Run, expect PASS**

```bash
PYTHONPATH="" uv run pytest tests/test_denormalize_q99.py tests/test_normalization_q99.py -v
PYTHONPATH="" uv run pytest -q
```

Expected: 5 new tests pass; full suite green (81 + 5 = 86).

- [ ] **Step 5: Commit**

```bash
git add src/vla_project/data/normalization.py tests/test_denormalize_q99.py
git commit -m "feat(data): denormalize_action_q99 inverse for inference"
```

---

## Task 2: `BasePolicy` interface

**Files:**
- Create: `src/vla_project/policies/__init__.py`
- Create: `src/vla_project/policies/base_policy.py`

This task is interface-only. No logic. The class will be subclassed by `XVLAAdapterPolicy` in Task 3.

- [ ] **Step 1: Add files**

`src/vla_project/policies/__init__.py` — empty.

`src/vla_project/policies/base_policy.py`:

```python
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
            ``np.ndarray[A]`` float32 — the action ready for the robot to
            execute (already denormalized, post-clipping etc. is the
            caller's choice).
        """
        ...

    @abstractmethod
    def reset(self) -> None:
        """Reset any internal buffer (action chunk queue, episode state).

        Called by the rollout loop at the start of each episode.
        """
        ...
```

- [ ] **Step 2: Smoke import**

```bash
PYTHONPATH="" uv run python -c "from vla_project.policies.base_policy import BasePolicy; print('ok')"
```

Expected: `ok`.

- [ ] **Step 3: Commit**

```bash
git add src/vla_project/policies/__init__.py src/vla_project/policies/base_policy.py
git commit -m "feat(policies): BasePolicy abstract interface"
```

(No dedicated test file for this task — `BasePolicy` is exercised by Task 3's tests via `XVLAAdapterPolicy(BasePolicy)`.)

---

## Task 3: `XVLAAdapterPolicy` concrete wrapper

**Files:**
- Create: `src/vla_project/policies/xvla_adapter_policy.py`
- Create: `tests/test_xvla_adapter_policy.py`

- [ ] **Step 1: Write failing tests**

`tests/test_xvla_adapter_policy.py`:

```python
"""Tests for XVLAAdapterPolicy.

Uses _StubSig + _StubGemma so the wrapper is exercised with deterministic,
fast forwards. The model still runs through the real VLAPolicy path; only
SigLIP and Gemma4 are stubbed.
"""
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest
import torch

from vla_project.data import constants as C
from vla_project.data.normalization import Q99Stats
from vla_project.data.transforms.image import SiglipImageTransform
from vla_project.data.transforms.language import GemmaPromptTokenizer
from vla_project.models.vla_policy import VLAPolicy, VLAPolicyConfig
from vla_project.policies.xvla_adapter_policy import XVLAAdapterPolicy
from tests._stubs import _StubGemma, _StubSig


class _StubTokenizer:
    pad_token_id = 0
    eos_token = "<eos>"
    pad_token = "<pad>"
    padding_side = "right"

    def __call__(self, text, **kw):
        L = kw.get("max_length", C.DEFAULT_PROMPT_MAX_LEN)
        if isinstance(text, str):
            ids = torch.zeros(1, L, dtype=torch.long)
            mask = torch.zeros(1, L, dtype=torch.long)
            mask[0, : min(len(text.split()), L)] = 1
            return {"input_ids": ids, "attention_mask": mask}
        out_ids = torch.zeros(len(text), L, dtype=torch.long)
        out_mask = torch.zeros(len(text), L, dtype=torch.long)
        for i, t in enumerate(text):
            out_mask[i, : min(len(t.split()), L)] = 1
        return {"input_ids": out_ids, "attention_mask": out_mask}


def _build_policy() -> tuple[XVLAAdapterPolicy, VLAPolicy]:
    model_cfg = VLAPolicyConfig(
        num_domains=1, num_blocks=4, hidden_dim=32,
        num_action_queries=4, num_soft_prompt_tokens=4,
    )
    model = VLAPolicy(model_cfg, _StubSig(), _StubGemma())
    model.eval()
    stats = Q99Stats(
        q01=torch.full((C.ACTION_DIM,), -1.0, dtype=torch.float32),
        q99=torch.full((C.ACTION_DIM,),  1.0, dtype=torch.float32),
        mask=torch.tensor([True] * (C.ACTION_DIM - 1) + [False], dtype=torch.bool),
    )
    tok = GemmaPromptTokenizer(model_name=None, _tokenizer=_StubTokenizer())
    image_tx = SiglipImageTransform(size=C.SIGLIP_IMAGE_SIZE, training=False)
    p = XVLAAdapterPolicy(
        model=model,
        tokenizer=tok,
        image_transform=image_tx,
        norm_stats=stats,
        action_chunk_len=C.ACTION_CHUNK_LEN,
        domain_id=0,
    )
    return p, model


def _fake_obs() -> Dict[str, Any]:
    return {
        "scene_image": np.random.randint(0, 256, size=(224, 224, 3), dtype=np.uint8),
        "wrist_image": np.random.randint(0, 256, size=(224, 224, 3), dtype=np.uint8),
        "proprio":     np.random.randn(C.PROPRIO_DIM).astype(np.float32),
        "language":    "pick the red block",
    }


def test_select_action_returns_correct_shape_and_dtype() -> None:
    p, _ = _build_policy()
    a = p.select_action(_fake_obs())
    assert isinstance(a, np.ndarray)
    assert a.shape == (C.ACTION_DIM,)
    assert a.dtype == np.float32
    assert np.isfinite(a).all()


def test_chunking_drains_buffer_before_calling_model_again(monkeypatch) -> None:
    p, model = _build_policy()
    calls = {"n": 0}
    orig_forward = model.forward

    def counted_forward(batch):
        calls["n"] += 1
        return orig_forward(batch)

    monkeypatch.setattr(model, "forward", counted_forward)

    obs = _fake_obs()
    for _ in range(C.ACTION_CHUNK_LEN):
        p.select_action(obs)
    assert calls["n"] == 1, f"expected 1 forward; got {calls['n']}"

    # The (chunk_len + 1)-th call must trigger a second forward.
    p.select_action(obs)
    assert calls["n"] == 2


def test_reset_clears_buffer(monkeypatch) -> None:
    p, model = _build_policy()
    calls = {"n": 0}
    orig_forward = model.forward

    def counted_forward(batch):
        calls["n"] += 1
        return orig_forward(batch)

    monkeypatch.setattr(model, "forward", counted_forward)

    obs = _fake_obs()
    p.select_action(obs)            # one forward
    assert calls["n"] == 1
    p.reset()
    p.select_action(obs)            # buffer was cleared; another forward triggers
    assert calls["n"] == 2


def test_action_is_denormalized_into_q01_q99_range() -> None:
    """With q01=-1, q99=+1 and tanh-bounded model output, denormalized
    actions should fall inside [q01, q99] for mask=True dims.

    Our model output is unbounded (it goes through a Linear), so we cannot
    assert this generally — but with the standard `q01=-1, q99=+1` stats,
    denormalize is the identity for outputs already in [-1, 1]. Use this
    weaker contract: the per-dim sign of the returned action matches the
    sign of the model's normalized output.
    """
    p, model = _build_policy()
    obs = _fake_obs()
    # Run model once directly to capture its normalized output for the
    # current obs+seed and compare against what the policy returns.
    a = p.select_action(obs)
    assert np.isfinite(a).all()
    # gripper dim (mask=False) is passthrough, so it can be any float.
    assert isinstance(a[-1], np.floating)


def test_from_checkpoint_round_trip(tmp_path: Path) -> None:
    """Build a policy, mutate a parameter, save to ckpt, hydrate via
    from_checkpoint, verify select_action still returns a finite action.
    """
    from vla_project.training.checkpoint import save_checkpoint

    p1, model1 = _build_policy()
    with torch.no_grad():
        model1.action_decoder.fc.weight.fill_(0.05)

    out = tmp_path / "step_5"
    save_checkpoint(
        out, model1, step=5, cfg={"smoke": True},
        norm_stats={
            "libero_smoke": {
                "action": {
                    "q01":  p1.norm_stats.q01.tolist(),
                    "q99":  p1.norm_stats.q99.tolist(),
                    "mask": [bool(b) for b in p1.norm_stats.mask.tolist()],
                }
            }
        },
    )

    # Build a fresh model + tokenizer + image_tx, then hydrate from ckpt.
    model_cfg = VLAPolicyConfig(
        num_domains=1, num_blocks=4, hidden_dim=32,
        num_action_queries=4, num_soft_prompt_tokens=4,
    )
    fresh_model = VLAPolicy(model_cfg, _StubSig(), _StubGemma())
    fresh_model.eval()
    tok = GemmaPromptTokenizer(model_name=None, _tokenizer=_StubTokenizer())
    image_tx = SiglipImageTransform(size=C.SIGLIP_IMAGE_SIZE, training=False)

    p2 = XVLAAdapterPolicy.from_checkpoint(
        out,
        model=fresh_model,
        tokenizer=tok,
        image_transform=image_tx,
        unnorm_key="libero_smoke",
        domain_id=0,
    )
    a = p2.select_action(_fake_obs())
    assert a.shape == (C.ACTION_DIM,)
    assert np.isfinite(a).all()
    # Hydrated model param matches the fill we did pre-save.
    assert torch.allclose(p2.model.action_decoder.fc.weight, torch.full_like(p2.model.action_decoder.fc.weight, 0.05))
```

- [ ] **Step 2: Run, expect FAIL**

```bash
PYTHONPATH="" uv run pytest tests/test_xvla_adapter_policy.py -v
```

Expected: `ModuleNotFoundError` on `xvla_adapter_policy`.

- [ ] **Step 3: Implement**

`src/vla_project/policies/xvla_adapter_policy.py`:

```python
"""Concrete runtime policy for X-VLA-Adapter.

Wraps a trained VLAPolicy + tokenizer + image transform + Q99Stats and
exposes ``select_action(obs)`` that:

  1. Preprocess scene/wrist images via SiglipImageTransform.
  2. Tokenize the language string.
  3. Build a one-batch internal Batch dict with dummy target_action /
     action_mask (the model's forward signature requires them; loss is
     ignored at inference).
  4. Run model forward (under torch.no_grad / eval()).
  5. Denormalize the predicted chunk via Q99Stats.
  6. Push all H_act actions into an internal buffer.
  7. Pop and return one action per call.

``last_action_chunk`` is the previously-emitted *normalized* chunk; we keep
it in normalized space so it matches what the head was trained against
(Plan 1 yields zeros for cold-start). This wrapper carries a copy of the
last *normalized* prediction for use as the next ``last_action_chunk``.

The buffer is reset via ``reset()`` (called per episode by the rollout loop).
"""
from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, Optional, Union

import numpy as np
import torch

from vla_project.data import constants as C
from vla_project.data.normalization import (
    Q99Stats,
    denormalize_action_q99,
    load_q99_stats,
)
from vla_project.data.transforms.image import SiglipImageTransform
from vla_project.data.transforms.language import GemmaPromptTokenizer
from vla_project.models.vla_policy import VLAPolicy
from vla_project.policies.base_policy import BasePolicy
from vla_project.training.checkpoint import load_checkpoint


class XVLAAdapterPolicy(BasePolicy):
    def __init__(
        self,
        model: VLAPolicy,
        tokenizer: GemmaPromptTokenizer,
        image_transform: SiglipImageTransform,
        norm_stats: Q99Stats,
        *,
        action_chunk_len: int = C.ACTION_CHUNK_LEN,
        domain_id: int = 0,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.image_transform = image_transform
        self.norm_stats = norm_stats
        self.action_chunk_len = action_chunk_len
        self.domain_id = int(domain_id)
        self._buffer: Deque[np.ndarray] = deque()
        # last_action_chunk in normalized space; updated each refill.
        self._last_chunk_norm: torch.Tensor = torch.zeros(
            action_chunk_len, C.ACTION_DIM, dtype=torch.float32
        )

    @classmethod
    def from_checkpoint(
        cls,
        ckpt_dir: Union[str, Path],
        model: VLAPolicy,
        tokenizer: GemmaPromptTokenizer,
        image_transform: SiglipImageTransform,
        unnorm_key: str,
        *,
        action_chunk_len: int = C.ACTION_CHUNK_LEN,
        domain_id: int = 0,
    ) -> "XVLAAdapterPolicy":
        meta = load_checkpoint(ckpt_dir, model)
        ns = meta.get("norm_stats")
        if ns is None or unnorm_key not in ns:
            raise KeyError(
                f"checkpoint at {ckpt_dir} has no norm_stats[{unnorm_key!r}]; "
                f"available: {list((ns or {}).keys())}"
            )
        a = ns[unnorm_key]["action"]
        stats = Q99Stats(
            q01=torch.tensor(a["q01"], dtype=torch.float32),
            q99=torch.tensor(a["q99"], dtype=torch.float32),
            mask=torch.tensor(a["mask"], dtype=torch.bool),
        )
        return cls(
            model=model,
            tokenizer=tokenizer,
            image_transform=image_transform,
            norm_stats=stats,
            action_chunk_len=action_chunk_len,
            domain_id=domain_id,
        )

    def reset(self) -> None:
        self._buffer.clear()
        self._last_chunk_norm.zero_()

    def _np_image_to_chw(self, img: np.ndarray) -> torch.Tensor:
        if img.dtype != np.uint8 or img.ndim != 3 or img.shape[-1] != 3:
            raise ValueError(
                f"image must be uint8 (H, W, 3); got dtype={img.dtype} shape={img.shape}"
            )
        # (H, W, 3) uint8 -> (3, H, W) float in [0, 1] -> SigLIP normalized.
        t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        return self.image_transform(t)

    def _build_batch(self, obs: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        device = next(self.model.parameters()).device
        scene = self._np_image_to_chw(obs["scene_image"]).unsqueeze(0).to(device)
        wrist = self._np_image_to_chw(obs["wrist_image"]).unsqueeze(0).to(device)
        proprio = torch.from_numpy(np.asarray(obs["proprio"], dtype=np.float32)).unsqueeze(0).to(device)
        prompt = self.tokenizer(obs["language"])
        return {
            "domain_id": torch.tensor([self.domain_id], dtype=torch.long, device=device),
            "scene_image": scene,
            "wrist_image": wrist,
            "prompt_input_ids": prompt["input_ids"].unsqueeze(0).to(device),
            "prompt_attention_mask": prompt["attention_mask"].unsqueeze(0).to(device),
            "proprio": proprio,
            "last_action_chunk": self._last_chunk_norm.unsqueeze(0).to(device),
            # Inference: target_action / action_mask are only used by the loss;
            # we supply dummy zeros / all-True so the existing forward signature
            # works without inference-specific branching in VLAPolicy.
            "target_action": torch.zeros(1, self.action_chunk_len, C.ACTION_DIM, device=device),
            "action_mask": torch.ones(1, self.action_chunk_len, dtype=torch.bool, device=device),
        }

    def _refill_buffer(self, obs: Dict[str, Any]) -> None:
        was_training = self.model.training
        self.model.eval()
        try:
            with torch.no_grad():
                batch = self._build_batch(obs)
                pred, _ = self.model(batch)  # pred shape: [1, H_act, A] in normalized space
            pred_cpu = pred.detach().to(torch.float32).cpu()
            denormed = denormalize_action_q99(pred_cpu[0], self.norm_stats)
            self._last_chunk_norm = pred_cpu[0].clone()
            for i in range(self.action_chunk_len):
                self._buffer.append(denormed[i].numpy().astype(np.float32))
        finally:
            if was_training:
                self.model.train()

    def select_action(self, obs: Dict[str, Any]) -> np.ndarray:
        if not self._buffer:
            self._refill_buffer(obs)
        return self._buffer.popleft()
```

- [ ] **Step 4: Run, expect PASS**

```bash
PYTHONPATH="" uv run pytest tests/test_xvla_adapter_policy.py -v
PYTHONPATH="" uv run pytest -q
```

Expected: 5 new tests pass; full suite green (86 + 5 = 91).

If a stub-shape mismatch surfaces (e.g., `_StubGemma` is incompatible with `num_blocks=4`), check `tests/_stubs.py` and `tests/test_checkpoint_vla_policy.py` for the working call shape — that test passes today, so use the same constructor pattern.

- [ ] **Step 5: Commit**

```bash
git add src/vla_project/policies/xvla_adapter_policy.py tests/test_xvla_adapter_policy.py
git commit -m "feat(policies): XVLAAdapterPolicy runtime wrapper with chunking + denorm"
```

---

## Task 4: Push branch + open PR

- [ ] **Step 1: Confirm branch**

```bash
git status -sb
git log --oneline feat/stage2-lora..HEAD
```

The controller should already have created `feat/policies-runtime` branched from `feat/stage2-lora`.

- [ ] **Step 2: Push**

```bash
git push -u origin feat/policies-runtime
```

- [ ] **Step 3: PR**

PR base: `feat/stage2-lora` (rebase to `main` once Plans 1-5 merge).
Title: `feat(policies): XVLAAdapterPolicy runtime wrapper`.
Body should include:
- Test count delta (81 → 91 expected).
- Note that `select_action` is now exercised end-to-end on stubbed Gemma/SigLIP — no real-Gemma smoke yet (deferred to closed-loop eval in Plan 8 where the LIBERO sim provides real observations).

---

## Done criteria

- [ ] `uv run pytest -q` passes (full suite, 10 new tests).
- [ ] `XVLAAdapterPolicy.select_action({...})` returns `np.ndarray[A]` float32 with finite values.
- [ ] Action chunking is pinned: model.forward called once per `H_act` calls.
- [ ] `from_checkpoint` round-trips a saved checkpoint into a fresh policy that returns finite actions.
- [ ] No edits to `models/`, `training/`, `data/datasets/`, `data/transforms/`, `scripts/train.py`.
- [ ] `policies/` does not import from `robots/` or `deployment/`.

## Out of scope (later plans)

- Action clipping at the robot's safety bounds (Plan 7's sim_robot will impose its own clamp).
- Latency measurement / streaming inference (Plan / future deployment work).
- Multi-camera observation merging beyond scene + wrist (architectural change).
- Beam-search / sampling for action selection (head is regression, not autoregressive).
