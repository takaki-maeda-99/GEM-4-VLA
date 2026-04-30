# Wrist Pooling + Huber Loss + InputPacker Refactor (Plan 9 / 10)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land two ablations gated by config flags — (a) wrist token pooling (`use_wrist_pool: true` → adaptive 7×7 avgpool reducing wrist tokens 256→49) and (b) Huber loss (`loss_type: huber`). Plumbing requires fixing the latent bug discovered during Plan 6's review: `InputPacker` reads `NUM_*_TOKENS` from constants instead of cfg, so `VLAPolicyConfig.num_action_queries` / `num_soft_prompt_tokens` are misleading. Refactor `InputPacker` to read all four token counts from constructor args, plumb them from `VLAPolicyConfig`, and wire the new `wrist_token_count` field that wrist pooling drives.

**Architecture:** Three layers:

1. `InputPacker` constructor takes four explicit token counts (`num_soft_prompt_tokens`, `num_scene_tokens`, `num_wrist_tokens`, `num_action_queries`) defaulting to the existing constants. The `register_buffer` placeholder templates are sized accordingly. Existing call sites pass through their cfg fields.

2. `VLAPolicyConfig` gains `num_wrist_tokens: int = NUM_WRIST_TOKENS`, `use_wrist_pool: bool = False`, `wrist_pool_tokens: int = 49`. When `use_wrist_pool=True`, the resolved `num_wrist_tokens` is set to `wrist_pool_tokens` for InputPacker construction. `VLAPolicy.forward` applies an `nn.AdaptiveAvgPool2d((7, 7))` (or matching) on the wrist token grid after SigLIP and before scattering. The action head's `num_task_tokens` is recomputed from the resolved wrist count.

3. Loss switch: `cfg.loss_type` and `cfg.huber_beta` already exist on `VLAPolicyConfig` and the forward already dispatches; we simply ensure they're config-driven (already are) and add a smoke that exercises Huber.

The synthetic + real-data smokes from earlier plans must continue to pass with both flags off (defaults). New smokes verify both flags on.

**Tech Stack:** existing torch, no new deps.

**Repo references:**
- `src/vla_project/data/packing/input_packer.py` — the refactor target.
- `src/vla_project/models/vla_policy.py:54-67` — current `VLAPolicy.__init__` constructs `InputPacker` and `L1RegressionActionHead` reading constants.
- `src/vla_project/models/vla_policy.py:82-148` — `forward` where wrist pool will splice.
- `src/vla_project/training/losses.py` — `masked_l1` and `masked_huber` already exist.
- `docs/architectures/x_vla_adapter.md` — Open questions / future work mentions wrist pooling and Huber as documented switches.

**Hard constraints from CLAUDE.md:**
- Architecture variants live in config.
- Fail-fast on shape mismatches: if `use_wrist_pool=True` but `num_wrist_tokens` does not equal `wrist_pool_tokens`, raise.
- Keep model code (vla_policy.py) free of robot/policy concerns.
- Existing tests must not regress.

---

## File Structure

**Modify:**
- `src/vla_project/data/packing/input_packer.py` (constructor takes token counts)
- `src/vla_project/models/vla_policy.py` (add `num_wrist_tokens`, `use_wrist_pool`, `wrist_pool_tokens`; pool in forward)
- `tests/test_input_packer.py` (verify config-driven behavior; existing tests still pass)
- `tests/test_vla_policy_forward.py` (add wrist-pool variant test)
- `tests/test_masked_loss.py` (extend Huber path coverage if not already there)
- `scripts/train.py` (no changes needed — passes cfg.model dict through)

**Create:**
- `tests/test_input_packer_config.py` (new tests pinning configurable token counts)
- `tests/test_wrist_pool_forward.py` (forward shape with pooled wrist)
- `configs/train/smoke_wristpool_huber.yaml` (real-data smoke with both ablations on)

**Do not modify:** `data/datasets/`, `data/transforms/`, `policies/`, `robots/`, `evaluation/`.

---

## Task 1: Refactor `InputPacker` to take token counts

**Files:**
- Modify: `src/vla_project/data/packing/input_packer.py`
- Create: `tests/test_input_packer_config.py`

- [ ] **Step 1: Write failing test**

`tests/test_input_packer_config.py`:

```python
"""Pin that InputPacker honors config-driven token counts (not just constants)."""
import torch

from vla_project.data import constants as C
from vla_project.data.packing.input_packer import InputPacker


def test_default_constructor_matches_constants() -> None:
    pk = InputPacker(bos_id=2, eos_id=1, prompt_max_len=10)
    out = pk(
        torch.zeros(1, 10, dtype=torch.long),
        torch.ones(1, 10, dtype=torch.long),
    )
    # 1 BOS + Ks + Ns + Lp + Nw + Q + 1 EOS
    expected_len = (
        1
        + C.NUM_SOFT_PROMPT_TOKENS
        + C.NUM_SCENE_TOKENS
        + 10
        + C.NUM_WRIST_TOKENS
        + C.NUM_ACTION_TOKENS
        + 1
    )
    assert out.input_ids.shape == (1, expected_len)
    assert out.idx["soft"].shape == (1, C.NUM_SOFT_PROMPT_TOKENS)
    assert out.idx["wrist"].shape == (1, C.NUM_WRIST_TOKENS)
    assert out.idx["action"].shape == (1, C.NUM_ACTION_TOKENS)


def test_custom_token_counts() -> None:
    pk = InputPacker(
        bos_id=2, eos_id=1, prompt_max_len=10,
        num_soft_prompt_tokens=8,
        num_scene_tokens=16,
        num_wrist_tokens=49,
        num_action_queries=12,
    )
    out = pk(
        torch.zeros(2, 10, dtype=torch.long),
        torch.ones(2, 10, dtype=torch.long),
    )
    expected_len = 1 + 8 + 16 + 10 + 49 + 12 + 1
    assert out.input_ids.shape == (2, expected_len)
    assert out.idx["soft"].shape  == (2, 8)
    assert out.idx["scene"].shape == (2, 16)
    assert out.idx["wrist"].shape == (2, 49)
    assert out.idx["action"].shape == (2, 12)


def test_wrist_pool_value_within_constants() -> None:
    """A pooled count of 49 must be a legal value to pass."""
    pk = InputPacker(
        bos_id=2, eos_id=1, prompt_max_len=10,
        num_wrist_tokens=49,
    )
    out = pk(
        torch.zeros(1, 10, dtype=torch.long),
        torch.ones(1, 10, dtype=torch.long),
    )
    assert out.idx["wrist"].shape == (1, 49)


def test_rejects_non_positive_counts() -> None:
    import pytest
    with pytest.raises(ValueError):
        InputPacker(bos_id=2, eos_id=1, prompt_max_len=10, num_wrist_tokens=0)
    with pytest.raises(ValueError):
        InputPacker(bos_id=2, eos_id=1, prompt_max_len=10, num_action_queries=-1)
```

- [ ] **Step 2: Run, expect FAIL**

```bash
PYTHONPATH="" uv run pytest tests/test_input_packer_config.py -v
```

Expected: TypeError on the constructor (extra kwargs rejected) or AssertionError on shape.

- [ ] **Step 3: Implement**

Edit `src/vla_project/data/packing/input_packer.py` — replace `__init__` signature and body. Preserve `forward()` as-is:

```python
def __init__(
    self,
    bos_id: int,
    eos_id: int,
    prompt_max_len: int,
    num_soft_prompt_tokens: int = C.NUM_SOFT_PROMPT_TOKENS,
    num_scene_tokens: int = C.NUM_SCENE_TOKENS,
    num_wrist_tokens: int = C.NUM_WRIST_TOKENS,
    num_action_queries: int = C.NUM_ACTION_TOKENS,
) -> None:
    super().__init__()
    if num_soft_prompt_tokens <= 0:
        raise ValueError(f"num_soft_prompt_tokens must be > 0; got {num_soft_prompt_tokens}")
    if num_scene_tokens <= 0:
        raise ValueError(f"num_scene_tokens must be > 0; got {num_scene_tokens}")
    if num_wrist_tokens <= 0:
        raise ValueError(f"num_wrist_tokens must be > 0; got {num_wrist_tokens}")
    if num_action_queries <= 0:
        raise ValueError(f"num_action_queries must be > 0; got {num_action_queries}")
    self.bos_id = bos_id
    self.eos_id = eos_id
    self.prompt_max_len = prompt_max_len
    self.num_soft_prompt_tokens = num_soft_prompt_tokens
    self.num_scene_tokens = num_scene_tokens
    self.num_wrist_tokens = num_wrist_tokens
    self.num_action_queries = num_action_queries

    soft = torch.arange(C.SOFT_PROMPT_BEGIN_IDX,
                        C.SOFT_PROMPT_BEGIN_IDX + num_soft_prompt_tokens)
    scene = torch.full((num_scene_tokens,), C.IMAGE_SOFT_TOKEN_ID, dtype=torch.long)
    wrist = torch.arange(C.WRIST_PLACEHOLDER_BEGIN_IDX,
                         C.WRIST_PLACEHOLDER_BEGIN_IDX + num_wrist_tokens)
    action = torch.arange(C.ACTION_TOKEN_BEGIN_IDX,
                          C.ACTION_TOKEN_BEGIN_IDX + num_action_queries)
    self.register_buffer("_soft", soft, persistent=False)
    self.register_buffer("_scene", scene, persistent=False)
    self.register_buffer("_wrist", wrist, persistent=False)
    self.register_buffer("_action", action, persistent=False)
```

`forward()` does NOT need changes — it already uses the registered buffers' shapes via `.shape[1]`.

- [ ] **Step 4: Run, expect PASS**

```bash
PYTHONPATH="" uv run pytest tests/test_input_packer_config.py tests/test_input_packer.py tests/test_ple_shape.py tests/test_inputs_embeds_overwrite.py -v
PYTHONPATH="" uv run pytest -q
```

Expected: 4 new tests pass; existing input_packer tests still pass; full suite green (102 + 4 = 106).

- [ ] **Step 5: Commit**

```bash
git add src/vla_project/data/packing/input_packer.py tests/test_input_packer_config.py
git commit -m "refactor(data): InputPacker takes per-block token counts via cfg"
```

---

## Task 2: Plumb token counts through `VLAPolicy`

**Files:**
- Modify: `src/vla_project/models/vla_policy.py`

- [ ] **Step 1: Read existing VLAPolicy and find the construction sites**

```bash
sed -n '17,67p' src/vla_project/models/vla_policy.py
```

The relevant lines to edit:
- `VLAPolicyConfig` dataclass: confirm `num_action_queries`, `num_soft_prompt_tokens` are present (they are). Add `num_wrist_tokens: int = C.NUM_WRIST_TOKENS`, `num_scene_tokens: int = C.NUM_SCENE_TOKENS`, `use_wrist_pool: bool = False`, `wrist_pool_tokens: int = 49`.
- `__init__` body where `InputPacker` and `L1RegressionActionHead` are constructed.

- [ ] **Step 2: Edit `VLAPolicyConfig`**

Append the four new fields:

```python
@dataclass
class VLAPolicyConfig:
    num_domains: int
    hidden_dim: int = C.LLM_HIDDEN_DIM
    siglip_hidden_dim: int = C.SIGLIP_HIDDEN_DIM
    action_dim: int = C.ACTION_DIM
    action_chunk_len: int = C.ACTION_CHUNK_LEN
    proprio_dim: int = C.PROPRIO_DIM
    prompt_max_len: int = C.DEFAULT_PROMPT_MAX_LEN
    num_blocks: int = C.NUM_LLM_LAYERS
    num_soft_prompt_tokens: int = C.NUM_SOFT_PROMPT_TOKENS
    num_action_queries: int = C.NUM_ACTION_TOKENS
    num_scene_tokens: int = C.NUM_SCENE_TOKENS  # NEW
    num_wrist_tokens: int = C.NUM_WRIST_TOKENS  # NEW (raw count from SigLIP)
    use_wrist_pool: bool = False                # NEW
    wrist_pool_tokens: int = 49                 # NEW (when use_wrist_pool=True)
    bos_id: int = 2
    eos_id: int = 1
    loss_type: str = "l1"  # or "huber"
    huber_beta: float = 0.1
    use_grad_checkpoint: bool = False
```

- [ ] **Step 3: Edit `VLAPolicy.__init__`**

Replace the `InputPacker` and `L1RegressionActionHead` constructions:

```python
        # Resolve effective wrist token count: pooled value when enabled.
        effective_num_wrist = (
            cfg.wrist_pool_tokens if cfg.use_wrist_pool else cfg.num_wrist_tokens
        )
        if cfg.use_wrist_pool and effective_num_wrist <= 0:
            raise ValueError(
                f"wrist_pool_tokens must be > 0 when use_wrist_pool=True; got {cfg.wrist_pool_tokens}"
            )
        self._effective_num_wrist = int(effective_num_wrist)

        self.input_packer = InputPacker(
            cfg.bos_id, cfg.eos_id, cfg.prompt_max_len,
            num_soft_prompt_tokens=cfg.num_soft_prompt_tokens,
            num_scene_tokens=cfg.num_scene_tokens,
            num_wrist_tokens=effective_num_wrist,
            num_action_queries=cfg.num_action_queries,
        )

        self.action_head = L1RegressionActionHead(
            hidden_dim=D,
            action_dim=A,
            num_action_chunks=cfg.action_chunk_len,
            num_blocks=cfg.num_blocks,
            num_task_tokens=cfg.num_scene_tokens + cfg.prompt_max_len + effective_num_wrist,
            use_grad_checkpoint=cfg.use_grad_checkpoint,
        )
```

- [ ] **Step 4: Edit `VLAPolicy.forward` to apply wrist pool**

Just after the SigLIP wrist encoding, optionally pool. Insert this between the existing scene/wrist tokenization and the projection. The current code reads:

```python
scene_tok = self.vision_encoder(batch["scene_image"])  # [B, 256, D_vis]
wrist_tok = self.vision_encoder(batch["wrist_image"])
```

Replace with:

```python
scene_tok = self.vision_encoder(batch["scene_image"])  # [B, Ns, D_vis]
wrist_tok = self.vision_encoder(batch["wrist_image"])  # [B, Nw_raw, D_vis]
if self.cfg.use_wrist_pool:
    wrist_tok = self._pool_wrist(wrist_tok)            # [B, Nw_pooled, D_vis]
```

And add a helper method on `VLAPolicy`:

```python
def _pool_wrist(self, wrist_tok: torch.Tensor) -> torch.Tensor:
    """Adaptive-pool a (B, N, D) sequence by reshaping to a square grid.

    Assumes ``N`` is a perfect square (16x16 = 256 for SigLIP@224 / patch14).
    Pools to a grid of side sqrt(self._effective_num_wrist) and flattens back.
    """
    import math
    B, N, D = wrist_tok.shape
    side = int(round(math.sqrt(N)))
    if side * side != N:
        raise ValueError(f"wrist token count {N} is not a perfect square")
    pooled_side = int(round(math.sqrt(self._effective_num_wrist)))
    if pooled_side * pooled_side != self._effective_num_wrist:
        raise ValueError(
            f"wrist_pool_tokens {self._effective_num_wrist} is not a perfect square"
        )
    grid = wrist_tok.transpose(1, 2).reshape(B, D, side, side)
    pooled = torch.nn.functional.adaptive_avg_pool2d(grid, (pooled_side, pooled_side))
    return pooled.reshape(B, D, pooled_side * pooled_side).transpose(1, 2)
```

- [ ] **Step 5: Verify forward shape (existing tests cover this)**

```bash
PYTHONPATH="" uv run pytest tests/test_vla_policy_forward.py -v
PYTHONPATH="" uv run pytest -q
```

Expected: existing forward test still passes (defaults: `use_wrist_pool=False`, `num_wrist_tokens=256`).

- [ ] **Step 6: Commit**

```bash
git add src/vla_project/models/vla_policy.py
git commit -m "feat(models): plumb token counts + wrist pool toggle through VLAPolicy"
```

---

## Task 3: Wrist-pool forward test

**Files:**
- Create: `tests/test_wrist_pool_forward.py`

- [ ] **Step 1: Write the test**

`tests/test_wrist_pool_forward.py`:

```python
"""Forward pass works when use_wrist_pool=True (49 tokens instead of 256)."""
from dataclasses import asdict

import torch

from vla_project.data import constants as C
from vla_project.models.vla_policy import VLAPolicy, VLAPolicyConfig
from tests._stubs import _StubGemma, _StubSig


def _make_batch(B: int = 1) -> dict:
    return {
        "domain_id": torch.zeros(B, dtype=torch.long),
        "scene_image": torch.randn(B, 3, C.SIGLIP_IMAGE_SIZE, C.SIGLIP_IMAGE_SIZE),
        "wrist_image": torch.randn(B, 3, C.SIGLIP_IMAGE_SIZE, C.SIGLIP_IMAGE_SIZE),
        "prompt_input_ids": torch.zeros(B, C.DEFAULT_PROMPT_MAX_LEN, dtype=torch.long),
        "prompt_attention_mask": torch.ones(B, C.DEFAULT_PROMPT_MAX_LEN, dtype=torch.long),
        "proprio": torch.randn(B, C.PROPRIO_DIM),
        "last_action_chunk": torch.randn(B, C.ACTION_CHUNK_LEN, C.ACTION_DIM),
        "target_action": torch.randn(B, C.ACTION_CHUNK_LEN, C.ACTION_DIM),
        "action_mask": torch.ones(B, C.ACTION_CHUNK_LEN, dtype=torch.bool),
    }


def test_forward_with_wrist_pool() -> None:
    cfg = VLAPolicyConfig(
        num_domains=1,
        hidden_dim=32,
        num_blocks=4,
        use_wrist_pool=True,
        wrist_pool_tokens=49,
    )
    model = VLAPolicy(cfg, _StubSig(), _StubGemma())
    pred, loss = model(_make_batch(B=1))
    assert pred.shape == (1, C.ACTION_CHUNK_LEN, C.ACTION_DIM)
    assert torch.isfinite(pred).all()
    assert torch.isfinite(loss)


def test_forward_without_wrist_pool_unchanged_default() -> None:
    """The default (use_wrist_pool=False) path still produces correct shape."""
    cfg = VLAPolicyConfig(num_domains=1, hidden_dim=32, num_blocks=4)
    model = VLAPolicy(cfg, _StubSig(), _StubGemma())
    pred, _loss = model(_make_batch(B=1))
    assert pred.shape == (1, C.ACTION_CHUNK_LEN, C.ACTION_DIM)
```

- [ ] **Step 2: Run, expect both PASS**

```bash
PYTHONPATH="" uv run pytest tests/test_wrist_pool_forward.py -v
PYTHONPATH="" uv run pytest -q
```

Expected: 2 passed; full suite green (106 + 2 = 108).

- [ ] **Step 3: Commit**

```bash
git add tests/test_wrist_pool_forward.py
git commit -m "test(models): wrist-pool forward shape contract"
```

---

## Task 4: Real-data smoke with both ablations on

**Files:**
- Create: `configs/train/smoke_wristpool_huber.yaml`

- [ ] **Step 1: Write the config**

`configs/train/smoke_wristpool_huber.yaml`:

```yaml
seed: 0
model:
  num_domains: 1
  hidden_dim: 1536
  num_blocks: 35
  use_grad_checkpoint: true
  use_wrist_pool: true
  wrist_pool_tokens: 49
  loss_type: huber
  huber_beta: 0.1
vision:
  model_name: google/siglip-so400m-patch14-224
language:
  model_name: google/gemma-4-E2B
data:
  type: libero_lerobot_real
  repo_id: lerobot/libero_spatial_image
  stats_path: ${oc.env:LIBERO_STATS_PATH,data/norm_stats/libero_spatial.json}
  unnorm_key: libero_spatial_no_noops
  fps: 10
  episodes: [0]
  download_videos: false
  domain_id: 0
  max_samples: 16
train:
  batch_size: 1
  lr: 1.0e-4
  soft_lr_coef: 1.0
  weight_decay: 0.01
  max_steps: 2
```

- [ ] **Step 2: Run the smoke**

```bash
cd /misc/dl00/takaki/X-VLA-Adapter
CUDA_VISIBLE_DEVICES=3 PYTHONPATH="" timeout 600 \
    uv run python scripts/train.py configs/train/smoke_wristpool_huber.yaml 2>&1 | tee /tmp/smoke_wristpool_huber.log | tail -3
```

Expected: `[train] losses=[<f1>, <f2>]` with two finite floats. Step 0 loss should differ from baseline (`losses=[0.696, ...]`) because:
- Huber loss has different scale than L1.
- Wrist pool removes 207 tokens from Gemma's input; the head sees fewer task tokens.

Both losses finite + no NaN ⇒ both ablations wire correctly.

- [ ] **Step 3: Verify backward-compat smokes still pass**

```bash
CUDA_VISIBLE_DEVICES=3 PYTHONPATH="" uv run python scripts/train.py configs/train/smoke.yaml 2>&1 | tail -3
CUDA_VISIBLE_DEVICES=3 PYTHONPATH="" uv run python scripts/train.py configs/train/smoke_real.yaml 2>&1 | tail -3
```

Expected: each prints two finite losses. The numerical values may differ slightly from Plan 6/7/8 baselines because the InputPacker constructor signature changed (semantically equivalent but RNG ordering is sensitive).

- [ ] **Step 4: Commit**

```bash
git add configs/train/smoke_wristpool_huber.yaml
git commit -m "feat(configs): real-data smoke with wrist-pool + Huber"
```

---

## Task 5: Push branch + open PR

- [ ] **Step 1: Push**

```bash
git status -sb
git log --oneline feat/libero-eval..HEAD
git push -u origin feat/wrist-pool-huber
```

- [ ] **Step 2: PR**

PR base: `feat/libero-eval` (rebase to `main` after Plans 1-8 merge).
Title: `feat(models): wrist-pool + Huber ablations + InputPacker cfg-driven token counts`.
Body should include:
- Test count delta (102 → 102 + 4 + 2 = 108 expected).
- The `losses=[...]` from the new wrist-pool + Huber smoke.
- Note that this fixes the latent bug surfaced in Plan 6's review (`InputPacker` ignored `num_soft_prompt_tokens` / `num_action_queries` cfg fields).

---

## Done criteria

- [ ] `uv run pytest -q` green (108 expected).
- [ ] `python scripts/train.py configs/train/smoke.yaml` (synthetic) still runs.
- [ ] `python scripts/train.py configs/train/smoke_real.yaml` (single-domain real) still runs.
- [ ] `python scripts/train.py configs/train/smoke_wristpool_huber.yaml` runs ≥ 2 steps without NaN.
- [ ] `InputPacker` honors all four token-count kwargs.
- [ ] `VLAPolicyConfig` exposes `use_wrist_pool` and `wrist_pool_tokens`.

## Out of scope (other plans)

- Pooled wrist with non-square count (e.g., 32 = 4×8). Sqrt-grid pooling is the simplest approach; non-square targets would need a different reshape.
- Wrist-tokens-only ablation without Huber (or vice versa) — independent flags already let users mix freely.
- Re-running the LIBERO closed-loop eval with wrist-pool enabled (Plan 8 smoke covers the no-pool path; pool path is functionally identical from the rollout's perspective).
