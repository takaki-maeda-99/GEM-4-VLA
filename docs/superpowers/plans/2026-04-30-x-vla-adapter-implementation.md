# X-VLA-Adapter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a complete, working X-VLA-Adapter VLA model from scratch — Gemma4-E2B backbone + SigLIP shared vision + per-domain projections + 35x `MLPResNetBlock_Pro` action head + masked L1 regression — and reach a green end-to-end smoke pass on a tiny LIBERO batch.

**Architecture:** See `docs/architectures/x_vla_adapter.md` for the module/tensor contract. Vision: SigLIP-So400m, shared between scene and wrist. LLM: Gemma4-E2B with PaliGemma-style placeholder `input_ids` and PLE pre-compute. Head: 35-block stack of `MLPResNetBlock_Pro` with per-block task gating, conditioned by `last_action_chunk` (as `x` init) and `proprio` (as `p`). Loss: masked L1 (Huber as ablation).

**Tech Stack:** Python 3.10+, uv, PyTorch 2.x, `transformers` (>= the first release that ships Gemma4 — `>=4.46`), `accelerate`, `pytest`, `omegaconf`, `einops`. SigLIP / Gemma4 weights from Hugging Face.

**Reference checkouts already on disk** (read-only — do **not** modify):
- `/home/takakimaeda/X-VLA-Adapter/X-VLA/` — X-VLA dataloader, soft prompt hub, `DomainAwareLinear`, `action_hub.py`.
- `/home/takakimaeda/X-VLA-Adapter/VLA-Adapter/` — `MLPResNetBlock`, `MLPResNetBlock_Pro`, `MLPResNet`, `L1RegressionActionHead`, multi-layer hidden-state extraction.
- `/home/takakimaeda/vla-gemma-4/VLA-Adapter/prismatic/extern/hf/modeling_prismatic_gemma4.py` — proven Gemma4 PLE handling, placeholder ID overwrite, soft prompt routing.
- `/home/takakimaeda/vla-gemma-4/VLA-Adapter/prismatic/vla/constants_gemma4.py` — exact unused-token ranges and Gemma4 metadata.

---

## Stages and tasks

| Stage | Goal | Tasks |
|-------|------|-------|
| 0 | Repository skeleton, tooling, CI sanity | 1-2 |
| 1 | Constants + batch schema + normalization | 3, 4, 4.5 |
| 2 | Per-domain primitives (`DomainAwareLinear`, `SoftPromptHub`, `ActionQueryHub`) | 5-7 |
| 3 | Vision encoder wrapper (SigLIP shared) | 8 |
| 4 | Input packer (placeholder construction + index dict) | 9-10 |
| 5 | Gemma4 backbone wrapper (PLE precompute, embed overwrite) | 11 |
| 6 | Action head (`RoPE`, `MLPResNetBlock_Pro`, `MLPResNet`, `L1RegressionActionHead`) | 12-16 |
| 7 | Masked loss (L1 default, Huber switch) | 17 |
| 8 | Combined `VLAPolicy` + shared stubs + freeze policy + spec tests | 17.5, 18, 19, 19.1-19.4 |
| 9 | LIBERO data layer | 20-22 |
| 10 | LR scheduler / Optimizer / Trainer / smoke train | 22.5, 23-25 |
| 11 | Config files | 26-28 |
| 12 | End-to-end smoke test | 29 |

Hard rules carried from `CLAUDE.md`:
- Keep `data` / `models` / `policies` / `training` / `robots` boundaries.
- All datasets must reach the model through the **internal batch schema** in `vla_project/data/schema.py`.
- Do not hardcode dataset paths or shapes inside model code; configs drive everything.
- Each architecture-touching task must have at least one shape/contract test.

## File Structure

```
X-VLA-Adapter/
├── pyproject.toml                  # uv project, deps
├── .gitignore                      # add outputs/, checkpoints/, .venv/
├── CLAUDE.md                       # already exists, do not edit
├── README.md                       # already exists, do not edit
├── docs/                           # already exists, do not edit
├── X-VLA/                          # submodule, read-only
├── VLA-Adapter/                    # submodule, read-only
│
├── src/vla_project/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── constants.py            # placeholder ID ranges, special tokens
│   │   ├── schema.py               # internal Batch dict typing
│   │   ├── normalization.py        # action / proprio mean+std utilities (centralized)
│   │   ├── transforms/
│   │   │   ├── __init__.py
│   │   │   ├── image.py            # SigLIP-aware resize + normalize
│   │   │   ├── proprio.py
│   │   │   ├── action.py
│   │   │   └── language.py         # Gemma4 tokenizer wrapping + padding
│   │   ├── packing/
│   │   │   ├── __init__.py
│   │   │   └── input_packer.py     # data-side: builds Gemma4 input_ids + idx
│   │   ├── collators.py
│   │   └── datasets/
│   │       ├── __init__.py
│   │       └── libero_dataset.py   # step-level reader for LIBERO
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── projectors/
│   │   │   ├── __init__.py
│   │   │   ├── domain_aware_linear.py
│   │   │   ├── soft_prompts.py     # SoftPromptHub (per CLAUDE.md projectors/)
│   │   │   └── action_queries.py   # ActionQueryHub (per CLAUDE.md projectors/)
│   │   ├── vision/
│   │   │   ├── __init__.py
│   │   │   └── siglip.py
│   │   ├── language/
│   │   │   ├── __init__.py
│   │   │   ├── gemma4_wrapper.py
│   │   │   └── embed_overwrite.py  # scatter_into_embeds (LLM-side helper)
│   │   ├── action_heads/
│   │   │   ├── __init__.py
│   │   │   ├── rope.py
│   │   │   ├── mlp_resnet_block_pro.py
│   │   │   ├── mlp_resnet.py
│   │   │   └── l1_regression_action_head.py
│   │   └── vla_policy.py           # combined nn.Module
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   ├── losses.py               # masked_l1, masked_huber
│   │   ├── optim.py                # per-group LR helper
│   │   ├── schedulers.py           # linear-warmup + cosine
│   │   └── trainer.py              # Accelerate-based loop
│   │
│   └── utils/
│       ├── __init__.py
│       ├── seed.py
│       └── io.py
│
├── configs/
│   ├── data/libero.yaml
│   ├── model/x_vla_adapter.yaml
│   └── train/smoke.yaml
│
├── scripts/
│   └── train.py
│
└── tests/
    ├── __init__.py
    ├── conftest.py
    ├── _stubs.py                   # shared _StubSig / _StubGemma fixtures
    ├── test_constants.py
    ├── test_schema.py
    ├── test_domain_aware_linear.py
    ├── test_soft_prompts.py
    ├── test_action_queries.py
    ├── test_siglip_wrapper.py
    ├── test_input_packer.py
    ├── test_inputs_embeds_overwrite.py
    ├── test_gemma4_wrapper.py
    ├── test_ple_shape.py
    ├── test_rope.py
    ├── test_mlp_resnet_block_pro.py
    ├── test_mlp_resnet.py
    ├── test_l1_regression_action_head.py
    ├── test_masked_loss.py
    ├── test_action_loss_mask_grad.py
    ├── test_vla_policy_forward.py
    ├── test_action_query_extraction.py
    ├── test_domain_aware_swap_full.py
    ├── test_trainable_parameters.py
    ├── test_normalization.py
    ├── test_libero_dataset.py
    └── test_one_batch_smoke.py
```

**Boundary notes (CLAUDE.md compliance):**
- `data/packing/` (not `models/packing/`): the input packer produces `input_ids` (a data tensor), so it lives in the data layer per CLAUDE.md's "Keep boundaries clear" rule.
- `models/language/embed_overwrite.py`: `scatter_into_embeds` is a Gemma-input helper, kept adjacent to the LLM wrapper.
- `projectors/soft_prompts.py` + `projectors/action_queries.py`: per-domain learnable embeddings are projection-like, so they live under `models/projectors/` (matching CLAUDE.md's "vision/proprio/action-token projection modules").
- `policies/`, `evaluation/`, `deployment/`, `robots/` are out of scope for Stage 1 smoke; tracked in **Follow-ups**.

---

## Stage 0: Repository skeleton

### Task 1: Bootstrap uv project and base layout

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `src/vla_project/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`

- [ ] **Step 1: Initialize uv project**

```bash
cd /home/takakimaeda/X-VLA-Adapter
uv init --package --name vla_project --no-readme
```

This creates `pyproject.toml` and `src/vla_project/__init__.py`. Do **not** let it overwrite the existing `README.md` or `CLAUDE.md`.

- [ ] **Step 2: Pin runtime + dev deps**

Edit `pyproject.toml` to:

```toml
[project]
name = "vla_project"
version = "0.0.1"
description = "X-VLA-Adapter: Gemma4 + SigLIP + per-domain VLA-Adapter head"
requires-python = ">=3.10"
dependencies = [
    "torch>=2.4",
    "torchvision>=0.19",
    "transformers>=4.50",  # Gemma3n / Gemma4 PLE landed in 4.50+
    "accelerate>=1.0",
    "omegaconf>=2.3",
    "einops>=0.8",
    "tqdm>=4.66",
    "numpy>=1.26",
    "pillow>=10.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-xdist>=3.5",
    "ruff>=0.6",
    "mypy>=1.10",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/vla_project"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-q -ra"
filterwarnings = ["ignore::UserWarning"]
```

Then:

```bash
uv sync --extra dev
```

- [ ] **Step 3: Add `.gitignore`**

```
.venv/
__pycache__/
*.pyc
.pytest_cache/
.mypy_cache/
.ruff_cache/
outputs/
checkpoints/
data/
*.egg-info/
.DS_Store
```

- [ ] **Step 4: Sanity test that imports the package**

`tests/conftest.py`:

```python
import torch

def pytest_configure(config):
    torch.manual_seed(0)
```

`tests/test_smoke.py`:

```python
import vla_project


def test_package_imports():
    assert hasattr(vla_project, "__name__")
```

- [ ] **Step 5: Run pytest, expect PASS**

```bash
uv run pytest -v
```

Expected: 1 passed.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml .gitignore src tests
git commit -m "chore: bootstrap uv project layout"
```

---

### Task 2: Utility modules — `seed`, `io`

**Files:**
- Create: `src/vla_project/utils/__init__.py`
- Create: `src/vla_project/utils/seed.py`
- Create: `src/vla_project/utils/io.py`
- Create: `tests/test_utils.py`

- [ ] **Step 1: Write failing tests**

`tests/test_utils.py`:

```python
import torch

from vla_project.utils.seed import set_seed
from vla_project.utils.io import load_yaml, save_yaml


def test_set_seed_makes_torch_deterministic(tmp_path):
    set_seed(42)
    a = torch.randn(3)
    set_seed(42)
    b = torch.randn(3)
    assert torch.equal(a, b)


def test_yaml_roundtrip(tmp_path):
    cfg = {"a": 1, "b": [2, 3], "c": {"d": "e"}}
    path = tmp_path / "x.yaml"
    save_yaml(cfg, path)
    assert load_yaml(path) == cfg
```

- [ ] **Step 2: Run, expect FAIL**

`uv run pytest tests/test_utils.py -v` → ImportError.

- [ ] **Step 3: Implement**

`src/vla_project/utils/seed.py`:

```python
import os
import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
```

`src/vla_project/utils/io.py`:

```python
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf


def load_yaml(path: str | Path) -> Any:
    return OmegaConf.to_container(OmegaConf.load(str(path)), resolve=True)


def save_yaml(obj: Any, path: str | Path) -> None:
    OmegaConf.save(OmegaConf.create(obj), str(path))
```

- [ ] **Step 4: Run, expect PASS**

- [ ] **Step 5: Commit**

```bash
git add src/vla_project/utils tests/test_utils.py
git commit -m "feat(utils): add deterministic seed and yaml io helpers"
```

---

## Stage 1: Constants and Batch schema

### Task 3: Placeholder ID ranges and special tokens

**Files:**
- Create: `src/vla_project/data/__init__.py`
- Create: `src/vla_project/data/constants.py`
- Create: `tests/test_constants.py`

These IDs are reused from `/home/takakimaeda/vla-gemma-4/VLA-Adapter/prismatic/vla/constants_gemma4.py`. They are **non-overlapping** sub-ranges of Gemma4's 6227 unused tokens.

- [ ] **Step 1: Write failing tests**

`tests/test_constants.py`:

```python
from vla_project.data import constants as C


def test_ranges_dont_overlap():
    soft = set(range(C.SOFT_PROMPT_BEGIN_IDX,
                     C.SOFT_PROMPT_BEGIN_IDX + C.NUM_SOFT_PROMPT_TOKENS))
    wrist = set(range(C.WRIST_PLACEHOLDER_BEGIN_IDX,
                      C.WRIST_PLACEHOLDER_BEGIN_IDX + C.NUM_WRIST_TOKENS))
    action = set(range(C.ACTION_TOKEN_BEGIN_IDX,
                       C.ACTION_TOKEN_BEGIN_IDX + C.NUM_ACTION_TOKENS))
    assert soft.isdisjoint(wrist)
    assert soft.isdisjoint(action)
    assert wrist.isdisjoint(action)


def test_image_soft_token_distinct():
    assert C.IMAGE_SOFT_TOKEN_ID not in range(
        C.SOFT_PROMPT_BEGIN_IDX,
        C.SOFT_PROMPT_BEGIN_IDX + C.NUM_SOFT_PROMPT_TOKENS,
    )
```

- [ ] **Step 2: Run, expect FAIL**

- [ ] **Step 3: Implement**

`src/vla_project/data/__init__.py`: empty.

`src/vla_project/data/constants.py`:

```python
"""Placeholder token IDs and Gemma4 metadata.

ID ranges are sub-slices of Gemma4's 6227 unused tokens (258884..262143)
and are kept disjoint so the input packer can identify each block by ID
membership alone. See docs/architectures/x_vla_adapter.md for layout.
"""

# === Gemma4 native image token (PaliGemma-style scene placeholder) ===
# tokenizer.convert_tokens_to_ids('<image_soft_token>') in Gemma4
IMAGE_SOFT_TOKEN_ID: int = 262144 - 1  # confirmed at runtime in Task 11

# === Action queries (carried from vla-gemma-4) ===
ACTION_TOKEN_BEGIN_IDX: int = 258885   # <unused2968>
NUM_ACTION_TOKENS: int = 64

# === Wrist patches ===
WRIST_PLACEHOLDER_BEGIN_IDX: int = 258949  # <unused3032>
NUM_WRIST_TOKENS: int = 256

# === Soft prompt ===
SOFT_PROMPT_BEGIN_IDX: int = 259461 + 1   # one past PROPRIO range from vla-gemma-4
NUM_SOFT_PROMPT_TOKENS: int = 32

# === Architecture-wide defaults (overridable in config) ===
LLM_HIDDEN_DIM: int = 1536
NUM_LLM_LAYERS: int = 35
PLE_DIM: int = 256

NUM_SCENE_TOKENS: int = 256
SIGLIP_HIDDEN_DIM: int = 1152
SIGLIP_IMAGE_SIZE: int = 224

DEFAULT_PROMPT_MAX_LEN: int = 50

ACTION_CHUNK_LEN: int = 8
ACTION_DIM: int = 7
PROPRIO_DIM: int = 8
```

- [ ] **Step 4: Run, expect PASS**

- [ ] **Step 5: Commit**

```bash
git add src/vla_project/data tests/test_constants.py
git commit -m "feat(data): add placeholder ID and Gemma4 metadata constants"
```

---

### Task 4: Internal Batch schema (TypedDict + light validation)

**Files:**
- Create: `src/vla_project/data/schema.py`
- Create: `tests/test_schema.py`

- [ ] **Step 1: Write failing tests**

`tests/test_schema.py`:

```python
import pytest
import torch

from vla_project.data.schema import Batch, validate_batch


def _make_batch(B=2):
    return Batch(
        domain_id=torch.zeros(B, dtype=torch.long),
        scene_image=torch.randn(B, 3, 224, 224),
        wrist_image=torch.randn(B, 3, 224, 224),
        prompt_input_ids=torch.zeros(B, 50, dtype=torch.long),
        prompt_attention_mask=torch.ones(B, 50, dtype=torch.long),
        proprio=torch.randn(B, 8),
        last_action_chunk=torch.randn(B, 8, 7),
        target_action=torch.randn(B, 8, 7),
        action_mask=torch.ones(B, 8, dtype=torch.bool),
    )


def test_validate_batch_accepts_valid():
    batch = _make_batch()
    validate_batch(batch)


def test_validate_batch_rejects_wrong_action_dim():
    batch = _make_batch()
    batch["target_action"] = torch.randn(2, 8, 5)
    with pytest.raises(AssertionError):
        validate_batch(batch)
```

- [ ] **Step 2: Run, expect FAIL**

- [ ] **Step 3: Implement**

`src/vla_project/data/schema.py`:

```python
from typing import TypedDict

import torch

from vla_project.data import constants as C


class Batch(TypedDict):
    domain_id: torch.Tensor          # [B] long
    scene_image: torch.Tensor        # [B, 3, H, W] float
    wrist_image: torch.Tensor        # [B, 3, H, W] float
    prompt_input_ids: torch.Tensor   # [B, Lt] long
    prompt_attention_mask: torch.Tensor  # [B, Lt] long
    proprio: torch.Tensor            # [B, D_prop] float
    last_action_chunk: torch.Tensor  # [B, T, A] float
    target_action: torch.Tensor      # [B, T, A] float
    action_mask: torch.Tensor        # [B, T] bool


def validate_batch(batch: Batch) -> None:
    B = batch["domain_id"].shape[0]
    assert batch["domain_id"].dtype == torch.long
    assert batch["domain_id"].shape == (B,)

    assert batch["scene_image"].shape[:2] == (B, 3)
    assert batch["scene_image"].shape[-2:] == (
        C.SIGLIP_IMAGE_SIZE, C.SIGLIP_IMAGE_SIZE,
    )
    assert batch["wrist_image"].shape == batch["scene_image"].shape

    assert batch["prompt_input_ids"].dtype == torch.long
    assert batch["prompt_attention_mask"].shape == batch["prompt_input_ids"].shape

    assert batch["proprio"].shape == (B, C.PROPRIO_DIM)

    T, A = C.ACTION_CHUNK_LEN, C.ACTION_DIM
    assert batch["last_action_chunk"].shape == (B, T, A)
    assert batch["target_action"].shape == (B, T, A)
    assert batch["action_mask"].shape == (B, T)
    assert batch["action_mask"].dtype == torch.bool
```

- [ ] **Step 4: Run, expect PASS**

- [ ] **Step 5: Commit**

```bash
git add src/vla_project/data/schema.py tests/test_schema.py
git commit -m "feat(data): typed Batch schema and runtime validator"
```

---

### Task 4.5: Centralized normalization (`data/normalization.py`)

CLAUDE.md mandates centralized normalization. Stage 1 smoke uses random tensors, but the module must exist so Stage 2+ has a canonical home (and so the per-domain normalization stats can be loaded into checkpoints).

**Files:**
- Create: `src/vla_project/data/normalization.py`
- Create: `tests/test_normalization.py`

- [ ] **Step 1: Failing tests**

```python
import torch

from vla_project.data.normalization import NormalizationStats, normalize, denormalize


def test_normalize_denormalize_roundtrip():
    stats = NormalizationStats(
        mean=torch.tensor([0.0, 1.0]),
        std=torch.tensor([2.0, 0.5]),
    )
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    y = normalize(x, stats)
    z = denormalize(y, stats)
    torch.testing.assert_close(x, z)


def test_zero_std_clamped():
    stats = NormalizationStats(
        mean=torch.tensor([0.0]),
        std=torch.tensor([0.0]),
    )
    x = torch.tensor([1.0])
    # must not div-by-zero
    y = normalize(x, stats)
    assert torch.isfinite(y).all()
```

- [ ] **Step 2: Run, expect FAIL**

- [ ] **Step 3: Implement**

```python
from dataclasses import dataclass

import torch


@dataclass
class NormalizationStats:
    mean: torch.Tensor   # [D]
    std: torch.Tensor    # [D]


def _safe_std(std: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return std.clamp_min(eps)


def normalize(x: torch.Tensor, stats: NormalizationStats) -> torch.Tensor:
    return (x - stats.mean) / _safe_std(stats.std)


def denormalize(x: torch.Tensor, stats: NormalizationStats) -> torch.Tensor:
    return x * _safe_std(stats.std) + stats.mean
```

- [ ] **Step 4: Run, expect PASS**

- [ ] **Step 5: Commit**

```bash
git add src/vla_project/data/normalization.py tests/test_normalization.py
git commit -m "feat(data): centralized action/proprio normalization helpers"
```

---

## Stage 2: Per-domain primitives

### Task 5: `DomainAwareLinear`

Reference: `X-VLA/models/transformer.py:226-250`. Per-domain weight matrix and bias via `nn.Embedding(num_dom, output*input)` and `nn.Embedding(num_dom, output)`.

**Files:**
- Create: `src/vla_project/models/__init__.py`
- Create: `src/vla_project/models/projectors/__init__.py`
- Create: `src/vla_project/models/projectors/domain_aware_linear.py`
- Create: `tests/test_domain_aware_linear.py`

- [ ] **Step 1: Write failing tests**

`tests/test_domain_aware_linear.py`:

```python
import torch

from vla_project.models.projectors.domain_aware_linear import DomainAwareLinear


def test_2d_input_shape():
    layer = DomainAwareLinear(input_size=8, output_size=16, num_domains=4)
    x = torch.randn(3, 8)
    domain_id = torch.tensor([0, 1, 3], dtype=torch.long)
    y = layer(x, domain_id)
    assert y.shape == (3, 16)


def test_3d_input_shape():
    layer = DomainAwareLinear(input_size=8, output_size=16, num_domains=4)
    x = torch.randn(3, 5, 8)
    domain_id = torch.tensor([0, 1, 3], dtype=torch.long)
    y = layer(x, domain_id)
    assert y.shape == (3, 5, 16)


def test_different_domains_yield_different_outputs():
    layer = DomainAwareLinear(input_size=4, output_size=4, num_domains=3)
    torch.nn.init.normal_(layer.fc.weight, std=0.5)
    torch.nn.init.normal_(layer.bias.weight, std=0.5)
    x = torch.randn(1, 4)
    y0 = layer(x, torch.tensor([0]))
    y1 = layer(x, torch.tensor([1]))
    assert not torch.allclose(y0, y1)
```

- [ ] **Step 2: Run, expect FAIL**

- [ ] **Step 3: Implement** (port of X-VLA `DomainAwareLinear`, with type hints + 3-D support)

`src/vla_project/models/__init__.py`: empty.

`src/vla_project/models/projectors/__init__.py`:
```python
from vla_project.models.projectors.domain_aware_linear import DomainAwareLinear

__all__ = ["DomainAwareLinear"]
```

`src/vla_project/models/projectors/domain_aware_linear.py`:

```python
import torch
import torch.nn as nn


class DomainAwareLinear(nn.Module):
    """Per-domain linear: y = x @ W[domain_id] + b[domain_id].

    Weights and biases are stored as `nn.Embedding` rows so that lookup is
    a single embedding gather. Adapted from X-VLA's `DomainAwareLinear`
    (X-VLA/models/transformer.py).
    """

    def __init__(self, input_size: int, output_size: int, num_domains: int) -> None:
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.num_domains = num_domains
        self.fc = nn.Embedding(num_domains, input_size * output_size)
        self.bias = nn.Embedding(num_domains, output_size)
        nn.init.normal_(self.fc.weight, std=(input_size ** -0.5))
        nn.init.zeros_(self.bias.weight)

    def forward(self, x: torch.Tensor, domain_id: torch.Tensor) -> torch.Tensor:
        squeeze_T = False
        if x.dim() == 2:
            x = x.unsqueeze(1)
            squeeze_T = True
        B = domain_id.shape[0]
        assert x.shape[0] == B, f"batch mismatch: x={x.shape[0]} vs dom={B}"
        W = self.fc(domain_id).view(B, self.input_size, self.output_size)
        b = self.bias(domain_id).view(B, 1, self.output_size)
        y = torch.matmul(x, W) + b
        if squeeze_T:
            y = y.squeeze(1)
        return y
```

- [ ] **Step 4: Run, expect PASS**

- [ ] **Step 5: Commit**

```bash
git add src/vla_project/models tests/test_domain_aware_linear.py
git commit -m "feat(models): DomainAwareLinear ported from X-VLA"
```

---

### Task 6: `SoftPromptHub`

`nn.Embedding(num_dom, Ks * D)` reshaped to `[B, Ks, D]` per domain.

**Files:**
- Create: `src/vla_project/models/projectors/soft_prompts.py`
- Create: `tests/test_soft_prompts.py`

Lives under `models/projectors/` per CLAUDE.md (per-domain learnable token hub = projection-like module).

- [ ] **Step 1: Write failing tests**

```python
import torch

from vla_project.models.projectors.soft_prompts import SoftPromptHub


def test_shape_and_per_domain_distinct():
    hub = SoftPromptHub(num_domains=3, num_tokens=4, hidden_dim=8)
    out0 = hub(torch.tensor([0, 0]))
    out1 = hub(torch.tensor([1, 2]))
    assert out0.shape == (2, 4, 8)
    assert out1.shape == (2, 4, 8)
    assert not torch.allclose(out0[0], out1[0])
```

- [ ] **Step 2: Run, expect FAIL**

- [ ] **Step 3: Implement**

```python
import torch
import torch.nn as nn


class SoftPromptHub(nn.Module):
    def __init__(self, num_domains: int, num_tokens: int, hidden_dim: int) -> None:
        super().__init__()
        self.num_domains = num_domains
        self.num_tokens = num_tokens
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(num_domains, num_tokens * hidden_dim)
        nn.init.normal_(self.embedding.weight, std=0.02)

    def forward(self, domain_id: torch.Tensor) -> torch.Tensor:
        B = domain_id.shape[0]
        return self.embedding(domain_id).view(B, self.num_tokens, self.hidden_dim)
```

- [ ] **Step 4: Run, expect PASS**

- [ ] **Step 5: Commit**

```bash
git add src/vla_project/models/projectors/soft_prompts.py tests/test_soft_prompts.py
git commit -m "feat(models): per-domain SoftPromptHub under projectors/"
```

---

### Task 7: `ActionQueryHub` (SHARED, not per-domain)

**Design choice:** action queries are *shared* across domains — a single learnable `nn.Parameter` of shape `[Q, D]` broadcast to `[B, Q, D]` in forward. Class is named "Hub" only for naming consistency with `SoftPromptHub`.

**Files:**
- Create: `src/vla_project/models/projectors/action_queries.py`
- Create: `tests/test_action_queries.py`

- [ ] **Step 1: Write failing tests**

```python
import torch

from vla_project.models.projectors.action_queries import ActionQueryHub


def test_shape():
    hub = ActionQueryHub(num_queries=64, hidden_dim=16)
    a = hub(2)
    assert a.shape == (2, 64, 16)


def test_broadcast_same_across_batch():
    """Shared queries: every batch entry sees the same [Q, D] tensor."""
    hub = ActionQueryHub(num_queries=4, hidden_dim=8)
    a = hub(3)
    assert torch.equal(a[0], a[1])
    assert torch.equal(a[0], a[2])
```

- [ ] **Step 2: Run, expect FAIL**

- [ ] **Step 3: Implement**

```python
import torch
import torch.nn as nn


class ActionQueryHub(nn.Module):
    """Shared learnable action queries (NOT per-domain).

    Design choice: action queries are shared across domains. The class is
    kept named "Hub" for naming consistency with SoftPromptHub, but it does
    not index by domain_id — it broadcasts a single [Q, D] parameter to
    [B, Q, D] in forward.
    """

    def __init__(self, num_queries: int, hidden_dim: int) -> None:
        super().__init__()
        self.num_queries = num_queries
        self.hidden_dim = hidden_dim
        self.queries = nn.Parameter(torch.zeros(num_queries, hidden_dim))
        nn.init.normal_(self.queries, std=0.02)

    def forward(self, batch_size: int) -> torch.Tensor:
        return self.queries.unsqueeze(0).expand(
            batch_size, self.num_queries, self.hidden_dim
        )
```

- [ ] **Step 4: Run, expect PASS**

- [ ] **Step 5: Commit**

```bash
git add src/vla_project/models/projectors/action_queries.py tests/test_action_queries.py
git commit -m "feat(models): shared ActionQueryHub (not per-domain)"
```

---

## Stage 3: Vision encoder

### Task 8: SigLIP wrapper (frozen, shared)

Use `transformers.SiglipVisionModel`. Outputs `[B, 256, 1152]` for 224x224 input.

**Files:**
- Create: `src/vla_project/models/vision/__init__.py`
- Create: `src/vla_project/models/vision/siglip.py`
- Create: `tests/test_siglip_wrapper.py`

- [ ] **Step 1: Write failing tests**

```python
import torch

from vla_project.models.vision.siglip import SigLIPEncoder


def test_output_shape_with_stub(monkeypatch):
    enc = SigLIPEncoder.__new__(SigLIPEncoder)
    enc.hidden_dim = 1152
    enc.num_tokens = 256
    enc._stub = True

    def fake_forward(self, pixel_values):
        B = pixel_values.shape[0]
        return torch.zeros(B, self.num_tokens, self.hidden_dim)

    monkeypatch.setattr(SigLIPEncoder, "forward", fake_forward)

    out = enc(torch.randn(4, 3, 224, 224))
    assert out.shape == (4, 256, 1152)


def test_frozen_by_default():
    # avoid hitting the network: stub the inner model
    enc = SigLIPEncoder(model_name=None, _skip_load=True)
    enc.model = torch.nn.Linear(3, 3)  # dummy module to check freezing
    enc.freeze()
    for p in enc.model.parameters():
        assert p.requires_grad is False
```

- [ ] **Step 2: Run, expect FAIL**

- [ ] **Step 3: Implement**

`src/vla_project/models/vision/__init__.py`: empty.

`src/vla_project/models/vision/siglip.py`:

```python
from typing import Optional

import torch
import torch.nn as nn


class SigLIPEncoder(nn.Module):
    """Wraps `transformers.SiglipVisionModel`. Always frozen.

    Forward returns the **last hidden state** (`[B, N, D_vis]`).
    """

    def __init__(
        self,
        model_name: Optional[str] = "google/siglip-so400m-patch14-224",
        hidden_dim: int = 1152,
        num_tokens: int = 256,
        _skip_load: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_tokens = num_tokens
        self.model: Optional[nn.Module] = None
        if not _skip_load:
            from transformers import SiglipVisionModel
            self.model = SiglipVisionModel.from_pretrained(model_name)
            self.freeze()

    def freeze(self) -> None:
        for p in self.parameters():
            p.requires_grad = False
        self.eval()

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        assert self.model is not None, (
            "SigLIPEncoder.forward called before model loaded "
            "(used _skip_load=True without overriding self.model?)"
        )
        out = self.model(pixel_values=pixel_values).last_hidden_state
        assert out.shape[1:] == (self.num_tokens, self.hidden_dim), (
            f"expected (B, {self.num_tokens}, {self.hidden_dim}), "
            f"got {tuple(out.shape)}"
        )
        return out
```

- [ ] **Step 4: Run, expect PASS** (without network: `pytest tests/test_siglip_wrapper.py -v`)

- [ ] **Step 5: Commit**

```bash
git add src/vla_project/models/vision tests/test_siglip_wrapper.py
git commit -m "feat(models): frozen SigLIP wrapper"
```

---

## Stage 4: Input packer

### Task 9: `InputPacker` — placeholder construction and index dict

Builds `input_ids` in the order:
`[BOS, SoftPrompt×Ks, Scene×256, prompt text..., Wrist×Nw, ActionQuery×Q, EOS]`

Returns an `idx` dict with positions of each block. **All downstream code uses these indices.**

Lives in `data/packing/` because the output is a data-side tensor (`input_ids`), not a learned representation. This keeps the CLAUDE.md "data ↔ model" boundary clean.

**Files:**
- Create: `src/vla_project/data/packing/__init__.py`
- Create: `src/vla_project/data/packing/input_packer.py`
- Create: `tests/test_input_packer.py`

- [ ] **Step 1: Write failing tests**

```python
import torch

from vla_project.data import constants as C
from vla_project.data.packing.input_packer import InputPacker, PackedIDs


def test_layout_and_indices():
    packer = InputPacker(
        bos_id=2, eos_id=1,
        prompt_max_len=10,
    )
    prompt_ids = torch.tensor([[42, 43, 44, 0, 0, 0, 0, 0, 0, 0]])
    prompt_mask = torch.tensor([[1, 1, 1, 0, 0, 0, 0, 0, 0, 0]])
    packed: PackedIDs = packer(prompt_ids, prompt_mask)
    L_expected = (
        1                              # BOS
        + C.NUM_SOFT_PROMPT_TOKENS
        + C.NUM_SCENE_TOKENS
        + 10                            # prompt
        + C.NUM_WRIST_TOKENS
        + C.NUM_ACTION_TOKENS
        + 1                              # EOS
    )
    assert packed.input_ids.shape == (1, L_expected)
    assert packed.input_ids[0, 0].item() == 2
    assert packed.input_ids[0, -1].item() == 1
    soft_idx = packed.idx["soft"][0]
    scene_idx = packed.idx["scene"][0]
    wrist_idx = packed.idx["wrist"][0]
    action_idx = packed.idx["action"][0]
    assert soft_idx.numel() == C.NUM_SOFT_PROMPT_TOKENS
    assert scene_idx.numel() == C.NUM_SCENE_TOKENS
    assert wrist_idx.numel() == C.NUM_WRIST_TOKENS
    assert action_idx.numel() == C.NUM_ACTION_TOKENS

    assert (packed.input_ids[0, soft_idx] >= C.SOFT_PROMPT_BEGIN_IDX).all()
    assert (packed.input_ids[0, scene_idx] == C.IMAGE_SOFT_TOKEN_ID).all()
    assert (packed.input_ids[0, wrist_idx] >= C.WRIST_PLACEHOLDER_BEGIN_IDX).all()
    assert (packed.input_ids[0, action_idx] >= C.ACTION_TOKEN_BEGIN_IDX).all()


def test_attention_mask_respects_prompt_padding():
    packer = InputPacker(bos_id=2, eos_id=1, prompt_max_len=4)
    prompt_ids = torch.tensor([[10, 11, 0, 0]])
    prompt_mask = torch.tensor([[1, 1, 0, 0]])
    packed = packer(prompt_ids, prompt_mask)
    # The two padded prompt positions must have attention_mask = 0
    prompt_start = packed.idx["prompt"][0][0].item()
    am = packed.attention_mask[0]
    assert am[prompt_start + 0].item() == 1
    assert am[prompt_start + 1].item() == 1
    assert am[prompt_start + 2].item() == 0
    assert am[prompt_start + 3].item() == 0
```

- [ ] **Step 2: Run, expect FAIL**

- [ ] **Step 3: Implement**

`src/vla_project/data/packing/__init__.py`:
```python
from vla_project.data.packing.input_packer import InputPacker, PackedIDs

__all__ = ["InputPacker", "PackedIDs"]
```

`src/vla_project/data/packing/input_packer.py`:

```python
from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn

from vla_project.data import constants as C


@dataclass
class PackedIDs:
    input_ids: torch.Tensor          # [B, L_total] long
    attention_mask: torch.Tensor     # [B, L_total] long
    idx: Dict[str, torch.Tensor]     # block name -> [B, K]


class InputPacker(nn.Module):
    """Constructs Gemma4 input_ids with placeholders + index dict.

    Layout (per sample):
        [BOS,
         SoftPrompt   x Ks   (range starting at SOFT_PROMPT_BEGIN_IDX),
         Scene        x Ns   (IMAGE_SOFT_TOKEN_ID repeated),
         prompt text  x Lt   (padded with 0),
         Wrist        x Nw   (range starting at WRIST_PLACEHOLDER_BEGIN_IDX),
         ActionQuery  x Q    (range starting at ACTION_TOKEN_BEGIN_IDX),
         EOS]

    No proprio in input_ids — it conditions only the action head.
    """

    def __init__(self, bos_id: int, eos_id: int, prompt_max_len: int) -> None:
        super().__init__()
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.prompt_max_len = prompt_max_len

        soft = torch.arange(C.SOFT_PROMPT_BEGIN_IDX,
                            C.SOFT_PROMPT_BEGIN_IDX + C.NUM_SOFT_PROMPT_TOKENS)
        scene = torch.full((C.NUM_SCENE_TOKENS,), C.IMAGE_SOFT_TOKEN_ID, dtype=torch.long)
        wrist = torch.arange(C.WRIST_PLACEHOLDER_BEGIN_IDX,
                             C.WRIST_PLACEHOLDER_BEGIN_IDX + C.NUM_WRIST_TOKENS)
        action = torch.arange(C.ACTION_TOKEN_BEGIN_IDX,
                              C.ACTION_TOKEN_BEGIN_IDX + C.NUM_ACTION_TOKENS)
        # Cached templates (registered as buffers so they move with .to(device))
        self.register_buffer("_soft", soft, persistent=False)
        self.register_buffer("_scene", scene, persistent=False)
        self.register_buffer("_wrist", wrist, persistent=False)
        self.register_buffer("_action", action, persistent=False)

    def forward(
        self,
        prompt_input_ids: torch.Tensor,        # [B, prompt_max_len] long
        prompt_attention_mask: torch.Tensor,   # [B, prompt_max_len] long
    ) -> PackedIDs:
        B = prompt_input_ids.shape[0]
        device = prompt_input_ids.device
        Lp = self.prompt_max_len
        assert prompt_input_ids.shape == (B, Lp)
        assert prompt_attention_mask.shape == (B, Lp)

        bos = torch.full((B, 1), self.bos_id, dtype=torch.long, device=device)
        eos = torch.full((B, 1), self.eos_id, dtype=torch.long, device=device)
        soft = self._soft.to(device).unsqueeze(0).expand(B, -1)
        scene = self._scene.to(device).unsqueeze(0).expand(B, -1)
        wrist = self._wrist.to(device).unsqueeze(0).expand(B, -1)
        action = self._action.to(device).unsqueeze(0).expand(B, -1)

        ids = torch.cat([bos, soft, scene, prompt_input_ids, wrist, action, eos], dim=1)

        # attention mask: 1 everywhere except prompt-padded positions
        ones = lambda n: torch.ones(B, n, dtype=torch.long, device=device)
        am = torch.cat(
            [
                ones(1),
                ones(soft.shape[1]),
                ones(scene.shape[1]),
                prompt_attention_mask,
                ones(wrist.shape[1]),
                ones(action.shape[1]),
                ones(1),
            ],
            dim=1,
        )

        # Indices
        cur = 1
        soft_idx = torch.arange(cur, cur + soft.shape[1], device=device).expand(B, -1)
        cur += soft.shape[1]
        scene_idx = torch.arange(cur, cur + scene.shape[1], device=device).expand(B, -1)
        cur += scene.shape[1]
        prompt_idx = torch.arange(cur, cur + Lp, device=device).expand(B, -1)
        cur += Lp
        wrist_idx = torch.arange(cur, cur + wrist.shape[1], device=device).expand(B, -1)
        cur += wrist.shape[1]
        action_idx = torch.arange(cur, cur + action.shape[1], device=device).expand(B, -1)
        cur += action.shape[1]
        # EOS not exposed

        idx: Dict[str, torch.Tensor] = {
            "soft": soft_idx,
            "scene": scene_idx,
            "prompt": prompt_idx,
            "wrist": wrist_idx,
            "action": action_idx,
        }

        return PackedIDs(input_ids=ids, attention_mask=am, idx=idx)
```

- [ ] **Step 4: Run, expect PASS**

- [ ] **Step 5: Commit**

```bash
git add src/vla_project/data/packing tests/test_input_packer.py
git commit -m "feat(data): InputPacker builds placeholder ids + index dict"
```

---

### Task 10: Embed-overwrite utility (`scatter_into_embeds`)

Pure function used by `Gemma4Wrapper` / `VLAPolicy` to clone `embed_tokens(input_ids)` and overwrite at index positions. Lives next to the LLM wrapper because it is an LLM-input shaping helper.

**Files:**
- Create: `src/vla_project/models/language/embed_overwrite.py`
- Create: `tests/test_inputs_embeds_overwrite.py`

- [ ] **Step 1: Write failing tests**

```python
import torch
from vla_project.models.language.embed_overwrite import scatter_into_embeds


def test_overwrite_replaces_at_indices_only():
    B, L, D = 2, 7, 4
    base = torch.zeros(B, L, D)
    new = torch.ones(B, 3, D)
    idx = torch.tensor([[1, 3, 5], [0, 2, 6]])
    out = scatter_into_embeds(base, idx, new)
    for b in range(B):
        for k, pos in enumerate(idx[b].tolist()):
            assert torch.equal(out[b, pos], new[b, k])
        zeros = torch.tensor([p for p in range(L) if p not in idx[b].tolist()])
        for pos in zeros.tolist():
            assert torch.equal(out[b, pos], torch.zeros(D))
```

- [ ] **Step 2: Run, expect FAIL**

- [ ] **Step 3: Implement**

```python
import torch


def scatter_into_embeds(
    embeds: torch.Tensor,    # [B, L, D]
    idx: torch.Tensor,       # [B, K] long
    new: torch.Tensor,       # [B, K, D]
) -> torch.Tensor:
    """Returns a clone of `embeds` with rows at `idx` replaced by `new`."""
    assert embeds.dim() == 3, embeds.shape
    assert idx.dim() == 2, idx.shape
    assert new.dim() == 3, new.shape
    assert embeds.shape[0] == idx.shape[0] == new.shape[0]
    assert idx.shape[1] == new.shape[1]
    assert embeds.shape[-1] == new.shape[-1]
    out = embeds.clone()
    bs = torch.arange(embeds.shape[0], device=embeds.device).unsqueeze(1).expand_as(idx)
    out[bs, idx] = new
    return out
```

- [ ] **Step 4: Run, expect PASS**

- [ ] **Step 5: Commit**

```bash
git add src/vla_project/models/language/embed_overwrite.py tests/test_inputs_embeds_overwrite.py
git commit -m "feat(models): scatter_into_embeds utility under language/"
```

---

## Stage 5: Gemma4 backbone wrapper

### Task 11: `Gemma4Wrapper`

Loads `Gemma4ForConditionalGeneration` (or `Gemma4TextModel` directly), pre-computes PLE in `no_grad`, and forwards with `inputs_embeds + per_layer_inputs`.

Reference: `/home/takakimaeda/vla-gemma-4/VLA-Adapter/prismatic/extern/hf/modeling_prismatic_gemma4.py:563-617`.

**Files:**
- Create: `src/vla_project/models/language/__init__.py`
- Create: `src/vla_project/models/language/gemma4_wrapper.py`
- Create: `tests/test_gemma4_wrapper.py`

- [ ] **Step 1: Write failing tests** (mock-based, no network)

```python
import torch
import torch.nn as nn

from vla_project.models.language.gemma4_wrapper import Gemma4Wrapper


class _StubText(nn.Module):
    def __init__(self, hidden=8, layers=4):
        super().__init__()
        self.layers = layers
        self.hidden = hidden
        self.embed = nn.Embedding(50, hidden)

    def get_per_layer_inputs(self, input_ids, **kwargs):
        B, L = input_ids.shape
        return torch.zeros(B, L, self.layers, 4)

    def embed_tokens(self, input_ids):
        return self.embed(input_ids)

    def forward(
        self,
        inputs_embeds=None,
        per_layer_inputs=None,
        attention_mask=None,
        position_ids=None,
        use_cache=False,
        output_hidden_states=True,
    ):
        B, L, D = inputs_embeds.shape
        hs = tuple(inputs_embeds + i for i in range(self.layers + 1))
        return type("Out", (), {"hidden_states": hs})()


def test_forward_returns_stacked_hidden_states():
    text = _StubText()
    wrapper = Gemma4Wrapper(model_name=None, _skip_load=True)
    wrapper.text_model = text
    wrapper.num_layers = text.layers

    B, L = 2, 5
    input_ids = torch.zeros(B, L, dtype=torch.long)
    am = torch.ones(B, L, dtype=torch.long)
    out = wrapper(input_ids, am)
    assert out.hidden_states.shape == (B, text.layers + 1, L, text.hidden)
    assert out.per_layer_inputs.shape == (B, L, text.layers, 4)
```

- [ ] **Step 2: Run, expect FAIL**

- [ ] **Step 3: Implement**

`src/vla_project/models/language/__init__.py`: empty.

`src/vla_project/models/language/gemma4_wrapper.py`:

```python
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class Gemma4Out:
    hidden_states: torch.Tensor   # [B, num_layers+1, L, D]
    per_layer_inputs: torch.Tensor  # [B, L, num_layers, ple_dim]


class Gemma4Wrapper(nn.Module):
    """Loads Gemma4-E2B `text_model` and runs forward with PLE precompute.

    Sequence of operations follows
    `vla-gemma-4/.../modeling_prismatic_gemma4.py:563-617`:

      1. `per_layer_inputs = get_per_layer_inputs(input_ids)` under `no_grad`
      2. caller computes `inputs_embeds` (clone + scatter_into_embeds)
      3. `text_model(inputs_embeds, per_layer_inputs, ...)` returns hidden_states tuple

    The wrapper exposes:
      - `embed_tokens(input_ids)` for the caller to obtain raw embeddings
      - `forward(inputs_embeds, per_layer_inputs, attention_mask)` returns Gemma4Out
    """

    def __init__(
        self,
        model_name: Optional[str] = "google/gemma-3n-E2B",  # placeholder; verify before training
        freeze: bool = True,
        _skip_load: bool = False,
    ) -> None:
        super().__init__()
        self.text_model: Optional[nn.Module] = None
        self.num_layers: int = 0
        if _skip_load:
            return
        if model_name is None:
            raise ValueError(
                "Gemma4Wrapper requires a model_name unless _skip_load=True is set"
            )
        from transformers import AutoModelForCausalLM
        full = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
        self.text_model = getattr(full, "model", full)
        self.num_layers = self.text_model.config.num_hidden_layers
        if freeze:
            for p in self.parameters():
                p.requires_grad = False
            self.text_model.eval()

    def embed_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
        assert self.text_model is not None, "Gemma4Wrapper not loaded"
        return self.text_model.embed_tokens(input_ids)

    def precompute_ple(self, input_ids: torch.Tensor) -> torch.Tensor:
        assert self.text_model is not None, "Gemma4Wrapper not loaded"
        with torch.no_grad():
            # Gemma4 / Gemma3n signature: get_per_layer_inputs(input_ids, **kwargs).
            # We pass only input_ids; second positional was previously `inputs_embeds`
            # in some HF revisions and is keyword-only in current ones.
            return self.text_model.get_per_layer_inputs(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        inputs_embeds: Optional[torch.Tensor] = None,
        per_layer_inputs: Optional[torch.Tensor] = None,
    ) -> Gemma4Out:
        if per_layer_inputs is None:
            per_layer_inputs = self.precompute_ple(input_ids)
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        B, L = input_ids.shape
        position_ids = torch.arange(L, device=input_ids.device).unsqueeze(0).expand(B, -1)

        out = self.text_model(
            inputs_embeds=inputs_embeds,
            per_layer_inputs=per_layer_inputs,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
            output_hidden_states=True,
        )
        hs = torch.stack(out.hidden_states, dim=1)  # [B, layers+1, L, D]
        return Gemma4Out(hidden_states=hs, per_layer_inputs=per_layer_inputs)
```

- [ ] **Step 4: Run, expect PASS**

- [ ] **Step 5: Commit**

```bash
git add src/vla_project/models/language tests/test_gemma4_wrapper.py
git commit -m "feat(models): Gemma4Wrapper with PLE precompute"
```

---

## Stage 6: Action head

### Task 12: RoPE utilities

Reference: `VLA-Adapter/prismatic/models/action_heads.py` `apply_rope` block (above `MLPResNetBlock_Pro`).

**Files:**
- Create: `src/vla_project/models/action_heads/__init__.py`
- Create: `src/vla_project/models/action_heads/rope.py`
- Create: `tests/test_rope.py`

- [ ] **Step 1: Failing test**

```python
import torch
from vla_project.models.action_heads.rope import RotaryEmbedding, apply_rope


def test_rope_shapes_and_does_not_change_q_norm():
    B, H, L, Dh = 2, 4, 6, 8
    q = torch.randn(B, H, L, Dh)
    k = torch.randn(B, H, L, Dh)
    rope = RotaryEmbedding(dim=Dh)
    cos, sin = rope(seq_len=L, device=q.device, dtype=q.dtype)
    qr, kr = apply_rope(q, k, cos, sin)
    assert qr.shape == q.shape
    assert kr.shape == k.shape
    # RoPE is norm-preserving along last dim
    torch.testing.assert_close(q.norm(dim=-1), qr.norm(dim=-1), atol=1e-4, rtol=1e-4)
```

- [ ] **Step 2: Run, expect FAIL**

- [ ] **Step 3: Implement** (port verbatim from `VLA-Adapter/prismatic/models/action_heads.py`)

```python
import torch
import torch.nn as nn


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, base: int = 10000) -> None:
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.dim = dim

    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        cos = emb.cos().to(dtype)
        sin = emb.sin().to(dtype)
        return cos, sin


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    q_rot = (q * cos) + (_rotate_half(q) * sin)
    k_rot = (k * cos) + (_rotate_half(k) * sin)
    return q_rot, k_rot
```

- [ ] **Step 4: Run, expect PASS**

- [ ] **Step 5: Commit**

```bash
git add src/vla_project/models/action_heads tests/test_rope.py
git commit -m "feat(action_heads): RoPE utilities"
```

---

### Task 13: `MLPResNetBlock_Pro`

Direct port of `VLA-Adapter/prismatic/models/action_heads.py:287-410`. Three attention branches (`self`, `adapter = h_a ⊕ p`, `task = h_t * ratio_g`).

**Files:**
- Create: `src/vla_project/models/action_heads/mlp_resnet_block_pro.py`
- Create: `tests/test_mlp_resnet_block_pro.py`

- [ ] **Step 1: Failing tests**

```python
import math
import torch
from vla_project.models.action_heads.mlp_resnet_block_pro import MLPResNetBlock_Pro


def test_forward_shape():
    B, T, D = 2, 8, 64
    Ka = 65   # h_a (64 action queries) + 1 (proprio) — caller concatenates
    Kt = 256  # task tokens
    blk = MLPResNetBlock_Pro(dim=D)
    x = torch.randn(B, T, D)
    h_a = torch.randn(B, 64, D)
    p = torch.randn(B, 1, D)
    h_t = torch.randn(B, Kt, D)
    out = blk(x, h_a=h_a, h_t=h_t, p=p)
    assert out.shape == (B, T, D)


def test_gating_init_is_zero():
    blk = MLPResNetBlock_Pro(dim=64)
    assert torch.equal(blk.gating_factor, torch.zeros(1))
```

- [ ] **Step 2: Run, expect FAIL**

- [ ] **Step 3: Implement** (verbatim port; quote `VLA-Adapter` source above each block in inline comment)

```python
import math

import torch
import torch.nn as nn

from vla_project.models.action_heads.rope import RotaryEmbedding, apply_rope


class MLPResNetBlock_Pro(nn.Module):
    """Direct port of VLA-Adapter MLPResNetBlock_Pro.

    Three attention branches:
      - self(x): weight 1
      - adapter(h_a concat p): weight 1
      - task(h_t): weight ratio_g = tanh(gating_factor)
    """

    def __init__(self, dim: int, num_heads: int = 8) -> None:
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q_proj = nn.Linear(dim, dim)
        self.k_self = nn.Linear(dim, dim)
        self.v_self = nn.Linear(dim, dim)
        self.k_adapter = nn.Linear(dim, dim)
        self.v_adapter = nn.Linear(dim, dim)
        self.k_task = nn.Linear(dim, dim)
        self.v_task = nn.Linear(dim, dim)
        self.o_proj = nn.Linear(dim, dim)

        self.ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.ReLU(),
        )

        self.gating_factor = nn.Parameter(torch.zeros(1))
        self.rope = RotaryEmbedding(dim=self.head_dim)

    def forward(
        self,
        x: torch.Tensor,
        h_a: torch.Tensor,
        h_t: torch.Tensor,
        p: torch.Tensor,
    ) -> torch.Tensor:
        ratio_g = torch.tanh(self.gating_factor)

        h_adapter = torch.cat([h_a, p], dim=1)
        h_task = h_t

        B, T, _ = x.shape
        K_a = h_adapter.shape[1]
        K_t = h_task.shape[1]

        def _heads(t: torch.Tensor, L: int) -> torch.Tensor:
            return t.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        q = _heads(self.q_proj(x), T)
        k_s = _heads(self.k_self(x), T)
        v_s = _heads(self.v_self(x), T)
        k_a = _heads(self.k_adapter(h_adapter), K_a)
        v_a = _heads(self.v_adapter(h_adapter), K_a)
        k_t = _heads(self.k_task(h_task), K_t)
        v_t = _heads(self.v_task(h_task), K_t)

        # RoPE on q+k_self
        cos, sin = self.rope(seq_len=T, device=x.device, dtype=x.dtype)
        q, k_s = apply_rope(q, k_s, cos, sin)
        cos_a, sin_a = self.rope(seq_len=K_a, device=x.device, dtype=x.dtype)
        _, k_a = apply_rope(k_a, k_a, cos_a, sin_a)
        cos_t, sin_t = self.rope(seq_len=K_t, device=x.device, dtype=x.dtype)
        _, k_t = apply_rope(k_t, k_t, cos_t, sin_t)

        scores = torch.cat(
            [
                torch.matmul(q, k_s.transpose(-2, -1)),
                torch.matmul(q, k_a.transpose(-2, -1)),
                torch.matmul(q, k_t.transpose(-2, -1)) * ratio_g,
            ],
            dim=-1,
        ) / math.sqrt(self.head_dim)
        weights = torch.softmax(scores, dim=-1)

        v = torch.cat([v_s, v_a, v_t], dim=2)
        out = torch.matmul(weights, v).transpose(1, 2).reshape(B, T, self.dim)
        out = self.o_proj(out)
        return self.ffn(out + x)
```

- [ ] **Step 4: Run, expect PASS**

- [ ] **Step 5: Commit**

```bash
git add src/vla_project/models/action_heads/mlp_resnet_block_pro.py tests/test_mlp_resnet_block_pro.py
git commit -m "feat(action_heads): MLPResNetBlock_Pro with task gating"
```

---

### Task 14: `MLPResNet` — 35-block stack

Reference: `VLA-Adapter/prismatic/models/action_heads.py:84-121`, but **`num_blocks=35`** for our design. Each block `i` consumes `h_t[:, i+1, :, :]` and `h_a[:, i+1, :, :]` from the LLM hidden states.

**Files:**
- Create: `src/vla_project/models/action_heads/mlp_resnet.py`
- Update: `tests/test_mlp_resnet_block_pro.py` to add a 2-block stack smoke test

- [ ] **Step 1: Failing test**

`tests/test_mlp_resnet.py`:

```python
import torch
from vla_project.models.action_heads.mlp_resnet import MLPResNet


def test_stack_forward_shape():
    B, T, D = 2, 8, 32
    L = 4   # 4-block stack
    K_t, K_a = 16, 8
    model = MLPResNet(num_blocks=L, hidden_dim=D, action_dim=7,
                      input_dim=D * 7, output_dim=7)
    x = torch.randn(B, T, D * 7)
    h_t = torch.randn(B, L + 1, K_t, D)
    h_a = torch.randn(B, L + 1, K_a, D)
    p = torch.randn(B, 1, D)
    y = model(x, h_a=h_a, h_t=h_t, p=p)
    assert y.shape == (B, T, 7)
```

- [ ] **Step 2: Run, expect FAIL**

- [ ] **Step 3: Implement**

```python
import torch
import torch.nn as nn

from vla_project.models.action_heads.mlp_resnet_block_pro import MLPResNetBlock_Pro


class MLPResNet(nn.Module):
    def __init__(
        self,
        num_blocks: int,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        action_dim: int,
    ) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.blocks = nn.ModuleList(
            [MLPResNetBlock_Pro(dim=hidden_dim) for _ in range(num_blocks)]
        )
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(
        self,
        x: torch.Tensor,            # [B, T, input_dim]
        h_a: torch.Tensor,          # [B, num_layers+1, K_a, D]
        h_t: torch.Tensor,          # [B, num_layers+1, K_t, D]
        p: torch.Tensor,            # [B, 1, D]
    ) -> torch.Tensor:
        x = self.relu(self.fc1(self.layer_norm1(x)))
        for i, blk in enumerate(self.blocks):
            x = blk(x, h_a=h_a[:, i + 1], h_t=h_t[:, i + 1], p=p)
        x = self.fc2(self.layer_norm2(x))
        return x
```

- [ ] **Step 4: Run, expect PASS**

- [ ] **Step 5: Commit**

```bash
git add src/vla_project/models/action_heads/mlp_resnet.py tests/test_mlp_resnet.py
git commit -m "feat(action_heads): MLPResNet stacks 35 Pro blocks"
```

---

### Task 15: `L1RegressionActionHead` — 35-block, last-action `x` init

Reference: `VLA-Adapter/prismatic/models/action_heads.py:21-82`, modified to:
- replace zero-initialized `cond_actions_hidden_states` with `last_action_chunk` projected via `LastActionProj`
- pass `proprio` through `proprio_projector` (existing arg) for `p`
- output via `DomainAwareLinear` action decoder (this lives in `VLAPolicy`, but the head's last layer outputs `[B, T, D]`)

**Files:**
- Create: `src/vla_project/models/action_heads/l1_regression_action_head.py`
- Create: `tests/test_l1_regression_action_head.py`

- [ ] **Step 1: Failing test**

```python
import torch
from vla_project.models.action_heads.l1_regression_action_head import L1RegressionActionHead


def test_predict_action_shape():
    B, T, D, A = 2, 8, 16, 7
    L = 3
    K_t = 12
    head = L1RegressionActionHead(
        hidden_dim=D, action_dim=A, num_action_chunks=T,
        num_blocks=L, num_task_tokens=K_t,
    )
    x_init = torch.randn(B, T, D)
    h_a = torch.randn(B, L + 1, 64, D)
    h_t = torch.randn(B, L + 1, K_t, D)
    p = torch.randn(B, 1, D)
    out = head(x_init, h_a=h_a, h_t=h_t, p=p)
    assert out.shape == (B, T, D)
```

- [ ] **Step 2: Run, expect FAIL**

- [ ] **Step 3: Implement**

```python
import torch
import torch.nn as nn

from vla_project.models.action_heads.mlp_resnet import MLPResNet


class L1RegressionActionHead(nn.Module):
    """Reduced from VLA-Adapter L1RegressionActionHead.

    Differences from VLA-Adapter original:
      - `x` is provided by caller (LastAction-projected sequence) — no zero init.
      - Output dim is the LLM hidden dim D, not action_dim. Final A-dim
        projection is done by the per-domain `action_decoder` in VLAPolicy.
    """

    def __init__(
        self,
        hidden_dim: int,
        action_dim: int,
        num_action_chunks: int,
        num_blocks: int,
        num_task_tokens: int,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.num_action_chunks = num_action_chunks
        self.num_task_tokens = num_task_tokens

        # The MLPResNet was originally fed [B, T, action_dim*hidden_dim]; we
        # keep the same input dim so the FC1 size matches the reference. The
        # caller reshapes `x` to [B, T, action_dim*hidden_dim] before passing.
        self.model = MLPResNet(
            num_blocks=num_blocks,
            input_dim=action_dim * hidden_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            action_dim=action_dim,
        )

    def forward(
        self,
        x: torch.Tensor,            # [B, T, A*D] — LastAction-projected and tiled
        h_a: torch.Tensor,          # [B, L+1, Q, D]
        h_t: torch.Tensor,          # [B, L+1, K_t, D]
        p: torch.Tensor,            # [B, 1, D]
    ) -> torch.Tensor:
        return self.model(x, h_a=h_a, h_t=h_t, p=p)
```

- [ ] **Step 4: Run, expect PASS**

- [ ] **Step 5: Commit**

```bash
git add src/vla_project/models/action_heads/l1_regression_action_head.py tests/test_l1_regression_action_head.py
git commit -m "feat(action_heads): L1RegressionActionHead with caller-supplied x init"
```

---

### Task 16: Gating-init invariance test (per-block `ratio_g = 0` at start)

**Files:**
- Update: `tests/test_mlp_resnet.py`

- [ ] **Step 1: Add test**

```python
def test_all_block_gates_init_zero():
    from vla_project.models.action_heads.mlp_resnet import MLPResNet
    m = MLPResNet(num_blocks=35, input_dim=7 * 16, hidden_dim=16,
                  output_dim=16, action_dim=7)
    for blk in m.blocks:
        assert torch.equal(blk.gating_factor, torch.zeros(1))
```

- [ ] **Step 2: Run, expect PASS**

- [ ] **Step 3: Commit**

```bash
git add tests/test_mlp_resnet.py
git commit -m "test(action_heads): assert all 35 task gates init to 0"
```

---

## Stage 7: Loss

### Task 17: `masked_l1` and `masked_huber`

**Files:**
- Create: `src/vla_project/training/__init__.py`
- Create: `src/vla_project/training/losses.py`
- Create: `tests/test_masked_loss.py`

- [ ] **Step 1: Failing tests**

```python
import torch
from vla_project.training.losses import masked_l1, masked_huber


def test_masked_l1_ignores_padded():
    pred = torch.tensor([[[1.0, 0.0]], [[0.0, 0.0]]])      # [2,1,2]
    targ = torch.tensor([[[0.0, 0.0]], [[10.0, 10.0]]])
    mask = torch.tensor([[True], [False]])
    loss = masked_l1(pred, targ, mask)
    # only first sample contributes; |1-0| + |0-0| over 2 elements -> 0.5
    torch.testing.assert_close(loss, torch.tensor(0.5))


def test_masked_huber_finite():
    pred = torch.randn(2, 8, 7)
    targ = torch.randn(2, 8, 7)
    mask = torch.ones(2, 8, dtype=torch.bool)
    loss = masked_huber(pred, targ, mask, beta=0.1)
    assert torch.isfinite(loss)
```

- [ ] **Step 2: Run, expect FAIL**

- [ ] **Step 3: Implement**

```python
import torch
import torch.nn.functional as F


def _expand_mask_f32(mask: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    """Broadcast mask to `like` shape; always float32 for safe accumulation
    even when `pred`/`target` are bf16 (bf16 sums underflow at small norms)."""
    return mask.unsqueeze(-1).expand_as(like).to(torch.float32)


def masked_l1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = _expand_mask_f32(mask, pred)
    diff = (pred - target).abs().to(torch.float32) * m
    return diff.sum() / m.sum().clamp_min(1.0)


def masked_huber(
    pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, beta: float = 0.1
) -> torch.Tensor:
    m = _expand_mask_f32(mask, pred)
    diff = F.smooth_l1_loss(pred, target, beta=beta, reduction="none").to(torch.float32) * m
    return diff.sum() / m.sum().clamp_min(1.0)
```

- [ ] **Step 4: Run, expect PASS**

- [ ] **Step 5: Commit**

```bash
git add src/vla_project/training tests/test_masked_loss.py
git commit -m "feat(training): masked L1 + Huber regression losses"
```

---

## Stage 8: Combined `VLAPolicy`

### Task 17.5: Shared test stubs (`tests/_stubs.py`)

Extract `_StubSig` and `_StubGemma` so multiple downstream tests can import them without depending on `tests/test_vla_policy_forward.py` (which would couple test discovery / renames).

**Files:**
- Create: `tests/_stubs.py`

- [ ] **Step 1: Implement**

```python
"""Shared lightweight stubs for VLAPolicy unit tests (no network, no real weights)."""
import torch
import torch.nn as nn


class _StubSig(nn.Module):
    """Mimics SigLIPEncoder output shape without loading weights."""

    def __init__(self):
        super().__init__()
        self.hidden_dim = 1152
        self.num_tokens = 256

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros(x.shape[0], self.num_tokens, self.hidden_dim)

    def freeze(self):
        pass


class _StubGemma(nn.Module):
    """Mimics Gemma4Wrapper API: embed_tokens / precompute_ple / forward.

    Hidden states are deterministic functions of input_ids so tests can verify
    that downstream code reads the expected positions.
    """

    num_layers = 4
    hidden_dim = 32

    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(300_000, self.hidden_dim)

    def embed_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed(input_ids)

    def precompute_ple(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, L = input_ids.shape
        return torch.zeros(B, L, self.num_layers, 4)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        inputs_embeds=None,
        per_layer_inputs=None,
    ):
        from vla_project.models.language.gemma4_wrapper import Gemma4Out
        if per_layer_inputs is None:
            per_layer_inputs = self.precompute_ple(input_ids)
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        # Each layer returns inputs_embeds + i so layer index is recoverable.
        hs = torch.stack([inputs_embeds + i for i in range(self.num_layers + 1)], dim=1)
        return Gemma4Out(hidden_states=hs, per_layer_inputs=per_layer_inputs)
```

- [ ] **Step 2: Commit**

```bash
git add tests/_stubs.py
git commit -m "test: shared _StubSig and _StubGemma fixtures"
```

---

### Task 18: `VLAPolicy.forward` (end-to-end glue)

Combines:
1. SigLIP encode (scene + wrist)
2. Domain-aware projections (scene_proj, wrist_proj)
3. Soft prompt + action query lookup
4. InputPacker → input_ids + idx
5. Gemma4Wrapper with overwrite at idx
6. Slice `h_t` (scene + prompt + wrist positions, layers 1..35) and `h_a` (action positions, layers 1..35)
7. `x = LastActionProj(last_action_chunk)`, tiled to `[B, T, A*D]`
8. `p = ProprioProj(proprio)`
9. `L1RegressionActionHead(x, h_a, h_t, p) -> [B, T, D]`
10. `ActionDecoder` (per-domain) → `[B, T, A]`
11. masked L1 loss

**Files:**
- Create: `src/vla_project/models/vla_policy.py`
- Create: `tests/test_vla_policy_forward.py`

- [ ] **Step 1: Failing test** (using stub Gemma + stub SigLIP, no real weights)

```python
import torch

from tests._stubs import _StubSig, _StubGemma
from vla_project.models.vla_policy import VLAPolicy, VLAPolicyConfig


def test_forward_shape_and_loss_finite():
    cfg = VLAPolicyConfig(
        num_domains=2, hidden_dim=32, action_dim=7, action_chunk_len=8, proprio_dim=8,
        prompt_max_len=10, num_blocks=4,
    )
    policy = VLAPolicy(cfg, vision_encoder=_StubSig(), gemma=_StubGemma())
    B = 2
    batch = dict(
        domain_id=torch.zeros(B, dtype=torch.long),
        scene_image=torch.randn(B, 3, 224, 224),
        wrist_image=torch.randn(B, 3, 224, 224),
        prompt_input_ids=torch.zeros(B, 10, dtype=torch.long),
        prompt_attention_mask=torch.ones(B, 10, dtype=torch.long),
        proprio=torch.randn(B, 8),
        last_action_chunk=torch.randn(B, 8, 7),
        target_action=torch.randn(B, 8, 7),
        action_mask=torch.ones(B, 8, dtype=torch.bool),
    )
    pred, loss = policy(batch)
    assert pred.shape == (B, 8, 7)
    assert torch.isfinite(loss)
```

- [ ] **Step 2: Run, expect FAIL**

- [ ] **Step 3: Implement**

```python
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn

from vla_project.data import constants as C
from vla_project.data.packing.input_packer import InputPacker
from vla_project.models.action_heads.l1_regression_action_head import L1RegressionActionHead
from vla_project.models.language.embed_overwrite import scatter_into_embeds
from vla_project.models.projectors.action_queries import ActionQueryHub
from vla_project.models.projectors.domain_aware_linear import DomainAwareLinear
from vla_project.models.projectors.soft_prompts import SoftPromptHub
from vla_project.training.losses import masked_l1, masked_huber


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
    bos_id: int = 2
    eos_id: int = 1
    loss_type: str = "l1"  # or "huber"
    huber_beta: float = 0.1


class VLAPolicy(nn.Module):
    def __init__(self, cfg: VLAPolicyConfig, vision_encoder: nn.Module, gemma: nn.Module) -> None:
        super().__init__()
        self.cfg = cfg
        self.vision_encoder = vision_encoder
        self.gemma = gemma

        D, A = cfg.hidden_dim, cfg.action_dim
        self.scene_proj = DomainAwareLinear(cfg.siglip_hidden_dim, D, cfg.num_domains)
        self.wrist_proj = DomainAwareLinear(cfg.siglip_hidden_dim, D, cfg.num_domains)
        self.proprio_proj = DomainAwareLinear(cfg.proprio_dim, D, cfg.num_domains)
        # NOTE: project last_action [B, T, A] -> [B, T, A*D] directly, so the
        # action-head MLPResNet's fc1 (input_dim = A*D) sees a non-redundant
        # representation per timestep. Earlier draft tiled a [B, T, D] vector
        # along a synthesized A axis, which collapses information.
        self.last_action_proj = DomainAwareLinear(A, A * D, cfg.num_domains)
        self.action_decoder = DomainAwareLinear(D, A, cfg.num_domains)

        self.soft_prompt_hub = SoftPromptHub(cfg.num_domains, cfg.num_soft_prompt_tokens, D)
        self.action_query_hub = ActionQueryHub(cfg.num_action_queries, D)  # shared, not per-domain

        self.input_packer = InputPacker(cfg.bos_id, cfg.eos_id, cfg.prompt_max_len)

        # action head expects per-step input dim = A * D
        # caller will tile last-action embeds into the [B, T, A*D] shape
        self.action_head = L1RegressionActionHead(
            hidden_dim=D,
            action_dim=A,
            num_action_chunks=cfg.action_chunk_len,
            num_blocks=cfg.num_blocks,
            num_task_tokens=C.NUM_SCENE_TOKENS + cfg.prompt_max_len + C.NUM_WRIST_TOKENS,
        )

    def _build_x(self, last_action: torch.Tensor, domain_id: torch.Tensor) -> torch.Tensor:
        """Project last_action [B, T, A] -> [B, T, A*D] directly via DomainAwareLinear.

        The head's MLPResNet.fc1 expects `A*D` per timestep. We project each
        timestep independently with the per-domain weight matrix; no tiling
        and no information collapse along the A axis.
        """
        B, T, A = last_action.shape
        D = self.cfg.hidden_dim
        flat = last_action.reshape(B * T, A)
        dom = domain_id.repeat_interleave(T)
        out = self.last_action_proj(flat, dom)  # [B*T, A*D]
        return out.view(B, T, A * D)

    def forward(self, batch: dict) -> Tuple[torch.Tensor, torch.Tensor]:
        cfg = self.cfg
        domain_id = batch["domain_id"]
        B = domain_id.shape[0]

        # 1. SigLIP encode (shared for both views)
        scene_tok = self.vision_encoder(batch["scene_image"])  # [B, 256, D_vis]
        wrist_tok = self.vision_encoder(batch["wrist_image"])

        # 2. Project to LLM dim, per domain
        scene_e = self.scene_proj(scene_tok, domain_id)        # [B, 256, D]
        wrist_e = self.wrist_proj(wrist_tok, domain_id)        # [B, 256, D]

        # 3. Soft prompts (per-domain) and action queries (shared, broadcast)
        soft_e = self.soft_prompt_hub(domain_id)
        action_q_e = self.action_query_hub(B)

        # 4. Build input_ids + indices
        packed = self.input_packer(batch["prompt_input_ids"], batch["prompt_attention_mask"])

        # 5. Gemma forward with overwrite
        raw_e = self.gemma.embed_tokens(packed.input_ids)
        emb = scatter_into_embeds(raw_e, packed.idx["soft"], soft_e)
        emb = scatter_into_embeds(emb, packed.idx["scene"], scene_e)
        emb = scatter_into_embeds(emb, packed.idx["wrist"], wrist_e)
        emb = scatter_into_embeds(emb, packed.idx["action"], action_q_e)

        out = self.gemma(
            input_ids=packed.input_ids,
            attention_mask=packed.attention_mask,
            inputs_embeds=emb,
        )
        hs = out.hidden_states  # [B, layers+1, L, D]

        # 6. Slice h_t (vision+text+wrist) and h_a (action)
        task_idx = torch.cat([packed.idx["scene"], packed.idx["prompt"], packed.idx["wrist"]], dim=1)
        bs = torch.arange(B, device=hs.device).view(B, 1, 1)
        layers = torch.arange(hs.shape[1], device=hs.device).view(1, hs.shape[1], 1)
        h_t = hs[bs, layers, task_idx.unsqueeze(1)]   # [B, layers+1, K_t, D]
        h_a = hs[bs, layers, packed.idx["action"].unsqueeze(1)]  # [B, layers+1, Q, D]

        # 7. x init from LastActionProj
        x_init = self._build_x(batch["last_action_chunk"], domain_id)

        # 8. proprio -> p
        p = self.proprio_proj(batch["proprio"], domain_id).unsqueeze(1)

        # 9. action head
        head_out = self.action_head(x_init, h_a=h_a, h_t=h_t, p=p)  # [B, T, D]

        # 10. action decoder
        pred = self.action_decoder(head_out, domain_id)              # [B, T, A]

        # 11. loss
        if cfg.loss_type == "l1":
            loss = masked_l1(pred, batch["target_action"], batch["action_mask"])
        elif cfg.loss_type == "huber":
            loss = masked_huber(pred, batch["target_action"], batch["action_mask"], beta=cfg.huber_beta)
        else:
            raise ValueError(f"unknown loss_type: {cfg.loss_type}")

        return pred, loss
```

- [ ] **Step 4: Run, expect PASS**

- [ ] **Step 5: Commit**

```bash
git add src/vla_project/models/vla_policy.py tests/test_vla_policy_forward.py
git commit -m "feat(models): VLAPolicy combines packer+gemma+head with per-domain projs"
```

---

### Task 19: Trainable parameter snapshot test (Stage 1 freeze policy)

**Files:**
- Create: `tests/test_trainable_parameters.py`

- [ ] **Step 1: Write test**

```python
import torch.nn as nn

from tests._stubs import _StubSig, _StubGemma
from vla_project.models.vla_policy import VLAPolicy, VLAPolicyConfig


def _freeze_module(m: nn.Module) -> None:
    for p in m.parameters():
        p.requires_grad = False


def test_stage1_freeze_policy_only_adapters_trainable():
    cfg = VLAPolicyConfig(num_domains=1, hidden_dim=32, num_blocks=4)
    policy = VLAPolicy(cfg, vision_encoder=_StubSig(), gemma=_StubGemma())
    _freeze_module(policy.vision_encoder)
    _freeze_module(policy.gemma)

    trainable_names = {n for n, p in policy.named_parameters() if p.requires_grad}
    # Should be: scene/wrist/proprio/last_action/action_decoder + soft + action_q + head
    expected_prefixes = (
        "scene_proj", "wrist_proj", "proprio_proj", "last_action_proj",
        "action_decoder", "soft_prompt_hub", "action_query_hub", "action_head",
    )
    for n in trainable_names:
        assert n.startswith(expected_prefixes), f"unexpected trainable: {n}"
    for prefix in expected_prefixes:
        assert any(n.startswith(prefix) for n in trainable_names), f"missing trainable: {prefix}"
```

- [ ] **Step 2: Run, expect PASS**

- [ ] **Step 3: Commit**

```bash
git add tests/test_trainable_parameters.py
git commit -m "test(models): Stage 1 freeze policy snapshot"
```

---

### Task 19.1: `test_ple_shape` — PLE wiring contract

Spec requires `per_layer_inputs.shape == [B, L_total, num_layers, ple_dim]`. Use the stub Gemma so the test runs offline.

**Files:**
- Create: `tests/test_ple_shape.py`

- [ ] **Step 1: Write test**

```python
import torch

from tests._stubs import _StubGemma


def test_per_layer_inputs_shape():
    gemma = _StubGemma()
    B, L = 2, 7
    input_ids = torch.zeros(B, L, dtype=torch.long)
    am = torch.ones(B, L, dtype=torch.long)
    out = gemma(input_ids, am)
    assert out.per_layer_inputs.shape[:2] == (B, L)
    assert out.per_layer_inputs.shape[2] == gemma.num_layers
```

- [ ] **Step 2: Run, expect PASS** — and **Step 3: Commit**

```bash
git add tests/test_ple_shape.py
git commit -m "test(models): PLE wiring shape contract"
```

---

### Task 19.2: `test_action_query_extraction` — `h_a` strictly from action positions

The spec requires that the head's `h_a` slice corresponds **only** to action-placeholder positions. Use `_StubGemma`'s deterministic `inputs_embeds + layer_idx` output and verify the values at `h_a[layer]` equal `inputs_embeds[layer, action_idx]`.

**Files:**
- Create: `tests/test_action_query_extraction.py`

- [ ] **Step 1: Write test**

```python
import torch

from tests._stubs import _StubSig, _StubGemma
from vla_project.models.vla_policy import VLAPolicy, VLAPolicyConfig


def test_h_a_comes_from_action_positions():
    cfg = VLAPolicyConfig(num_domains=1, hidden_dim=32, num_blocks=4, prompt_max_len=10)
    policy = VLAPolicy(cfg, _StubSig(), _StubGemma())

    B = 1
    batch = {
        "domain_id": torch.zeros(B, dtype=torch.long),
        "scene_image": torch.randn(B, 3, 224, 224),
        "wrist_image": torch.randn(B, 3, 224, 224),
        "prompt_input_ids": torch.zeros(B, 10, dtype=torch.long),
        "prompt_attention_mask": torch.ones(B, 10, dtype=torch.long),
        "proprio": torch.randn(B, 8),
        "last_action_chunk": torch.randn(B, 8, 7),
        "target_action": torch.randn(B, 8, 7),
        "action_mask": torch.ones(B, 8, dtype=torch.bool),
    }
    # Re-run the front of forward to reach hidden_states + indices.
    packed = policy.input_packer(batch["prompt_input_ids"], batch["prompt_attention_mask"])
    raw = policy.gemma.embed_tokens(packed.input_ids)
    out = policy.gemma(packed.input_ids, packed.attention_mask, inputs_embeds=raw)
    # In _StubGemma: hs[i] = raw + i; so hs at action_idx for layer i
    # equals raw[..., action_idx, :] + i.
    expected_layer1 = raw[:, packed.idx["action"][0]] + 1
    actual_layer1 = out.hidden_states[:, 1, packed.idx["action"][0]]
    assert torch.allclose(actual_layer1, expected_layer1)
```

- [ ] **Step 2: Run, expect PASS** — **Step 3: Commit**

```bash
git add tests/test_action_query_extraction.py
git commit -m "test(models): h_a sliced strictly from action placeholder positions"
```

---

### Task 19.3: `test_action_loss_mask_grad` — masked timesteps yield zero gradient

Spec rule: padded action timesteps must not contribute to gradient. Verify by retaining `pred_action.grad` and asserting it's zero on masked rows.

**Files:**
- Create: `tests/test_action_loss_mask_grad.py`

- [ ] **Step 1: Write test**

```python
import torch

from vla_project.training.losses import masked_l1


def test_masked_positions_have_zero_grad():
    pred = torch.randn(2, 4, 3, requires_grad=True)
    target = torch.randn(2, 4, 3)
    mask = torch.tensor([[True, True, False, False],
                         [True, False, False, False]])
    loss = masked_l1(pred, target, mask)
    loss.backward()
    # Masked rows must have grad == 0
    assert torch.equal(pred.grad[0, 2:], torch.zeros(2, 3))
    assert torch.equal(pred.grad[1, 1:], torch.zeros(3, 3))
    # Unmasked rows must have non-zero grad somewhere
    assert pred.grad[0, :2].abs().sum() > 0
```

- [ ] **Step 2: Run, expect PASS** — **Step 3: Commit**

```bash
git add tests/test_action_loss_mask_grad.py
git commit -m "test(training): padded timesteps yield zero gradient under masked_l1"
```

---

### Task 19.4: `test_domain_aware_swap_full` — swap `domain_id` end-to-end

Verify that flipping `domain_id` between two distinct domains produces different `pred_action` outputs from the full forward.

**Files:**
- Create: `tests/test_domain_aware_swap_full.py`

- [ ] **Step 1: Write test**

```python
import torch

from tests._stubs import _StubSig, _StubGemma
from vla_project.models.vla_policy import VLAPolicy, VLAPolicyConfig


def test_pred_action_differs_per_domain():
    cfg = VLAPolicyConfig(num_domains=2, hidden_dim=32, num_blocks=4, prompt_max_len=10)
    policy = VLAPolicy(cfg, _StubSig(), _StubGemma())
    # ensure per-domain weights are actually distinct
    for proj in [policy.scene_proj, policy.wrist_proj, policy.proprio_proj,
                 policy.last_action_proj, policy.action_decoder]:
        torch.nn.init.normal_(proj.fc.weight, std=0.5)

    B = 1
    common = dict(
        scene_image=torch.randn(B, 3, 224, 224),
        wrist_image=torch.randn(B, 3, 224, 224),
        prompt_input_ids=torch.zeros(B, 10, dtype=torch.long),
        prompt_attention_mask=torch.ones(B, 10, dtype=torch.long),
        proprio=torch.randn(B, 8),
        last_action_chunk=torch.randn(B, 8, 7),
        target_action=torch.randn(B, 8, 7),
        action_mask=torch.ones(B, 8, dtype=torch.bool),
    )
    pred0, _ = policy({**common, "domain_id": torch.zeros(B, dtype=torch.long)})
    pred1, _ = policy({**common, "domain_id": torch.ones(B, dtype=torch.long)})
    assert not torch.allclose(pred0, pred1)
```

- [ ] **Step 2: Run, expect PASS** — **Step 3: Commit**

```bash
git add tests/test_domain_aware_swap_full.py
git commit -m "test(models): per-domain pred_action divergence end-to-end"
```

---

## Stage 9: LIBERO data layer

### Task 20: Image transform (`SigLIP-aware resize + normalize`)

**Files:**
- Create: `src/vla_project/data/transforms/__init__.py`
- Create: `src/vla_project/data/transforms/image.py`
- Create: `tests/test_image_transform.py`

- [ ] **Step 1: Failing test**

```python
import torch
from vla_project.data.transforms.image import SiglipImageTransform


def test_resize_and_normalize_shape():
    t = SiglipImageTransform(size=224, training=False)
    img = torch.zeros(3, 100, 100)
    out = t(img)
    assert out.shape == (3, 224, 224)
    assert out.dtype == torch.float32
```

- [ ] **Step 2: Run, expect FAIL**

- [ ] **Step 3: Implement**

```python
import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode


class SiglipImageTransform(nn.Module):
    """SigLIP-So400m expects 224x224, normalized by SigLIP statistics."""

    MEAN = (0.5, 0.5, 0.5)
    STD = (0.5, 0.5, 0.5)

    def __init__(self, size: int = 224, training: bool = False) -> None:
        super().__init__()
        ops = [T.Resize((size, size), interpolation=InterpolationMode.BICUBIC)]
        if training:
            ops.append(T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.0))
        ops.append(T.Normalize(self.MEAN, self.STD))
        self.transform = T.Compose(ops)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        return self.transform(img.float())
```

- [ ] **Step 4: Run, expect PASS**

- [ ] **Step 5: Commit**

```bash
git add src/vla_project/data/transforms tests/test_image_transform.py
git commit -m "feat(data): SigLIP image transform"
```

---

### Task 21: Action transform (delta / abs / gripper)

Reference: `X-VLA/datasets/utils.py:90-108` (`action_slice`).

**Files:**
- Create: `src/vla_project/data/transforms/action.py`
- Create: `tests/test_action_transform.py`

- [ ] **Step 1: Failing test**

```python
import torch
from vla_project.data.transforms.action import action_slice


def test_action_slice_delta_indices():
    abs_traj = torch.tensor([[1.0, 2.0], [4.0, 6.0], [5.0, 7.0]])  # H=2, D=2
    out = action_slice(abs_traj, idx_for_delta=[0])
    assert torch.equal(out["proprio"], torch.tensor([1.0, 2.0]))
    assert out["action"].shape == (2, 2)
    assert out["action"][0, 0].item() == 4.0 - 1.0
```

- [ ] **Step 2: Run, expect FAIL**

- [ ] **Step 3: Implement** (port from X-VLA `utils.py`)

```python
from typing import Dict, Sequence

import torch


def action_slice(
    abs_traj: torch.Tensor,
    idx_for_delta: Sequence[int] = (),
    idx_for_mask_proprio: Sequence[int] = (),
) -> Dict[str, torch.Tensor]:
    if abs_traj.dim() != 2 or abs_traj.size(0) < 2:
        raise ValueError("abs_traj must be [H+1, D] with H>=1")

    proprio = abs_traj[0].clone()
    action = abs_traj[1:].clone()

    if idx_for_delta:
        idx = torch.as_tensor(list(idx_for_delta), dtype=torch.long)
        action[:, idx] -= proprio[idx]
    if idx_for_mask_proprio:
        idx = torch.as_tensor(list(idx_for_mask_proprio), dtype=torch.long)
        proprio[idx] = 0.0
    return {"proprio": proprio, "action": action}
```

- [ ] **Step 4: Run, expect PASS**

- [ ] **Step 5: Commit**

```bash
git add src/vla_project/data/transforms/action.py tests/test_action_transform.py
git commit -m "feat(data): action_slice helper from X-VLA"
```

---

### Task 22: LIBERO step-level dataset (smoke version with synthetic data)

For Stage 1 smoke we don't need real LIBERO HDF5/RLDS plumbing yet — provide a `SyntheticLIBEROBatchDataset` that yields random tensors of correct shape and `domain_id=0`. This lets the trainer wire up. A later task can add the real LIBERO reader.

**Files:**
- Create: `src/vla_project/data/datasets/__init__.py`
- Create: `src/vla_project/data/datasets/libero_dataset.py`
- Create: `tests/test_libero_dataset.py`

- [ ] **Step 1: Failing test**

```python
import torch
from torch.utils.data import DataLoader
from vla_project.data.datasets.libero_dataset import SyntheticLIBEROBatchDataset
from vla_project.data.schema import validate_batch


def test_yields_valid_batch():
    ds = SyntheticLIBEROBatchDataset(length=8, prompt_max_len=10)
    dl = DataLoader(ds, batch_size=2, collate_fn=ds.collate_fn)
    batch = next(iter(dl))
    validate_batch(batch)
    assert batch["domain_id"].shape[0] == 2
```

- [ ] **Step 2: Run, expect FAIL**

- [ ] **Step 3: Implement**

```python
import torch
from torch.utils.data import Dataset

from vla_project.data import constants as C


class SyntheticLIBEROBatchDataset(Dataset):
    """Yields random tensors that match the internal Batch schema.

    Used for smoke tests until the real LIBERO reader is hooked up.
    """

    def __init__(self, length: int = 64, prompt_max_len: int = C.DEFAULT_PROMPT_MAX_LEN):
        self.length = length
        self.prompt_max_len = prompt_max_len

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return {
            "domain_id": torch.tensor(0, dtype=torch.long),
            "scene_image": torch.randn(3, C.SIGLIP_IMAGE_SIZE, C.SIGLIP_IMAGE_SIZE),
            "wrist_image": torch.randn(3, C.SIGLIP_IMAGE_SIZE, C.SIGLIP_IMAGE_SIZE),
            "prompt_input_ids": torch.zeros(self.prompt_max_len, dtype=torch.long),
            "prompt_attention_mask": torch.zeros(self.prompt_max_len, dtype=torch.long),
            "proprio": torch.randn(C.PROPRIO_DIM),
            "last_action_chunk": torch.randn(C.ACTION_CHUNK_LEN, C.ACTION_DIM),
            "target_action": torch.randn(C.ACTION_CHUNK_LEN, C.ACTION_DIM),
            "action_mask": torch.ones(C.ACTION_CHUNK_LEN, dtype=torch.bool),
        }

    @staticmethod
    def collate_fn(samples):
        keys = samples[0].keys()
        return {k: torch.stack([s[k] for s in samples]) for k in keys}
```

- [ ] **Step 4: Run, expect PASS**

- [ ] **Step 5: Commit**

```bash
git add src/vla_project/data/datasets tests/test_libero_dataset.py
git commit -m "feat(data): synthetic LIBERO dataset for smoke wiring"
```

---

## Stage 10: Optimizer / Trainer / smoke train

### Task 22.5: LR scheduler stub (`training/schedulers.py`)

CLAUDE.md prescribes `training/schedulers.py`. Provide linear-warmup → cosine-decay (matching X-VLA `linear_warmup_cosine`). Used in Stage 2+ training; for Stage 1 smoke, only the unit test runs.

**Files:**
- Create: `src/vla_project/training/schedulers.py`
- Create: `tests/test_schedulers.py`

- [ ] **Step 1: Failing tests**

```python
from vla_project.training.schedulers import linear_warmup_cosine


def test_warmup_then_decay():
    base_lr = 1.0
    total = 100
    warmup = 10
    # at step 0: ~0
    assert linear_warmup_cosine(0, freeze_steps=0, warmup_steps=warmup,
                                 total_steps=total, base_lr=base_lr,
                                 min_lr_ratio=0.1) == 0.0
    # at warmup boundary: ~base_lr
    lr_warm = linear_warmup_cosine(warmup, 0, warmup, total, base_lr, 0.1)
    assert abs(lr_warm - base_lr) < 1e-6
    # at end: min_lr
    lr_end = linear_warmup_cosine(total, 0, warmup, total, base_lr, 0.1)
    assert abs(lr_end - 0.1 * base_lr) < 1e-6
```

- [ ] **Step 2: Run, expect FAIL**

- [ ] **Step 3: Implement**

```python
import math


def linear_warmup_cosine(
    step: int,
    freeze_steps: int,
    warmup_steps: int,
    total_steps: int,
    base_lr: float,
    min_lr_ratio: float,
) -> float:
    """Linear warmup over `warmup_steps`, then cosine decay to `min_lr_ratio * base_lr`.

    `freeze_steps` are steps before training starts (LR=0).
    """
    if step < freeze_steps:
        return 0.0
    s = step - freeze_steps
    if s < warmup_steps:
        return base_lr * (s / max(warmup_steps, 1))
    progress = (s - warmup_steps) / max(total_steps - warmup_steps, 1)
    progress = min(progress, 1.0)
    cos = 0.5 * (1.0 + math.cos(math.pi * progress))
    return base_lr * (min_lr_ratio + (1.0 - min_lr_ratio) * cos)
```

- [ ] **Step 4: Run, expect PASS** — **Step 5: Commit**

```bash
git add src/vla_project/training/schedulers.py tests/test_schedulers.py
git commit -m "feat(training): linear-warmup + cosine-decay scheduler"
```

---

### Task 23: Per-group optimizer builder

**Files:**
- Create: `src/vla_project/training/optim.py`
- Create: `tests/test_optim_groups.py`

- [ ] **Step 1: Failing test**

```python
import torch.nn as nn
from vla_project.training.optim import build_optimizer

# reuse VLAPolicy + stubs
from tests._stubs import _StubSig, _StubGemma
from vla_project.models.vla_policy import VLAPolicy, VLAPolicyConfig


def test_param_groups_present_and_no_frozen_group():
    cfg = VLAPolicyConfig(num_domains=1, hidden_dim=32, num_blocks=4)
    policy = VLAPolicy(cfg, _StubSig(), _StubGemma())
    # simulate Stage 1 freeze policy
    for p in policy.vision_encoder.parameters():
        p.requires_grad = False
    for p in policy.gemma.parameters():
        p.requires_grad = False
    optim = build_optimizer(policy, lr=1e-4, soft_lr_coef=2.0, weight_decay=0.01)
    names = {g["name"] for g in optim.param_groups}
    assert {"soft_prompts", "action_queries", "domain_projs", "action_head"} <= names
    assert "vlm_frozen" not in names
```

- [ ] **Step 2: Run, expect FAIL**

- [ ] **Step 3: Implement**

```python
import torch
import torch.nn as nn

from vla_project.models.vla_policy import VLAPolicy


def _trainable(params):
    return [p for p in params if p.requires_grad]


def build_optimizer(model: VLAPolicy, lr: float, soft_lr_coef: float, weight_decay: float):
    """Build AdamW with per-group LRs. Frozen params (SigLIP, Gemma in Stage 1)
    are *excluded* — not added with lr=0 — so AdamW does not allocate momentum
    state for them.
    """
    soft = _trainable(model.soft_prompt_hub.parameters())
    aq = _trainable(model.action_query_hub.parameters())
    head = _trainable(model.action_head.parameters())
    domain_projs = _trainable(
        list(model.scene_proj.parameters())
        + list(model.wrist_proj.parameters())
        + list(model.proprio_proj.parameters())
        + list(model.last_action_proj.parameters())
        + list(model.action_decoder.parameters())
    )

    groups = [
        {"name": "soft_prompts", "params": soft, "lr": lr * soft_lr_coef, "weight_decay": weight_decay},
        {"name": "action_queries", "params": aq, "lr": lr, "weight_decay": weight_decay},
        {"name": "domain_projs", "params": domain_projs, "lr": lr, "weight_decay": weight_decay},
        {"name": "action_head", "params": head, "lr": lr, "weight_decay": weight_decay},
    ]
    # filter out empty groups (defensive, e.g. if VLAPolicy has no soft prompts)
    groups = [g for g in groups if g["params"]]
    return torch.optim.AdamW(groups, betas=(0.9, 0.95))
```

- [ ] **Step 4: Run, expect PASS**

- [ ] **Step 5: Commit**

```bash
git add src/vla_project/training/optim.py tests/test_optim_groups.py
git commit -m "feat(training): per-group AdamW builder"
```

---

### Task 24: `Trainer` (Accelerate-based)

Minimum viable training loop: dataloader iter, forward, backward, optim step, log.

**Files:**
- Create: `src/vla_project/training/trainer.py`
- Create: `tests/test_trainer_one_step.py`

- [ ] **Step 1: Failing test**

```python
import torch
from torch.utils.data import DataLoader

from tests._stubs import _StubSig, _StubGemma
from vla_project.models.vla_policy import VLAPolicy, VLAPolicyConfig
from vla_project.data.datasets.libero_dataset import SyntheticLIBEROBatchDataset
from vla_project.training.optim import build_optimizer
from vla_project.training.trainer import Trainer, TrainerConfig


def test_one_training_step_decreases_or_holds_loss():
    cfg = VLAPolicyConfig(num_domains=1, hidden_dim=32, num_blocks=4, prompt_max_len=10)
    policy = VLAPolicy(cfg, _StubSig(), _StubGemma())
    ds = SyntheticLIBEROBatchDataset(length=4, prompt_max_len=10)
    dl = DataLoader(ds, batch_size=2, collate_fn=ds.collate_fn)
    optim = build_optimizer(policy, lr=1e-4, soft_lr_coef=1.0, weight_decay=0.0)
    trainer = Trainer(policy, optim, TrainerConfig(max_steps=2))
    losses = trainer.fit(dl)
    assert all(torch.isfinite(torch.tensor(l)) for l in losses)
    assert len(losses) == 2
```

- [ ] **Step 2: Run, expect FAIL**

- [ ] **Step 3: Implement**

```python
from dataclasses import dataclass
from typing import Iterable, List

import torch
import torch.nn as nn


@dataclass
class TrainerConfig:
    max_steps: int = 100
    log_every: int = 10
    grad_clip_norm: float = 1.0


class Trainer:
    def __init__(self, model: nn.Module, optimizer, cfg: TrainerConfig) -> None:
        self.model = model
        self.optimizer = optimizer
        self.cfg = cfg

    def fit(self, dataloader: Iterable) -> List[float]:
        """Train for exactly `max_steps` optimizer steps, re-iterating the
        dataloader as needed. Calling `iter(dataloader)` again starts a new
        epoch (fresh shuffling for map-style datasets, restart for iterable).
        """
        self.model.train()
        losses: List[float] = []
        step = 0
        while step < self.cfg.max_steps:
            for batch in dataloader:
                self.optimizer.zero_grad()
                _, loss = self.model(batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip_norm)
                self.optimizer.step()
                losses.append(loss.item())
                step += 1
                if step >= self.cfg.max_steps:
                    break
        return losses
```

(`accelerate` integration can be added as a follow-up — keep this trainer single-process for now.)

- [ ] **Step 4: Run, expect PASS**

- [ ] **Step 5: Commit**

```bash
git add src/vla_project/training/trainer.py tests/test_trainer_one_step.py
git commit -m "feat(training): minimal Trainer with grad clip + per-step logging"
```

---

### Task 25: `scripts/train.py` (thin entrypoint)

**Files:**
- Create: `scripts/train.py`

No test needed — entrypoint exercised by smoke test in Task 29.

- [ ] **Step 1: Implement**

```python
"""Thin training entrypoint. Heavy lifting lives in vla_project.training.trainer."""
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from vla_project.data.datasets.libero_dataset import SyntheticLIBEROBatchDataset
from vla_project.models.language.gemma4_wrapper import Gemma4Wrapper
from vla_project.models.vision.siglip import SigLIPEncoder
from vla_project.models.vla_policy import VLAPolicy, VLAPolicyConfig
from vla_project.training.optim import build_optimizer
from vla_project.training.trainer import Trainer, TrainerConfig
from vla_project.utils.seed import set_seed


def main(cfg_path: str) -> None:
    cfg = OmegaConf.load(cfg_path)
    set_seed(cfg.seed)

    policy_cfg = VLAPolicyConfig(**cfg.model)
    vision = SigLIPEncoder(model_name=cfg.vision.model_name)
    gemma = Gemma4Wrapper(model_name=cfg.language.model_name, freeze=True)
    policy = VLAPolicy(policy_cfg, vision, gemma)

    ds = SyntheticLIBEROBatchDataset(length=cfg.data.length, prompt_max_len=policy_cfg.prompt_max_len)
    dl = DataLoader(ds, batch_size=cfg.train.batch_size, collate_fn=ds.collate_fn)

    optim = build_optimizer(
        policy, lr=cfg.train.lr,
        soft_lr_coef=cfg.train.soft_lr_coef, weight_decay=cfg.train.weight_decay,
    )
    trainer = Trainer(policy, optim, TrainerConfig(max_steps=cfg.train.max_steps))
    trainer.fit(dl)


if __name__ == "__main__":
    import sys
    main(sys.argv[1])
```

- [ ] **Step 2: Commit**

```bash
git add scripts/train.py
git commit -m "feat(scripts): thin train.py entrypoint"
```

---

## Stage 11: Configs

### Task 26-28: Author config files

**Files:**
- Create: `configs/model/x_vla_adapter.yaml`
- Create: `configs/data/libero.yaml`
- Create: `configs/train/smoke.yaml`

- [ ] **Step 1: `configs/model/x_vla_adapter.yaml`**

```yaml
num_domains: 1
hidden_dim: 1536
siglip_hidden_dim: 1152
action_dim: 7
action_chunk_len: 8
proprio_dim: 8
prompt_max_len: 50
num_blocks: 35
num_soft_prompt_tokens: 32
num_action_queries: 64
loss_type: l1
huber_beta: 0.1
```

- [ ] **Step 2: `configs/data/libero.yaml`**

```yaml
type: libero_synthetic
length: 16
```

- [ ] **Step 3: `configs/train/smoke.yaml`**

```yaml
seed: 0
model:
  num_domains: 1
  hidden_dim: 1536
  num_blocks: 35
vision:
  model_name: google/siglip-so400m-patch14-224
language:
  model_name: google/gemma-3n-E2B   # placeholder; substitute the real Gemma4 ID
data:
  length: 16
train:
  batch_size: 1
  lr: 1.0e-4
  soft_lr_coef: 1.0
  weight_decay: 0.01
  max_steps: 2
```

- [ ] **Step 4: Commit**

```bash
git add configs
git commit -m "feat(configs): smoke configs for x_vla_adapter"
```

---

## Stage 12: End-to-end smoke

### Task 29: `tests/test_one_batch_smoke.py`

Run forward + backward on the synthetic dataloader using `_StubSig` + `_StubGemma` to avoid HF model downloads. Verify:
- `pred.shape == [B, T, A]`
- `loss` is a scalar finite tensor
- backward populates gradients on all trainable params
- frozen params have `grad is None`

**Files:**
- Create: `tests/test_one_batch_smoke.py`

- [ ] **Step 1: Write test**

```python
import torch
from torch.utils.data import DataLoader

from tests._stubs import _StubSig, _StubGemma
from vla_project.models.vla_policy import VLAPolicy, VLAPolicyConfig
from vla_project.data.datasets.libero_dataset import SyntheticLIBEROBatchDataset


def test_full_forward_backward():
    cfg = VLAPolicyConfig(num_domains=1, hidden_dim=32, num_blocks=4, prompt_max_len=10)
    policy = VLAPolicy(cfg, _StubSig(), _StubGemma())
    for p in policy.vision_encoder.parameters():
        p.requires_grad = False
    for p in policy.gemma.parameters():
        p.requires_grad = False

    ds = SyntheticLIBEROBatchDataset(length=2, prompt_max_len=10)
    dl = DataLoader(ds, batch_size=2, collate_fn=ds.collate_fn)
    batch = next(iter(dl))

    pred, loss = policy(batch)
    assert pred.shape == (2, 8, 7)
    assert torch.isfinite(loss)

    loss.backward()
    for n, p in policy.named_parameters():
        if p.requires_grad:
            assert p.grad is not None and torch.isfinite(p.grad).all(), f"bad grad: {n}"
        else:
            assert p.grad is None, f"frozen got grad: {n}"
```

- [ ] **Step 2: Run, expect PASS**

- [ ] **Step 3: Commit**

```bash
git add tests/test_one_batch_smoke.py
git commit -m "test(e2e): one-batch forward+backward smoke"
```

---

## Done criteria

When the plan is fully executed:

- [ ] `uv run pytest -q` passes (all tests).
- [ ] `uv run python scripts/train.py configs/train/smoke.yaml` runs ≥ 2 steps without NaNs (real Gemma4 + SigLIP weights, on a GPU).
- [ ] `git log --oneline` shows ~30 commits, one per task.
- [ ] No file in `models/` references ROS, robot hardware, or LeRobot.
- [ ] No hardcoded dataset paths in `src/`.
- [ ] `docs/architectures/x_vla_adapter.md` is the only authoritative spec; `docs/superpowers/plans/2026-04-30-x-vla-adapter-implementation.md` (this file) tracks task completion.

## Follow-ups (out of scope of this plan)

### Data
- Real LIBERO RLDS / HDF5 reader (replace `SyntheticLIBEROBatchDataset`).
- Multi-domain mixing (`DATA_WEIGHTS` weighted infinite sampler from X-VLA).
- Compute and persist real `NormalizationStats` (`tools/compute_norm_stats.py`).

### Model / training
- LoRA Stage 2 on Gemma4 attention modules.
- Wrist-token pooling ablation (`Nw = 49` via 4x4 avg-pool).
- Huber loss ablation run (`loss_type: huber`).
- `training/checkpoint.py`: save/load `state_dict` + config + norm_stats + git commit hash (CLAUDE.md "Experiment Outputs").
- `training/distributed.py`: `accelerate launch` integration.

### Inference / evaluation / deployment (CLAUDE.md modules currently absent)
- `policies/`: `BasePolicy` and concrete `XVLAAdapterPolicy` runtime wrapper that loads checkpoints, applies action chunking, denormalizes. CLAUDE.md draws a hard boundary between `models/vla_policy.py` (nn.Module) and `policies/` (runtime).
- `evaluation/`: `metrics.py`, `rollout.py`, `libero_eval.py`, `real_robot_eval.py`.
- `deployment/`: `inference_server.py`, `inference_client.py`, `runtime_policy.py`, `safety_filter.py`.
- `robots/`: `BaseRobot` interface + `sim_robot.py` for LIBERO closed-loop eval.
