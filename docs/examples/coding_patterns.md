# Coding Patterns and Examples

Reference templates for the X-VLA-Adapter project. These examples show the
**patterns** to follow when implementing configs, packing modules, tests,
experiment cards, and agent task prompts.

The numerical defaults in these templates (e.g. `hidden_size: 2560`, Gemma2-2B,
`horizon: 16`, `masked_mse`) are illustrative only. The authoritative values
for the X-VLA-Adapter architecture live in
`docs/architectures/x_vla_adapter.md` (Gemma4-E2B, `D = 1536`, `H_act = 8`,
masked L1).

When implementing the actual project files, reuse these structural patterns
but substitute the X-VLA-Adapter values from the architecture card.

---

# Example 4: `configs/model/gemma_siglip_action_query.yaml`

```yaml
name: gemma_siglip_action_query

model:
  type: vla_policy
  hidden_size: 2560

vision:
  enabled: true
  encoder:
    type: siglip
    pretrained_model_name_or_path: google/siglip-base-patch16-224
    freeze: true
    image_size: 224
    output: patch_tokens
  projector:
    type: mlp
    input_dim: 768
    hidden_dim: 2048
    output_dim: 2560
    num_layers: 2

language:
  backbone:
    type: gemma
    pretrained_model_name_or_path: google/gemma-2-2b
    freeze_base: true
    gradient_checkpointing: true
  lora:
    enabled: true
    r: 16
    alpha: 32
    dropout: 0.05
    target_modules:
      - q_proj
      - k_proj
      - v_proj
      - o_proj

proprio:
  enabled: true
  input_dim: 8
  num_tokens: 1
  projector:
    type: mlp
    hidden_dim: 512
    output_dim: 2560
    num_layers: 2

fusion:
  type: input_packing
  order:
    - text
    - vision
    - proprio
    - action_queries
  return_token_indices: true

action_queries:
  enabled: true
  num_queries: 64
  hidden_dim: 2560
  init_std: 0.02

action_head:
  type: continuous_mlp
  input: action_query_states
  hidden_dim: 1024
  num_layers: 3
  horizon: 16
  action_dim: 7

loss:
  type: masked_mse
  prediction_key: pred_actions
  target_key: action
  mask_key: action_mask

normalization:
  action:
    enabled: true
    stats_path: outputs/norm_stats/action_stats.json
  proprio:
    enabled: true
    stats_path: outputs/norm_stats/proprio_stats.json
```

---

# Example 5: `configs/data/lerobot.yaml`

```yaml
name: lerobot_example

dataset:
  type: lerobot
  repo_id: your_org/your_dataset
  split: train

schema:
  image_keys:
    image_primary: observation.images.primary
    image_wrist: observation.images.wrist
  proprio_key: observation.state
  action_key: action
  language_key: task

sequence:
  obs_horizon: 1
  action_horizon: 16
  stride: 1

image:
  size: 224
  channels_first: true
  normalize: true

action:
  dim: 7
  space: delta_ee_pose_gripper
  normalize: true

proprio:
  dim: 8
  normalize: true
```

---

# Example 6: `src/vla_project/models/packing/action_query_packer.py`

```python
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class PackedInputs:
    inputs_embeds: Tensor
    attention_mask: Tensor
    action_query_start: int
    action_query_end: int


class ActionQueryInputPacker:
    """
    Packs text, vision, proprio, and action query tokens into one LLM input sequence.

    Packing order:
        [text] [vision] [proprio] [action_queries]
    """

    def __call__(
        self,
        *,
        text_embeds: Tensor,
        text_attention_mask: Tensor,
        visual_tokens: Tensor,
        proprio_tokens: Tensor | None,
        action_queries: Tensor,
    ) -> PackedInputs:
        assert text_embeds.ndim == 3, (
            f"text_embeds must be [B, L_text, D], got {tuple(text_embeds.shape)}"
        )
        assert visual_tokens.ndim == 3, (
            f"visual_tokens must be [B, N_vis, D], got {tuple(visual_tokens.shape)}"
        )
        assert action_queries.ndim == 3, (
            f"action_queries must be [B, Q, D], got {tuple(action_queries.shape)}"
        )

        batch_size, _, hidden_size = text_embeds.shape

        assert visual_tokens.shape[0] == batch_size
        assert visual_tokens.shape[2] == hidden_size
        assert action_queries.shape[0] == batch_size
        assert action_queries.shape[2] == hidden_size

        pieces = [text_embeds, visual_tokens]

        if proprio_tokens is not None:
            assert proprio_tokens.ndim == 3, (
                f"proprio_tokens must be [B, N_prop, D], got {tuple(proprio_tokens.shape)}"
            )
            assert proprio_tokens.shape[0] == batch_size
            assert proprio_tokens.shape[2] == hidden_size
            pieces.append(proprio_tokens)

        action_query_start = sum(x.shape[1] for x in pieces)
        pieces.append(action_queries)
        action_query_end = action_query_start + action_queries.shape[1]

        inputs_embeds = torch.cat(pieces, dim=1)

        extra_len = inputs_embeds.shape[1] - text_attention_mask.shape[1]
        extra_mask = torch.ones(
            batch_size,
            extra_len,
            dtype=text_attention_mask.dtype,
            device=text_attention_mask.device,
        )
        attention_mask = torch.cat([text_attention_mask, extra_mask], dim=1)

        assert inputs_embeds.shape[:2] == attention_mask.shape

        return PackedInputs(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            action_query_start=action_query_start,
            action_query_end=action_query_end,
        )
```

---

# Example 7: `tests/test_gemma_siglip_action_query_forward.py`

```python
import torch

from vla_project.models.packing.action_query_packer import ActionQueryInputPacker


def test_action_query_packer_shapes():
    batch_size = 2
    l_text = 12
    n_vis = 196
    n_prop = 1
    q = 64
    d = 2560

    text_embeds = torch.randn(batch_size, l_text, d)
    text_attention_mask = torch.ones(batch_size, l_text, dtype=torch.long)
    visual_tokens = torch.randn(batch_size, n_vis, d)
    proprio_tokens = torch.randn(batch_size, n_prop, d)
    action_queries = torch.randn(batch_size, q, d)

    packer = ActionQueryInputPacker()

    packed = packer(
        text_embeds=text_embeds,
        text_attention_mask=text_attention_mask,
        visual_tokens=visual_tokens,
        proprio_tokens=proprio_tokens,
        action_queries=action_queries,
    )

    expected_len = l_text + n_vis + n_prop + q

    assert packed.inputs_embeds.shape == (batch_size, expected_len, d)
    assert packed.attention_mask.shape == (batch_size, expected_len)

    assert packed.action_query_start == l_text + n_vis + n_prop
    assert packed.action_query_end == expected_len

    action_query_states = packed.inputs_embeds[
        :, packed.action_query_start : packed.action_query_end
    ]

    assert action_query_states.shape == (batch_size, q, d)


def test_action_loss_mask_shape():
    batch_size = 2
    horizon = 16
    action_dim = 7

    pred_actions = torch.randn(batch_size, horizon, action_dim)
    target_actions = torch.randn(batch_size, horizon, action_dim)
    action_mask = torch.ones(batch_size, horizon)

    assert pred_actions.shape == target_actions.shape
    assert action_mask.shape == pred_actions.shape[:2]

    loss_per_step = ((pred_actions - target_actions) ** 2).mean(dim=-1)
    masked_loss = (loss_per_step * action_mask).sum() / action_mask.sum().clamp_min(1.0)

    assert masked_loss.ndim == 0
    assert torch.isfinite(masked_loss)
```

---

# Example 8: エージェントに渡す実装依頼

```md
Task:
Implement the input packing module for the `gemma_siglip_action_query` architecture.

Reference files:
- `docs/architectures/gemma_siglip_action_query.md`
- `configs/model/gemma_siglip_action_query.yaml`
- `CLAUDE.md`

Allowed files to modify:
- `src/vla_project/models/packing/action_query_packer.py`
- `src/vla_project/models/packing/__init__.py`
- `tests/test_gemma_siglip_action_query_forward.py`

Do not modify:
- dataset code
- training loop
- robot code
- policy runtime
- unrelated configs

Required behavior:
- pack text, vision, proprio, and action query tokens
- packing order must be:
  `[text] [vision] [proprio] [action_queries]`
- return explicit action query start/end indices
- do not infer action query indices outside the packer
- fail fast on invalid tensor shapes

Required tests:
- packed input shape
- attention mask shape
- action query index correctness
- extraction of action query states

Before editing:
Summarize the current intended data flow and tensor shapes.
```

---

# Example 9: アーキテクチャ変更時の依頼

```md
Task:
Create a new architecture variant based on `gemma_siglip_action_query`.

New variant:
`gemma_siglip_flow_matching`

Change only:
- replace `continuous_mlp` action head with `flow_matching_head`
- keep SigLIP
- keep Gemma
- keep input packing order
- keep action queries
- keep LoRA settings

Create:
- `docs/architectures/gemma_siglip_flow_matching.md`
- `configs/model/gemma_siglip_flow_matching.yaml`
- `tests/test_gemma_siglip_flow_matching_forward.py`

Do not modify:
- existing `gemma_siglip_action_query` files
- dataset code
- robot code
- training loop unless required by the new loss interface

The new architecture card must include:
- Mermaid dataflow
- module contract table
- tensor contract table
- loss description
- required tests
```

---

# Example 10: 実験ごとの `docs/experiments/exp001.md`

```md
# Experiment: exp001_gemma_siglip_action_query_libero

## Goal

Test whether Gemma + SigLIP + action queries can overfit a small LIBERO subset.

## Architecture

- Architecture card: `docs/architectures/gemma_siglip_action_query.md`
- Model config: `configs/model/gemma_siglip_action_query.yaml`
- Data config: `configs/data/libero.yaml`

## Training setup

```yaml
batch_size: 8
max_steps: 2000
learning_rate: 1.0e-4
precision: bf16
gradient_checkpointing: true
```

## Expected checks

- model forward works
- loss is finite
- action loss decreases on a tiny subset
- trainable parameters are only:
  - projector
  - proprio projector
  - action queries
  - LoRA adapters
  - action head

## Notes

This experiment is not intended to prove benchmark performance.
It is only a pipeline and overfit check.
```
