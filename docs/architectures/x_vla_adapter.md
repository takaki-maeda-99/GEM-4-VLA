# Architecture: x_vla_adapter (Gemma4-E2B + SigLIP + VLA-Adapter head)

## Purpose

Combine X-VLA's data infrastructure (per-domain projection switching, soft prompts, weighted multi-domain sampling) with VLA-Adapter's per-layer cross-attention action head, on a Gemma4-E2B backbone with SigLIP vision encoders, to predict continuous action chunks via L1 regression.

Diagram: see `architectures.mmd`.

## Backbone configuration

- LLM: Gemma4-E2B
  - `text_config.hidden_size = 1536`
  - `num_hidden_layers = 35`
  - `hidden_size_per_layer_input = 256` (PLE / G3)
  - causal attention, MQA (`num_kv_shared_layers = 20`), dual `head_dim` (sliding 256 / global 512)
  - `tie_word_embeddings = True`
- Vision: SigLIP-So400m at 224x224, shared between scene and wrist
  - patch tokens: 256 per view, hidden 1152

## Symbol table

| Symbol | Meaning | Default |
|--------|---------|---------|
| B       | batch size | — |
| D       | LLM hidden dim | 1536 |
| D_vis   | SigLIP hidden | 1152 |
| D_prop  | proprio dim | 8 |
| D_ple   | PLE dim per layer | 256 |
| Ks      | soft prompt tokens | 32 |
| Ns      | scene tokens (SigLIP @ 224) | 256 |
| Nw      | wrist tokens (SigLIP @ 224) | 256 |
| Lt      | prompt text tokens (padded) | 50 |
| Q       | action query tokens in LLM | 64 |
| L_total | full LLM seq length | 1 + Ks + Ns + Lt + Nw + Q + 1 |
| H_act   | action chunk length | 8 |
| A       | action dim | 7 |
| L_layer | LLM transformer layers tapped | 35 |
| num_dom | number of training domains | per config |

## Module contract

| Module              | Role                                                 | Input                                      | Output                                  | Trainable               |
|---------------------|------------------------------------------------------|--------------------------------------------|------------------------------------------|--------------------------|
| SigLIP              | Extract image features (shared scene + wrist)        | `[B, V, 3, 224, 224]`                      | `[B, V, 256, D_vis]`                     | No                       |
| Scene Proj          | DomainAwareLinear `D_vis → D`                        | `[B, Ns, D_vis]`, `domain_id [B]`          | `[B, Ns, D]`                             | Yes                      |
| Wrist Proj          | DomainAwareLinear `D_vis → D`                        | `[B, Nw, D_vis]`, `domain_id [B]`          | `[B, Nw, D]`                             | Yes                      |
| SoftPrompt Hub      | DomainAwareEmbedding `(num_dom, Ks * D)`             | `domain_id [B]`                            | `[B, Ks, D]`                             | Yes                      |
| ActionQuery Hub     | DomainAwareEmbedding `(num_dom, Q * D)`              | `domain_id [B]`                            | `[B, Q, D]`                              | Yes                      |
| Proprio Proj        | DomainAwareLinear `D_prop → D` (head only)           | `[B, D_prop]`, `domain_id [B]`             | `[B, 1, D]`                              | Yes                      |
| LastAction Proj     | DomainAwareLinear `A → D` (head only)                | `[B, H_act, A]`, `domain_id [B]`           | `[B, H_act, D]`                          | Yes                      |
| Input Packer        | Build placeholder `input_ids` and overwrite indices  | text ids, scene/wrist/soft/aq embeds       | `input_ids`, `inputs_embeds`, `idx dict` | No                       |
| Gemma4 E2B          | Semantic backbone (35 layers, PLE)                   | `inputs_embeds`, `per_layer_inputs`        | `hidden_states (36-tuple)`               | Frozen Stage 1 / LoRA Stage 2 |
| Action Head         | 35 x MLPResNetBlock_Pro, cross-attn over LLM layers  | `x [B, H_act, D]`, `h_t`, `h_a`, `p`       | `[B, H_act, D]`                          | Yes                      |
| Action Decoder      | DomainAwareLinear `D → A` per domain                 | `[B, H_act, D]`, `domain_id [B]`           | `[B, H_act, A]`                          | Yes                      |

## Tensor contract

| Name                    | Shape                                | Dtype  | Meaning |
|-------------------------|--------------------------------------|--------|---------|
| `domain_id`             | `[B]`                                | long   | per-sample domain index |
| `scene_image`           | `[B, 3, 224, 224]`                   | f32    | normalized RGB |
| `wrist_image`           | `[B, 3, 224, 224]`                   | f32    | normalized RGB |
| `proprio`               | `[B, D_prop]`                        | f32    | current robot state (head only) |
| `last_action_chunk`     | `[B, H_act, A]`                      | f32    | previously emitted chunk (head only) |
| `target_action`         | `[B, H_act, A]`                      | f32    | training target |
| `action_mask`           | `[B, H_act]`                         | bool   | valid action timesteps |
| `input_ids`             | `[B, L_total]`                       | long   | prompt + placeholders (no proprio) |
| `attention_mask`        | `[B, L_total]`                       | long   | LLM padding mask |
| `scene_tokens`          | `[B, Ns, D_vis]`                     | bf16   | SigLIP scene features |
| `wrist_tokens`          | `[B, Nw, D_vis]`                     | bf16   | SigLIP wrist features |
| `scene_embeds`          | `[B, Ns, D]`                         | bf16   | scene projected to LLM dim |
| `wrist_embeds`          | `[B, Nw, D]`                         | bf16   | wrist projected to LLM dim |
| `soft_prompt_embeds`    | `[B, Ks, D]`                         | bf16   | per-domain soft prompts |
| `action_query_embeds`   | `[B, Q, D]`                          | bf16   | per-domain learnable queries |
| `proprio_embed`         | `[B, 1, D]`                          | bf16   | head conditioning `p` |
| `last_action_embed`     | `[B, H_act, D]`                      | bf16   | head input `x` initialization |
| `inputs_embeds`         | `[B, L_total, D]`                    | bf16   | LLM token embeddings (post-overwrite) |
| `per_layer_inputs`      | `[B, L_total, L_layer, D_ple]`       | bf16   | PLE from `input_ids` only |
| `hidden_states`         | tuple of 36 of `[B, L_total, D]`     | bf16   | Gemma `output_hidden_states` |
| `h_t`                   | `[B, L_layer, Ns + Lt + Nw, D]`      | bf16   | task tokens per layer |
| `h_a`                   | `[B, L_layer, Q, D]`                 | bf16   | action token states per layer |
| `pred_action`           | `[B, H_act, A]`                      | f32    | predicted action chunk |
| `ratio_g`               | `[L_layer]` (one scalar per block)   | f32    | per-block task gates `tanh(g_i)` |

## Input packing order

```
[BOS]
+ [SoftPrompt placeholder x Ks]   ( <unusedA> range )
+ [Scene placeholder x Ns]        ( <image_soft_token>, Gemma4 native )
+ [prompt text ids]               ( padded to Lt )
+ [Wrist placeholder x Nw]        ( <unusedB> range )
+ [ActionQuery placeholder x Q]   ( <unusedC> range )
+ [EOS]
```

Rules:

- Proprio is **not** in `input_ids`. It conditions the action head only.
- Placeholder ID ranges are reserved from Gemma4 vocab unused tokens; ranges defined in `data/constants.py`.
- The Input Packer must return explicit indices for each placeholder block:
  - `idx["soft"]:   [B, Ks]`
  - `idx["scene"]:  [B, Ns]`
  - `idx["wrist"]:  [B, Nw]`
  - `idx["action"]: [B, Q]`
- Downstream code overwrites `inputs_embeds` and extracts hidden states **only via these indices**.
- Hardcoded offsets outside the packer are forbidden.

## Forward pipeline

1. SigLIP encodes scene and wrist (shared weights, frozen).
2. Domain-aware projectors (`scene_proj`, `wrist_proj`) lift `D_vis → D`.
3. SoftPrompt Hub and ActionQuery Hub look up by `domain_id`.
4. Input Packer assembles `input_ids` with placeholders, returns block indices.
5. PLE: `gemma.get_per_layer_inputs(input_ids)` under `no_grad` (OOM-safe).
6. Embeds: `gemma.embed_tokens(input_ids)`, then **clone** and overwrite at placeholder indices using SoftPrompt / Scene / Wrist / ActionQuery embeddings.
7. Forward `gemma(inputs_embeds, per_layer_inputs, output_hidden_states=True)`.
8. Stack `hidden_states` into `[B, 36, L_total, D]`; index layers `1..35` for the head.
9. Extract `h_t[i]` at scene + text + wrist positions, `h_a[i]` at action positions, for `i in 1..35`.
10. Build head conditioning: `x = LastActionProj(last_action_chunk)`, `p = ProprioProj(proprio)`. **No fusion** between `x` and `p`.
11. Run 35 x MLPResNetBlock_Pro: at block `i`, attention over `{self(x), h_a[i+1] ⊕ p, ratio_g_i * h_t[i+1]}` with softmax + residual + FFN.
12. Apply Action Decoder (`DomainAwareLinear D → A`) on the head output.
13. Apply masked L1 loss against `target_action`.

### Per-block gating (carried over from VLA-Adapter MLPResNetBlock_Pro)

```
ratio_g_i = tanh(gate_i)         # gate_i = nn.Parameter(torch.zeros(1))
attn_self    = q @ k_self^T                # weight 1
attn_adapter = q @ k_adapter^T             # weight 1   (k_adapter = h_a[i+1] concat p)
attn_task    = q @ k_task^T  * ratio_g_i   # weight ratio_g_i  (k_task = h_t[i+1])
attn = softmax([attn_self, attn_adapter, attn_task] / sqrt(d_h))
```

`gate_i` initialized to 0 -> `tanh(0) = 0` -> task tokens ignored at start, gated in over training.

## Loss

```
loss = masked_l1(pred_action, target_action, action_mask)
```

Definition:

```
m_expanded = action_mask.unsqueeze(-1).expand_as(pred_action)   # [B, H_act, A]
loss = (|pred_action - target_action| * m_expanded).sum() / max(m_expanded.sum(), 1)
```

Constraints:

- Padded action timesteps must not contribute to gradient.
- Optional ablation: swap to `smooth_l1_loss(beta=0.1)` with the same masking, behind a config switch (`loss_type: "l1" | "huber"`).

## Trainable parameters

### Stage 1 (default)

Trainable:

- SoftPrompt Hub
- ActionQuery Hub
- Scene Proj, Wrist Proj
- Proprio Proj
- LastAction Proj
- Action Decoder
- Action Head (`MLPResNet` wrapper + 35 x `MLPResNetBlock_Pro`)

Frozen:

- SigLIP base encoder
- Gemma4 E2B base weights (no LoRA)

### Stage 2 (optional)

- Add LoRA adapters on Gemma4 attention modules (typical: `q_proj`, `v_proj`).
- Keep all Stage 1 modules trainable.
- SigLIP remains frozen.

### Per-group learning rate

Following X-VLA convention (`build_optimizer` in `train.py`):

| Group              | Stage 1 LR        | Stage 2 LR        |
|--------------------|-------------------|-------------------|
| Gemma base         | 0 (frozen)        | 0 (LoRA only)     |
| Gemma LoRA         | n/a               | `lr`              |
| SigLIP             | 0 (frozen)        | 0 (frozen)        |
| Soft prompts       | `lr * coef_soft`  | `lr * coef_soft`  |
| Action Queries     | `lr`              | `lr`              |
| Domain projections | `lr`              | `lr`              |
| Action Head        | `lr`              | `lr`              |

## Required tests

- `test_packer_indices`: `input_ids` length matches `1 + Ks + Ns + Lt + Nw + Q + 1`; placeholder indices land at expected ranges.
- `test_ple_shape`: `per_layer_inputs.shape == [B, L_total, 35, 256]`.
- `test_inputs_embeds_overwrite`: at each placeholder index, `inputs_embeds` equals the corresponding adapter output (not the raw token embedding).
- `test_domain_aware_swap`: changing `domain_id` produces different projection / soft prompt / action query outputs.
- `test_forward_shape`: `model(batch)` returns `pred_action` of shape `[B, H_act, A]`.
- `test_action_query_extraction`: head consumes `h_a` strictly from action placeholder positions.
- `test_action_loss_mask`: padded action timesteps yield zero gradient on `pred_action`.
- `test_trainable_parameters`: Stage 1 — SigLIP and Gemma have `requires_grad=False`; everything else trainable.
- `test_gating_init`: every `MLPResNetBlock_Pro.gate_i` initialized to 0; `tanh(gate) = 0` at step 0.
- `test_one_batch_smoke`: forward + backward on a tiny LIBERO batch completes without NaN or shape error.

## Open questions / future work

- Multi-domain mixing: when `num_dom > 1`, decide per-domain `H_act`, `A`, and `D_prop` policy (X-VLA uses padded common dim with masking).
- LoRA on Gemma in Stage 2: target module set, rank, alpha.
- Variable prompt length `Lt`: pad to `Lt_max = 50` (X-VLA convention) vs dynamic per-batch.
- Wrist token pooling: keep `Nw = 256` raw or pool to 49 if VRAM constrained.
- Ablation switches behind config: `loss_type`, `use_pro_version`, `wrist_bridge_layer_mode`, `freeze_steps`.
