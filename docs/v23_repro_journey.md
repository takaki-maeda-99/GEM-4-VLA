# X-VLA-Adapter: vla-gemma-4 73% baseline 再現 (v6 → v23)

## ゴール

`vla-gemma-4/runs/.../libero_b_siglip_10k_wristb_b16_v2` (LIBERO-Spatial 73%
baseline) と等価な closed-loop 性能を、**我々の X-VLA-Adapter 実装で**
再現する。

- 環境再現確認: vla-gemma-4 codebase を本機で再学習 → step_10000 で
  21/50 = 42% closed-loop 達成（元論文 73% より低いのは re-train ノイズ +
  layout 互換性、論文値は old prompt-first ckpt + `VLA_OLD_PROMPT_FIRST=1`
  combination）。
- ターゲット: 我々の codebase で baseline 同等 (40%+) を出す。

---

## アーキテクチャ (v23 = 73% baseline match)

```
Inputs                                                        Action head
─────────────────────────────────────────                     ────────────
scene image  ─→ SigLIP (frozen, 248→crop 224)  ──┐
wrist image  ─→ SigLIP                            │
prompt text  ─→ tokenizer (max=20, no BOS)        │
                                                  ▼
LLM input_ids:  [BOS][prompt(20)][V(256)][PROPRIO(1)][A(64)][EOS]
                                                  │
embed_tokens(input_ids) → embeddings              │
  scene-pos     ←  scene_proj(SigLIP_scene_last)  │
  action-pos    ←  action_queries (zero-init)     │
                                                  ▼
Gemma4-E2B (35 layers, frozen, attn_mask all-1)
  → hidden_states tuple (36 layers incl. embedding)
                                                  │
slice [0..24]                                     │
  vision_hidden = HS[i, vpos]      (B,25,256,D)   │
  action_hidden = HS[i, apos]      (B,25,64,D)    │
  combined      = cat([vision, action], dim=2)    ▼

action_head.predict_action:
  task_h     = combined[:, :, :256, :]      = vision_hidden
  actions_h  = combined[:, :, 256:, :]      = action_hidden
  proprio_e  = proprio_proj(BOUNDS_Q99(proprio))   (B,1,D)
  x_init     = zeros(B, 8, 7*D) + 0.02·randn      (Training only)

  ┌────────── MLPResNet (24 blocks, legacy MLPResNetBlock_Pro) ──────────┐
  │ x = LN1(x); fc1; ReLU                                                │
  │ x = cat(x, h_sp)         # h_sp=None for libero (use_soft_prompt=F)  │
  │ for i in range(24):                                                  │
  │   block_i:                                                            │
  │     q = q_proj(x)                                                    │
  │     k_self/v_self  = (x)                                             │
  │     k_adapter/v_adapter = (cat(actions_h[:,i+1], proprio_e))         │
  │     k_task/v_task  = (vision_h[:, i+1])  # tanh(g)·attn  gated       │
  │     k_wrist/v_wrist = (h_w_bridge[:, i+1])  # SigLIP block (i+1)     │
  │                       per-layer wrist-bridge cross-attn               │
  │                       tanh(g_wrist)·attn  gated                       │
  │     softmax over [self, adapter, task, wrist]                        │
  │     RoPE on q/k_self/k_adapter/k_task/k_wrist                        │
  │     x = ffn(o_proj(attn) + x)                                        │
  │ x = x[:, :8, :]    # trim back to action positions                   │
  │ x = fc2(LN2(x))                                                       │
  └──────────────────────────────────────────────────────────────────────┘
                                                  │
action_decoder (DA-Linear D→7)                    ▼
                                          predicted action chunk (B, 8, 7)
loss = F.l1_loss(predicted, actions)          [actions: BOUNDS_Q99 normalized]
```

学習対象 (Mode B, no LoRA):
- soft_prompt_library: **None** (use_soft_prompt=False)
- action_queries: shared (B,Q,D), zero-init
- proprio_projector / scene_proj / wrist_proj / wrist_projector_bridge / action_decoder
- action_head (24 blocks × MLPResNetBlock_Pro)

凍結:
- Gemma4-E2B (full LLM)
- SigLIP-So400m

ハイパー:
- bs=8 / GPU × 4 GPU DDP = effective 32
- lr=2e-4, wd=0.01, clip=1.0, warmup=500, flat after warmup
- AdamW betas=(0.9, 0.999) (PyTorch default, matches baseline)
- max_steps=10000

---

## トラシュー履歴 (v6 → v23)

| run | 主な diff | step | train loss plateau | closed-loop |
|-----|-----------|------|--------------------|-------------|
| v6  | レイアウト + prompt fmt 修正 | 10000 | 0.39 | 0/50 |
| v9  | wrist_bridge port (gated, cold init) | 5000 | 0.20+ | 0/20 |
| v10 | gating warm-init=0.5 | 2500 | 0.50 (frozen) | 0/20 |
| v11 | ungated streams (avoid bf16 trap) | killed | — | — |
| v12 | proprio Q99 normalization | killed | 0.30 (early) | — |
| v13 | SigLIP transform Resize 248 + CenterCrop 224 | killed | — | — |
| v14 | AdamW betas (0.9, 0.999) | killed | — | — |
| v15 | action_queries zero init + soft_prompt off | killed | — | — |
| v16 | prompt_max_len 50 → 20 | 3500 | 0.30 | — |
| v17 | DDP-4 effective bs=32 | 7500 | 0.25 | 0/50 |
| v18 | all-ones attention_mask | 5000 | 0.25 | 0/50 |
| v19 | SigLIP layer indexing fix (skip embedding) | 2400 | 0.30 | — |
| v20 | explicit DDP rank shard | 1300 | 0.30 high-variance | — |
| v21 | shuffle frame indices | 2500 | 0.28 plateau | 0/50 |
| v22 | revert ungated → gated (init=0) | 900 | 0.30 plateau | — |
| **v23** | **RLDS data swap (modified_libero_rlds)** | **4270** | **0.14** ✓ | **TBD** |

### 主要 root causes

| # | 問題 | 発見ステップ | 修正 |
|---|------|--------------|------|
| 1 | proprio が raw axis-angle/m units (z≈0.92, rx≈3.14) | v6 const-pred 診断 | BOUNDS_Q99 正規化 (v12) |
| 2 | SigLIP 直接 224 resize で FOV 違い | コード比較 | Resize 248 + CenterCrop 224 (v13) |
| 3 | AdamW betas (0.9, 0.95) が aggressive | コード比較 | (0.9, 0.999) PyTorch default (v14) |
| 4 | action_queries Normal init | コード比較 | zero init (v15) |
| 5 | soft_prompt 不要だが allocate されてた | コード比較 | use_soft_prompt=False (v15) |
| 6 | prompt_max_len=50 で RoPE 位置ずれ | コード比較 | =20 に (v16) |
| 7 | gated cross-attn が ramp 遅すぎ → ungated に変更 → bf16 で warm-init 凍結 | gating 診断 | Mode B + bs=32 で十分 ramp、gated 戻す (v22) |
| 8 | attention_mask が prompt padding を mask してた | コード比較 | all-ones (v18) |
| 9 | wrist_bridge layer indexing off-by-one (HF embedding 含めてた) | コード比較 | skip embedding, blocks 0..24 (v19) |
| 10 | DDP IterableDataset shard が機能してなかった | loss curve 観察 | explicit rank shard (v20) |
| 11 | shuffle なしで batch correlation 高 | loss variance | per-rank shuffle (v21) |
| 12 | **LeRobot dataset 値が baseline と微妙に違う** | loss plateau 0.28 残存 | **RLDS 直 (v23)** ← 決定打 |

### v23 (RLDS) 設定上の注意

- 我々の X-VLA-Adapter codebase + vla-gemma-4 venv (transformers 5.5.4 + tf 2.15)
- 起動コマンド (`docs/v23_repro_journey.md` 参照):
  ```bash
  CUDA_VISIBLE_DEVICES=3,4,5,6 \
  PYTHONPATH=/misc/dl00/takaki/X-VLA-Adapter/src:\
/misc/dl00/takaki/vla-gemma-4/VLA-Adapter:\
/misc/dl00/takaki/vla-gemma-4 \
  /misc/dl00/takaki/vla-gemma-4/.venv-gemma4/bin/python \
    -m accelerate.commands.launch \
    --num_processes 4 --num_machines 1 --multi_gpu --mixed_precision no \
    --main_process_port 29509 \
    /misc/dl00/takaki/X-VLA-Adapter/scripts/train.py \
    /misc/dl00/takaki/X-VLA-Adapter/configs/train/libero_spatial_v23.yaml
  ```
- eval も同 venv (ckpt の SigLIP module hierarchy が transformers 5.5.4 版で固定):
  ```bash
  CUDA_VISIBLE_DEVICES=2 \
  PYTHONPATH=…(同上) \
  /misc/dl00/takaki/vla-gemma-4/.venv-gemma4/bin/python \
    /misc/dl00/takaki/X-VLA-Adapter/scripts/eval.py \
    /misc/dl00/takaki/X-VLA-Adapter/configs/eval/libero_v23_step2500.yaml
  ```
- データ path: `/misc/dl00/takaki/vla-gemma-4/data/modified_libero_rlds/`
- stats path: `/misc/dl00/takaki/vla-gemma-4/VLA-Adapter/outputs/LIBERO-Spatial-Pro/dataset_statistics.json`

---

## 現状 (2026-05-04 00:?? 時点)

- v23 train step 4270, loss 0.14 (baseline 0.13 と等価)
- v23 step_2500 eval = 0/50 (baseline step_1000 も 0/20、early ckpt は閾値以下)
- step_5000 ETA ~30 min、step_10000 ETA ~4h
- step_10000 closed-loop が baseline 42% 同等出れば再現完了

## 次の判断分岐

| step_10000 closed-loop | 結論 | 次の手 |
|------------------------|------|--------|
| ≥ 35% | ✅ 再現成功 | 実験終了、論文値 (73%) は別問題 (元 ckpt + old layout) |
| 15–35% | △ 部分一致 | seed / DDP world_size の違いを潰す |
| < 15% | ❌ まだバグあり | eval-time policy / sim_robot のさらに深い diff を追う |
