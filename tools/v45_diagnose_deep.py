"""v45 step_35000 deep diagnose:

  C. task-stratified occlusion (group per-sample loss by prompt_input_ids identity)
  D. soft_prompt occlusion (zero out policy.soft_prompt_hub output)
  E. task swap (shuffle prompts within batch -> wrong language for image)

Uses n_batches batches × bs=4 = 4*n_batches samples for noise floor.

Usage:
    .venv/bin/python tools/v45_diagnose_deep.py [n_batches]
"""
from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from vla_project.data.datasets.rlds_libero_dataset import RLDSLiberoDataset
from vla_project.data.transforms.language import GemmaPromptTokenizer
from vla_project.models.language.gemma4_wrapper import Gemma4Wrapper
from vla_project.models.vision.factory import build_vision_encoder
from vla_project.models.vla_policy import VLAPolicy, VLAPolicyConfig


_DEFAULT_CKPT = (
    "outputs/oxe_pretrain_v45_nb18even_proper_mlp_alllinear_libero_dl50_bs8"
    "/checkpoints/step_35000"
)
# argv layout: tools/v45_diagnose_deep.py [ckpt_path] [n_batches]
CKPT = sys.argv[1] if len(sys.argv) > 1 else _DEFAULT_CKPT
N_BATCHES = int(sys.argv[2]) if len(sys.argv) > 2 else 30
BATCH_SIZE = 4


def load_policy():
    meta = json.loads((Path(CKPT) / "meta.json").read_text())
    cfg = OmegaConf.create(meta["cfg"])
    md = OmegaConf.to_container(cfg.model, resolve=True)
    lora_cfg = md.pop("lora", None)
    policy = VLAPolicy(
        VLAPolicyConfig(**md),
        build_vision_encoder(vision_type="timm", model_name=cfg.vision.model_name),
        Gemma4Wrapper(model_name=cfg.language.model_name, freeze=True, lora=lora_cfg),
    ).to("cuda").to(torch.bfloat16)
    sd = torch.load(Path(CKPT) / "model.pt", map_location="cpu", weights_only=False)
    if any(k.startswith("_orig_mod.") for k in sd):
        sd = {k.removeprefix("_orig_mod."): v for k, v in sd.items()}
    policy.load_state_dict(sd, strict=False)
    policy.eval()
    return policy, cfg


def _per_sample_l1(pred, target, mask):
    """Return per-sample L1 loss, shape (B,)."""
    # pred, target: (B, T, A); mask: (B, T) bool
    abs_diff = (pred - target).abs()  # (B, T, A)
    m = mask.unsqueeze(-1).to(abs_diff.dtype)  # (B, T, 1)
    masked = abs_diff * m
    A = abs_diff.shape[-1]
    denom = m.expand_as(abs_diff).sum(dim=(1, 2)).clamp_min(1)
    return (masked.sum(dim=(1, 2)) / denom).detach().float()


@contextmanager
def _zero_soft_prompt(policy):
    """Force soft_prompt_hub(domain_id) -> zeros for the duration of the block."""
    hub = getattr(policy, "soft_prompt_hub", None)
    if hub is None:
        yield
        return
    orig = hub.forward
    def _zero_fwd(domain_id):
        out = orig(domain_id)
        return torch.zeros_like(out)
    hub.forward = _zero_fwd
    try:
        yield
    finally:
        hub.forward = orig


@contextmanager
def _zero_action_queries(policy):
    """Temporarily zero out action_query_hub.queries (learned 64 anchor tokens)."""
    hub = getattr(policy, "action_query_hub", None)
    if hub is None or not hasattr(hub, "queries"):
        yield
        return
    saved = hub.queries.data.detach().clone()
    hub.queries.data.zero_()
    try:
        yield
    finally:
        hub.queries.data.copy_(saved)


def _to_device(b, device, dtype):
    out = {}
    for k, v in b.items():
        if torch.is_tensor(v):
            if v.dtype.is_floating_point:
                out[k] = v.to(device).to(dtype)
            else:
                out[k] = v.to(device)
        else:
            out[k] = v
    return out


def main():
    print(f"[deep] ckpt={CKPT}")
    print(f"[deep] n_batches={N_BATCHES} bs={BATCH_SIZE}")
    policy, cfg = load_policy()

    tok = GemmaPromptTokenizer(model_name=cfg.language.model_name, max_len=20)
    ds = RLDSLiberoDataset(
        data_dir="/misc/dl00/takaki/vla-gemma-4/data/modified_libero_rlds",
        dataset_name="libero_spatial_no_noops",
        tokenizer=tok,
        action_chunk_len=8,
        shuffle_buffer_size=2048,
        train=True,
        domain_id=12,
        seed=42,
    )
    dl = DataLoader(ds, batch_size=BATCH_SIZE, collate_fn=RLDSLiberoDataset.collate_fn)
    device = next(policy.parameters()).device
    dtype = next(policy.parameters()).dtype

    # Per-sample losses across conditions.
    per_sample_losses: dict[str, list[float]] = defaultdict(list)
    # Per-task aggregation: key = tuple(prompt_input_ids) for the sample
    per_task: dict[tuple, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    n = 0
    with torch.no_grad():
        for batch in dl:
            if n >= N_BATCHES:
                break
            n += 1
            b = _to_device(batch, device, dtype)
            target = b["target_action"]
            mask = b["action_mask"]
            B = target.shape[0]

            # Per-sample task-key (token sequence as tuple of ints)
            task_keys = [tuple(b["prompt_input_ids"][i].tolist()) for i in range(B)]

            def _fwd_and_record(label, _b):
                pred, _ = policy(_b)
                ps = _per_sample_l1(pred, target, mask)
                for i in range(B):
                    v = float(ps[i])
                    per_sample_losses[label].append(v)
                    per_task[task_keys[i]][label].append(v)

            # baseline
            _fwd_and_record("baseline", b)
            # no_scene
            _fwd_and_record("no_scene", {**b, "scene_image": torch.zeros_like(b["scene_image"])})
            # no_wrist
            _fwd_and_record("no_wrist", {**b, "wrist_image": torch.zeros_like(b["wrist_image"])})
            # no_proprio
            _fwd_and_record("no_proprio", {**b, "proprio": torch.zeros_like(b["proprio"])})
            # no_lang
            _fwd_and_record(
                "no_lang",
                {
                    **b,
                    "prompt_input_ids": torch.zeros_like(b["prompt_input_ids"]),
                    "prompt_attention_mask": torch.zeros_like(b["prompt_attention_mask"]),
                },
            )
            # no_softprompt
            with _zero_soft_prompt(policy):
                _fwd_and_record("no_softprompt", b)
            # no_action_queries: zero out action_query_hub.queries; the Gemma
            # action positions now carry pure zero embeddings, so h_a becomes
            # "what zero-queries attend out of other tokens via Gemma" only
            with _zero_action_queries(policy):
                _fwd_and_record("no_action_queries", b)
            # swap_lang: roll prompt_input_ids / mask by 1 within batch
            _fwd_and_record(
                "swap_lang",
                {
                    **b,
                    "prompt_input_ids": torch.roll(b["prompt_input_ids"], shifts=1, dims=0),
                    "prompt_attention_mask": torch.roll(b["prompt_attention_mask"], shifts=1, dims=0),
                },
            )

    # ------- Report -------
    def _stats(vs):
        if not vs:
            return (0.0, 0.0, 0.0)
        m = sum(vs) / len(vs)
        var = sum((x - m) ** 2 for x in vs) / max(1, len(vs) - 1)
        sd = math.sqrt(max(var, 0.0))
        # 95% CI half-width = 1.96 * sd / sqrt(n)
        ci = 1.96 * sd / math.sqrt(len(vs))
        return m, sd, ci

    print(f"\n=== Aggregate over {len(per_sample_losses['baseline'])} samples ===")
    base_m, base_sd, base_ci = _stats(per_sample_losses["baseline"])
    print(f"  baseline = {base_m:.4f} ± {base_ci:.4f} (95% CI; sd={base_sd:.4f})")
    print(f"  {'condition':<15s} {'mean':>10s} {'95% CI':>10s} {'Δ vs base':>12s} {'rel %':>8s} {'signif?':>8s}")
    print(f"  {'-'*15} {'-'*10} {'-'*10} {'-'*12} {'-'*8} {'-'*8}")
    for cond in ("no_scene", "no_wrist", "no_proprio", "no_lang", "no_softprompt", "no_action_queries", "swap_lang"):
        m, sd, ci = _stats(per_sample_losses[cond])
        delta = m - base_m
        # Paired t-test simplification: mean of diffs / (sd of diffs / sqrt(n)), 95% threshold ~ 1.96
        diffs = [per_sample_losses[cond][i] - per_sample_losses["baseline"][i] for i in range(len(per_sample_losses[cond]))]
        dm, dsd, dci = _stats(diffs)
        signif = "*" if abs(dm) > dci else ""
        rel = (delta / base_m) * 100 if base_m > 0 else 0.0
        print(f"  {cond:<15s} {m:>10.4f} ±{ci:>9.4f} {delta:>+12.4f} {rel:>+7.1f}% {signif:>8s}")
    print("  (* = paired Δ exceeds 95% CI, statistically significant)")

    print(f"\n=== Per-task (LIBERO Spatial; grouped by prompt token sequence) ===")
    print(f"  {len(per_task)} unique prompts encountered\n")
    rows = []
    for task_key, by_cond in per_task.items():
        n_s = len(by_cond["baseline"])
        bm, _, _ = _stats(by_cond["baseline"])
        nl_m, _, _ = _stats(by_cond["no_lang"])
        sl_m, _, _ = _stats(by_cond["swap_lang"])
        sp_m, _, _ = _stats(by_cond["no_softprompt"])
        rows.append((n_s, bm, nl_m - bm, sl_m - bm, sp_m - bm, task_key))
    rows.sort(key=lambda r: -r[0])
    # Try to decode the task by detokenizing
    print(f"  {'n':>4s}  {'base':>8s}  {'Δno_lang':>10s}  {'Δswap_lang':>11s}  {'Δno_sp':>9s}  task")
    print(f"  {'-'*4}  {'-'*8}  {'-'*10}  {'-'*11}  {'-'*9}  {'-'*40}")
    for n_s, bm, dnl, dsl, dsp, task_key in rows:
        # Decode token ids to text (strip pads)
        ids = [t for t in task_key if t != 0]
        try:
            text = tok.tokenizer.decode(ids, skip_special_tokens=True).strip().replace("\n", " ")[:48]
        except Exception:
            text = "<decode-err>"
        print(f"  {n_s:>4d}  {bm:>8.4f}  {dnl:>+10.4f}  {dsl:>+11.4f}  {dsp:>+9.4f}  {text}")


if __name__ == "__main__":
    main()
