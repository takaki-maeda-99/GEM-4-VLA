"""Save / load self-contained checkpoint directories.

A checkpoint is a directory with the following layout::

  step_<N>/
  ├── model.pt        # torch.save(model.state_dict())
  ├── optimizer.pt    # torch.save(optimizer.state_dict())   [optional]
  └── meta.json       # step, cfg, norm_stats, git_commit, tokenizer_settings

Saves are atomic: the contents are first written to a sibling ``.tmp`` dir,
then renamed into place, so a crashed save cannot leave a half-baked dir.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn


def _git_commit() -> str:
    """Return the current HEAD commit, suffixed with ``-dirty`` if the working
    tree has uncommitted changes. Returns ``"unknown"`` if not in a repo."""
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"
    try:
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"],
            stderr=subprocess.DEVNULL,
        ).decode().strip())
    except (subprocess.CalledProcessError, FileNotFoundError):
        dirty = False
    return f"{sha}-dirty" if dirty else sha


def _resolve_cfg(cfg: Any) -> Any:
    """Best-effort coerce OmegaConf DictConfig (or plain dict) into a plain
    JSON-serializable container, resolving ``${oc.env:...}`` interpolations."""
    try:
        from omegaconf import DictConfig, ListConfig, OmegaConf
    except ImportError:  # pragma: no cover — omegaconf is a project dep
        return cfg
    if isinstance(cfg, (DictConfig, ListConfig)):
        return OmegaConf.to_container(cfg, resolve=True)
    return cfg


def save_checkpoint(
    out_dir: Union[str, Path],
    model: nn.Module,
    *,
    step: int,
    cfg: Any,
    norm_stats: Optional[Dict[str, Any]] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    tokenizer_settings: Optional[Dict[str, Any]] = None,
) -> None:
    out = Path(out_dir)
    parent = out.parent
    parent.mkdir(parents=True, exist_ok=True)
    tmp = parent / (out.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir()

    torch.save(model.state_dict(), tmp / "model.pt")
    if optimizer is not None:
        torch.save(optimizer.state_dict(), tmp / "optimizer.pt")

    meta = {
        "step": int(step),
        "cfg": _resolve_cfg(cfg),
        "norm_stats": norm_stats,
        "tokenizer_settings": tokenizer_settings,
        "git_commit": _git_commit(),
    }
    (tmp / "meta.json").write_text(json.dumps(meta, indent=2))

    # Atomic replace: remove any existing dir at the target path, then rename.
    if out.exists():
        shutil.rmtree(out)
    tmp.rename(out)


def load_checkpoint(
    in_dir: Union[str, Path],
    model: nn.Module,
    *,
    optimizer: Optional[torch.optim.Optimizer] = None,
    strict: bool = True,
) -> Dict[str, Any]:
    in_path = Path(in_dir)
    if not in_path.is_dir():
        raise FileNotFoundError(f"checkpoint dir not found: {in_path}")
    model_pt = in_path / "model.pt"
    if not model_pt.is_file():
        raise FileNotFoundError(f"missing model.pt under {in_path}")
    meta_path = in_path / "meta.json"

    state = torch.load(model_pt, map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(state, strict=strict)
    if not strict:
        # Report what was skipped so the caller can sanity-check the load.
        # Filter the obvious frozen-backbone ranges (vision_encoder.*, gemma.*)
        # because those are not expected in adapter-only ckpts (e.g. ours, or
        # the converted baseline ckpt — both store only trainable params).
        rel_missing = [
            k for k in missing
            if not k.startswith(("vision_encoder.", "gemma."))
        ]
        if rel_missing:
            print(f"[load_checkpoint] missing (random-init): {rel_missing[:8]} total={len(rel_missing)}")
        if unexpected:
            print(f"[load_checkpoint] unexpected keys (skipped): {unexpected[:8]} total={len(unexpected)}")

    if optimizer is not None:
        opt_pt = in_path / "optimizer.pt"
        if not opt_pt.is_file():
            raise FileNotFoundError(f"missing optimizer.pt under {in_path}")
        optimizer.load_state_dict(torch.load(opt_pt, map_location="cpu", weights_only=False))

    if meta_path.is_file():
        return json.loads(meta_path.read_text())
    return {}


def load_pretrain_with_da_row_expansion(
    in_dir: Union[str, Path],
    model: nn.Module,
    *,
    new_num_domains: int,
    init_strategy: str = "copy_row_1",
) -> Dict[str, Any]:
    """Resume from a v37-style checkpoint when the FT model has a larger
    ``num_domains`` than the saved one (e.g. 9 → 10 for adding LIBERO as
    domain 9).

    Per-domain weights live in ``nn.Embedding`` rows of:
      - 8 DomainAwareLinear instances (scene/proprio/wrist/action_decoder
        DA-2-MLPs, fc1+fc2 each = 8 linears, each has fc.weight + bias.weight)
      - SoftPromptHub (embedding.weight)
    Total: 17 row-expanded state_dict tensors. Other tensors are loaded
    normally with strict shape checks.

    init_strategy:
      - "copy_row_1": fill new rows by replicating row 1 from the source
        (taco_play / Franka EEF — closest to LIBERO Franka). Default.
      - "copy_row_<n>": replicate from a specific source row.
      - "random": leave new rows at the model's default init (don't touch).
      - "zero": zero-fill new rows.

    Returns the meta.json contents from the source ckpt (unchanged), so the
    caller can decide whether to merge / inherit it into the FT checkpoint.
    """
    in_path = Path(in_dir)
    if not in_path.is_dir():
        raise FileNotFoundError(f"checkpoint dir not found: {in_path}")
    model_pt = in_path / "model.pt"
    if not model_pt.is_file():
        raise FileNotFoundError(f"missing model.pt under {in_path}")
    meta_path = in_path / "meta.json"

    src_state = torch.load(model_pt, map_location="cpu", weights_only=True)
    dst_state = model.state_dict()

    # Resolve init_strategy → source row index (or special markers).
    copy_idx: Optional[int] = None
    if init_strategy.startswith("copy_row_"):
        copy_idx = int(init_strategy.removeprefix("copy_row_"))
    elif init_strategy not in ("random", "zero"):
        raise ValueError(
            f"unknown init_strategy={init_strategy!r}; expected "
            f"'copy_row_<n>' / 'random' / 'zero'"
        )

    expanded_keys: list = []
    skipped_keys: list = []
    for key, src_tensor in src_state.items():
        if key not in dst_state:
            # New key in source not present in FT model — skip with warning.
            skipped_keys.append(key)
            continue
        dst_tensor = dst_state[key]
        if src_tensor.shape == dst_tensor.shape:
            dst_state[key] = src_tensor
            continue
        # Shape mismatch: only allow row-axis (dim 0) expansion to new_num_domains.
        if (
            src_tensor.dim() >= 1
            and dst_tensor.dim() == src_tensor.dim()
            and src_tensor.shape[0] < new_num_domains
            and dst_tensor.shape[0] == new_num_domains
            and tuple(src_tensor.shape[1:]) == tuple(dst_tensor.shape[1:])
        ):
            old_n = src_tensor.shape[0]
            # Build expanded: copy ckpt rows 0..old_n-1, fill rows old_n..new_num_domains-1.
            expanded = dst_tensor.clone()  # default values from FT model's own init
            expanded[:old_n] = src_tensor
            if init_strategy == "random":
                pass  # keep model's default init for new rows
            elif init_strategy == "zero":
                expanded[old_n:].zero_()
            else:
                # copy_row_<n>
                if copy_idx is None or not (0 <= copy_idx < old_n):
                    raise ValueError(
                        f"init_strategy {init_strategy!r}: copy_idx={copy_idx} "
                        f"out of range [0, {old_n})"
                    )
                src_row = src_tensor[copy_idx]
                expanded[old_n:] = src_row.unsqueeze(0).expand(
                    new_num_domains - old_n, *src_row.shape
                ).clone()
            dst_state[key] = expanded
            expanded_keys.append((key, list(src_tensor.shape), list(dst_tensor.shape)))
        else:
            # Different shape mismatch — skip and let strict load surface error.
            skipped_keys.append(key)

    missing, unexpected = model.load_state_dict(dst_state, strict=True)
    print(
        f"[resume v37] expanded {len(expanded_keys)} per-domain rows "
        f"(strategy={init_strategy!r}); skipped {len(skipped_keys)} keys"
    )
    if expanded_keys:
        for k, src_shape, dst_shape in expanded_keys[:6]:
            print(f"    {k}: {src_shape} → {dst_shape}")
        if len(expanded_keys) > 6:
            print(f"    ... and {len(expanded_keys) - 6} more")
    if skipped_keys:
        print(f"    skipped (shape unsupported): {skipped_keys[:4]}")

    if meta_path.is_file():
        return json.loads(meta_path.read_text())
    return {}
