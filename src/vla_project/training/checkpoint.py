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
