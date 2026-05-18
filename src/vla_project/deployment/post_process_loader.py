"""Loader for <ckpt_dir>/post_process.py — per-checkpoint action post-processing.

Trust model: model.pt loads with weights_only=True (no RCE), so
post_process.py IS a new RCE surface. Local paths load by default with
a WARN log; HF-resolved paths require explicit --trust-checkpoint-code.
See spec §6 'Trust model'.
"""
from __future__ import annotations

import importlib
import logging
import sys
from pathlib import Path
from typing import Callable

logger = logging.getLogger("vla_project.deployment.post_process_loader")


class HardFailAssertion(Exception):
    """Raised when post_process.py is malformed and the server must not start."""


def load_post_process(
    ckpt_dir: Path,
    *,
    is_local: bool,
    trust_checkpoint_code: bool,
) -> Callable | None:
    """Load post_process.apply from <ckpt_dir>/post_process.py.

    Returns the apply callable, or None if the file is absent or the
    HF trust gate is not opened. Raises HardFailAssertion on malformed
    file (ImportError, missing apply, etc.).
    """
    pp_file = Path(ckpt_dir) / "post_process.py"
    if not pp_file.is_file():
        return None
    if not is_local and not trust_checkpoint_code:
        logger.warning(
            f"{pp_file} present but skipped: ckpt was HF-resolved and "
            f"--trust-checkpoint-code was not passed. Actions returned "
            f"WITHOUT post-processing."
        )
        return None
    sys.path.insert(0, str(ckpt_dir))
    try:
        if "post_process" in sys.modules:
            del sys.modules["post_process"]
        try:
            mod = importlib.import_module("post_process")
        except Exception as e:
            raise HardFailAssertion(
                f"failed to import {pp_file}: {type(e).__name__}: {e}"
            ) from e
        fn = getattr(mod, "apply", None)
        if not callable(fn):
            raise HardFailAssertion(
                f"{pp_file} missing callable apply(actions, meta)"
            )
        logger.warning(
            f"loaded executable post_process from ckpt: {pp_file}. "
            f"This file runs with full server privileges."
        )
        return fn
    finally:
        sys.path.pop(0)
