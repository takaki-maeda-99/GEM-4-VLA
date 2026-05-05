"""ModelRuntime — Phase 0 stub.

Phase 0: loads meta.json (so xvla_adapter startup validation can run) but
__call__(batch) raises NotImplementedError. Phase 1 fills in the torch
forward path per spec §Section 5 ModelRuntime.

The classmethod from_export(ckpt_dir) is the canonical entry; tests assert
its behavior on synthetic meta.json fixtures.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class MetaJsonError(Exception):
    """meta.json missing, malformed, or missing required keys."""


class ModelRuntime:
    def __init__(self, *, step: int, cfg: dict, norm_stats: dict, ckpt_dir: Path) -> None:
        self.step = step
        self.cfg = cfg
        self.norm_stats = norm_stats
        self.ckpt_dir = ckpt_dir

    @classmethod
    def from_export(
        cls,
        ckpt_dir: str | Path,
        *,
        device: str = "cuda:0",
        dtype: str = "bf16",
        torch_compile: str = "off",
        warmup_iters: int = 1,
    ) -> "ModelRuntime":
        ckpt_dir = Path(ckpt_dir)
        meta_path = ckpt_dir / "meta.json"
        if not meta_path.is_file():
            raise MetaJsonError(f"missing meta.json under {ckpt_dir}")
        meta = json.loads(meta_path.read_text())
        for required_key in ("step", "cfg", "norm_stats"):
            if required_key not in meta:
                raise MetaJsonError(f"meta.json missing required key {required_key!r}")
        # Phase 0 ignores device / dtype / torch_compile / warmup_iters; Phase 1 wires them.
        _ = (device, dtype, torch_compile, warmup_iters)
        return cls(
            step=int(meta["step"]),
            cfg=meta["cfg"],
            norm_stats=meta["norm_stats"],
            ckpt_dir=ckpt_dir,
        )

    def __call__(self, batch: dict[str, Any]) -> Any:
        raise NotImplementedError(
            "ModelRuntime forward path is Phase 1 work. "
            "See docs/superpowers/specs/2026-05-06-vla-inference-server-design.md "
            "§Section 5 ModelRuntime."
        )
