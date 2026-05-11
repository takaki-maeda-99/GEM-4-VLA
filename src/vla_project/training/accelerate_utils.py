"""Accelerate construction helpers."""
from __future__ import annotations


def default_ddp_kwargs_handlers(find_unused_parameters: bool = True):
    """Return kwargs handlers used by this project's DDP training path.

    Mode-B / baseline-compatible configs freeze the LLM and may bypass
    ``wrist_proj`` via wrist bridge, so some registered parameters do not
    participate in every backward pass. Those configs need
    ``find_unused_parameters=True``. v37 OXE pretrain has all DA paths active
    on every step (every domain hits scene/proprio/wrist/action_decoder), so
    setting ``find_unused_parameters=False`` is safe and saves the per-step
    autograd-graph traversal that otherwise costs 20-30%.

    v37 bumped ProcessGroupNCCL timeout from accelerate's default 10 min to
    30 min. dl40 6-GPU config spans NUMA 0+1 (GPU 2-7 cross CPU sockets via
    SYS interconnect), so init broadcasts can be slow. Without this bump we
    hit 10-min timeouts on the first parameter sync (137M numel) before any
    training step runs.
    v39 (2026-05-10): bumped 30 → 60 min. The 13-domain LIBERO+OXE pretrain
    triggers a brand-new torch.compile shape graph (multi-source RLDS sees
    13 distinct sample shape variants across ranks during shuffle-buffer
    fill). First-step compile observed at 24 min and still progressing on
    rank 1/3 while rank 0 had completed and entered the first ALLREDUCE —
    classic compile-stagger desync that 30 min timeout couldn't absorb.
    """
    from datetime import timedelta
    from accelerate.utils import DistributedDataParallelKwargs, InitProcessGroupKwargs

    return [
        DistributedDataParallelKwargs(find_unused_parameters=bool(find_unused_parameters)),
        InitProcessGroupKwargs(timeout=timedelta(minutes=60)),
    ]
