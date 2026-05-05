"""Accelerate construction helpers."""
from __future__ import annotations


def default_ddp_kwargs_handlers():
    """Return kwargs handlers used by this project's DDP training path.

    Mode-B / baseline-compatible configs freeze the LLM and may bypass
    ``wrist_proj`` via wrist bridge, so some registered parameters do not
    participate in every backward pass. DDP needs ``find_unused_parameters``
    enabled for those runs.
    """
    from accelerate.utils import DistributedDataParallelKwargs

    return [DistributedDataParallelKwargs(find_unused_parameters=True)]
