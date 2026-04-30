import math


def linear_warmup_cosine(
    step: int,
    freeze_steps: int,
    warmup_steps: int,
    total_steps: int,
    base_lr: float,
    min_lr_ratio: float,
) -> float:
    """Linear warmup over `warmup_steps`, then cosine decay to `min_lr_ratio * base_lr`.

    `freeze_steps` are steps before training starts (LR=0).
    """
    if step < freeze_steps:
        return 0.0
    s = step - freeze_steps
    if s < warmup_steps:
        return base_lr * (s / max(warmup_steps, 1))
    progress = (s - warmup_steps) / max(total_steps - warmup_steps, 1)
    progress = min(progress, 1.0)
    cos = 0.5 * (1.0 + math.cos(math.pi * progress))
    return base_lr * (min_lr_ratio + (1.0 - min_lr_ratio) * cos)
