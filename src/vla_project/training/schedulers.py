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

    Schedule (with `step` 0-indexed and `total_steps == max_steps`):
      - ``step ∈ [0, freeze_steps)``                             -> lr = 0
      - ``step ∈ [freeze_steps, freeze_steps + warmup_steps)``   -> linear ramp 0 -> base_lr
      - ``step ∈ [freeze_steps + warmup_steps, total_steps)``    -> cosine decay base_lr -> base_lr * min_lr_ratio

    The cosine decay denominator is ``total_steps - freeze_steps - warmup_steps``
    so the schedule ends exactly at ``step == total_steps`` regardless of how
    many freeze / warmup steps were used. Earlier versions used
    ``total_steps - warmup_steps`` which silently shortened the decay when
    ``freeze_steps > 0``.
    """
    if step < freeze_steps:
        return 0.0
    s = step - freeze_steps
    if s < warmup_steps:
        return base_lr * (s / max(warmup_steps, 1))
    progress = (s - warmup_steps) / max(total_steps - freeze_steps - warmup_steps, 1)
    progress = min(progress, 1.0)
    cos = 0.5 * (1.0 + math.cos(math.pi * progress))
    return base_lr * (min_lr_ratio + (1.0 - min_lr_ratio) * cos)
