from vla_project.training.schedulers import linear_warmup_cosine


def test_warmup_then_decay():
    base_lr = 1.0
    total = 100
    warmup = 10
    # at step 0: ~0
    assert linear_warmup_cosine(0, freeze_steps=0, warmup_steps=warmup,
                                 total_steps=total, base_lr=base_lr,
                                 min_lr_ratio=0.1) == 0.0
    # at warmup boundary: ~base_lr
    lr_warm = linear_warmup_cosine(warmup, 0, warmup, total, base_lr, 0.1)
    assert abs(lr_warm - base_lr) < 1e-6
    # at end: min_lr
    lr_end = linear_warmup_cosine(total, 0, warmup, total, base_lr, 0.1)
    assert abs(lr_end - 0.1 * base_lr) < 1e-6


def test_freeze_then_warmup_then_decay_reaches_floor() -> None:
    """With freeze_steps > 0, cosine still decays to min_lr_ratio at step=total_steps.
    Regression test: earlier denom used (total - warmup) instead of
    (total - freeze - warmup), so freeze>0 silently shortened decay."""
    base_lr = 1.0
    total = 1000
    freeze = 100
    warmup = 200
    assert linear_warmup_cosine(0, freeze, warmup, total, base_lr, 0.1) == 0.0
    assert linear_warmup_cosine(freeze - 1, freeze, warmup, total, base_lr, 0.1) == 0.0
    lr_peak = linear_warmup_cosine(freeze + warmup, freeze, warmup, total, base_lr, 0.1)
    assert abs(lr_peak - base_lr) < 1e-6
    lr_end = linear_warmup_cosine(total, freeze, warmup, total, base_lr, 0.1)
    assert abs(lr_end - 0.1 * base_lr) < 1e-6
    half = freeze + warmup + (total - freeze - warmup) // 2
    lr_half = linear_warmup_cosine(half, freeze, warmup, total, base_lr, 0.1)
    assert 0.1 * base_lr < lr_half < base_lr
