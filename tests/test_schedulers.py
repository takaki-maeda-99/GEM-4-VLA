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
