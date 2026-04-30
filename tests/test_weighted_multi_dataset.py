"""Stochastic tests for WeightedMultiDataset.

Uses two trivial in-memory IterableDatasets that yield deterministic per-sample
domain_ids so we can statistically verify the mixer's draw distribution.
"""
from typing import Iterator, List

import numpy as np
import pytest
import torch
from torch.utils.data import IterableDataset

from vla_project.data.datasets.weighted_multi_dataset import WeightedMultiDataset


class _ConstDomainDataset(IterableDataset):
    """Yields {"domain_id": <fixed int>, "value": rand} forever."""

    def __init__(self, domain_id: int) -> None:
        super().__init__()
        self.domain_id = domain_id

    def __iter__(self) -> Iterator[dict]:
        while True:
            yield {
                "domain_id": torch.tensor(self.domain_id, dtype=torch.long),
                "value": torch.randn(1),
            }


def _draw_domain_ids(mix: WeightedMultiDataset, n: int) -> List[int]:
    out: List[int] = []
    it = iter(mix)
    for _ in range(n):
        s = next(it)
        out.append(int(s["domain_id"].item()))
    return out


def test_weights_one_zero_returns_only_first() -> None:
    mix = WeightedMultiDataset(
        datasets=[_ConstDomainDataset(0), _ConstDomainDataset(1)],
        weights=[1.0, 0.0],
        seed=0,
    )
    ids = _draw_domain_ids(mix, n=200)
    assert set(ids) == {0}


def test_equal_weights_split_roughly_half() -> None:
    mix = WeightedMultiDataset(
        datasets=[_ConstDomainDataset(0), _ConstDomainDataset(1)],
        weights=[1.0, 1.0],
        seed=0,
    )
    ids = _draw_domain_ids(mix, n=2000)
    frac = sum(1 for x in ids if x == 0) / len(ids)
    # Allow generous tolerance — this is a smoke test, not a chi-squared.
    assert 0.45 <= frac <= 0.55


def test_three_to_one_weights_match() -> None:
    mix = WeightedMultiDataset(
        datasets=[_ConstDomainDataset(0), _ConstDomainDataset(1)],
        weights=[3.0, 1.0],
        seed=0,
    )
    ids = _draw_domain_ids(mix, n=4000)
    frac0 = sum(1 for x in ids if x == 0) / len(ids)
    assert 0.70 <= frac0 <= 0.80


def test_seed_reproducible() -> None:
    a = _draw_domain_ids(
        WeightedMultiDataset(
            datasets=[_ConstDomainDataset(0), _ConstDomainDataset(1)],
            weights=[1.0, 1.0],
            seed=42,
        ),
        n=100,
    )
    b = _draw_domain_ids(
        WeightedMultiDataset(
            datasets=[_ConstDomainDataset(0), _ConstDomainDataset(1)],
            weights=[1.0, 1.0],
            seed=42,
        ),
        n=100,
    )
    assert a == b


def test_rejects_empty_datasets() -> None:
    with pytest.raises(ValueError):
        WeightedMultiDataset(datasets=[], weights=[])


def test_rejects_weight_length_mismatch() -> None:
    with pytest.raises(ValueError):
        WeightedMultiDataset(
            datasets=[_ConstDomainDataset(0), _ConstDomainDataset(1)],
            weights=[1.0],
        )


def test_rejects_negative_weights() -> None:
    with pytest.raises(ValueError):
        WeightedMultiDataset(
            datasets=[_ConstDomainDataset(0), _ConstDomainDataset(1)],
            weights=[1.0, -1.0],
        )


def test_rejects_zero_total_weight() -> None:
    with pytest.raises(ValueError):
        WeightedMultiDataset(
            datasets=[_ConstDomainDataset(0), _ConstDomainDataset(1)],
            weights=[0.0, 0.0],
        )
