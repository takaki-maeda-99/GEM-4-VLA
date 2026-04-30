"""Weighted infinite mixer over multiple IterableDatasets.

Each child yields per-sample dicts (assumed to already include a ``domain_id``
field; the mixer does not inject one). On each draw, the mixer picks a child
index ``i`` from a categorical distribution proportional to the supplied
``weights`` and yields the next sample from child ``i``. When a child's
iterator exhausts, the mixer restarts that child's iterator (so the mix is
effectively infinite even if the children are finite).

This is the data-side analogue of X-VLA's `DATA_WEIGHTS` weighted sampler.
"""
from __future__ import annotations

from typing import Iterator, List, Optional, Sequence

import numpy as np
from torch.utils.data import IterableDataset


class WeightedMultiDataset(IterableDataset):
    def __init__(
        self,
        datasets: Sequence[IterableDataset],
        weights: Sequence[float],
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()
        if len(datasets) == 0:
            raise ValueError("datasets is empty")
        if len(datasets) != len(weights):
            raise ValueError(
                f"len(weights)={len(weights)} != len(datasets)={len(datasets)}"
            )
        w = np.asarray(weights, dtype=np.float64)
        if (w < 0).any():
            raise ValueError(f"negative weight in {list(weights)!r}")
        total = float(w.sum())
        if total <= 0.0:
            raise ValueError(f"weights sum to {total!r}; must be > 0")
        self._datasets: List[IterableDataset] = list(datasets)
        self._probs: np.ndarray = w / total
        self._seed = seed

    def __iter__(self) -> Iterator[dict]:
        rng = np.random.default_rng(self._seed)
        iters: List[Iterator] = [iter(d) for d in self._datasets]
        n = len(self._datasets)
        while True:
            idx = int(rng.choice(n, p=self._probs))
            try:
                yield next(iters[idx])
            except StopIteration:
                iters[idx] = iter(self._datasets[idx])
                yield next(iters[idx])
