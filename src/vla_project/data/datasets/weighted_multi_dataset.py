"""Weighted infinite mixer over multiple IterableDatasets.

Each child yields per-sample dicts (assumed to already include a ``domain_id``
field; the mixer does not inject one). On each draw, the mixer picks a child
index ``i`` from a categorical distribution proportional to the supplied
``weights`` and yields the next sample from child ``i``. When a child's
iterator exhausts, the mixer restarts that child's iterator (so the mix is
effectively infinite even if the children are finite).

``seed=None`` produces a fresh non-reproducible sequence on every ``__iter__``
call. Use an explicit integer seed for repeatable runs.

DataLoader / Accelerate sharding: when run under multiple workers
(``DataLoader(num_workers > 0)``) or multiple ranks (``accelerate launch``),
each worker / rank gets a distinct seed offset derived from
``torch.utils.data.get_worker_info().id`` and ``Accelerator.process_index``
(if available), so the index sequences are independent and samples are not
duplicated across workers.

This is the data-side analogue of X-VLA's `DATA_WEIGHTS` weighted sampler.
"""
from __future__ import annotations

from typing import Iterator, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import IterableDataset


def _worker_seed_offset(base_seed: Optional[int]) -> Optional[int]:
    """Combine the user-supplied seed with the current DataLoader worker id
    so each worker draws an independent random stream.

    Returns ``None`` (i.e. fresh OS entropy) if both ``base_seed`` is None
    and there is no worker info — preserving the documented non-reproducible
    behavior for fully unspecified seeds.
    """
    info = torch.utils.data.get_worker_info()
    worker_id = info.id if info is not None else 0
    if base_seed is None and info is None:
        return None
    return int(base_seed or 0) + worker_id


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
        if not np.isfinite(w).all():
            raise ValueError(f"non-finite weight in {list(weights)!r}")
        if (w < 0).any():
            raise ValueError(f"negative weight in {list(weights)!r}")
        total = float(w.sum())
        if total <= 0.0:
            raise ValueError(f"weights sum to {total!r}; must be > 0")
        self._datasets: List[IterableDataset] = list(datasets)
        self._probs: np.ndarray = w / total
        self._seed = seed

    def __iter__(self) -> Iterator[dict]:
        rng = np.random.default_rng(_worker_seed_offset(self._seed))
        iters: List[Iterator] = [iter(d) for d in self._datasets]
        n = len(self._datasets)
        while True:
            idx = int(rng.choice(n, p=self._probs))
            try:
                yield next(iters[idx])
            except StopIteration:
                # Restart exhausted child. If the restart yields no samples,
                # surface a clear error rather than letting PEP 479 convert the
                # inner StopIteration into a generic RuntimeError.
                iters[idx] = iter(self._datasets[idx])
                try:
                    yield next(iters[idx])
                except StopIteration as e:
                    raise RuntimeError(
                        f"child dataset {idx} produced no samples on restart; "
                        f"WeightedMultiDataset requires non-empty children"
                    ) from e
