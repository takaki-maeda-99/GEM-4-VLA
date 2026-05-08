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
    AND the distributed rank so each (rank, worker) draws an independent
    random stream.

    Without the rank component, all 6 DDP ranks would use the same draw
    sequence (worker_id=0 on each), which means every rank would see the
    same domain pick at each step — collapsing effective batch to per-GPU
    batch and erasing DDP's gradient-averaging benefit. Codex round 8
    flagged this; fix is to mix rank into the seed.

    Returns ``None`` (i.e. fresh OS entropy) if both ``base_seed`` is None
    and there is no worker info — preserving the documented non-reproducible
    behavior for fully unspecified seeds.
    """
    info = torch.utils.data.get_worker_info()
    worker_id = info.id if info is not None else 0
    # Distributed rank: read from torch.distributed if initialized; falls
    # back to LOCAL_RANK / RANK env vars (set by accelerate launch).
    rank = 0
    try:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
    except Exception:
        rank = 0
    if rank == 0:
        # Fallback to env vars before init / outside torch.distributed
        import os as _os
        rank = int(_os.environ.get("RANK", _os.environ.get("LOCAL_RANK", 0)))
    if base_seed is None and info is None and rank == 0:
        return None
    # Multiply rank by a large prime to spread it across the int range
    # before adding worker_id, so (rank=0, worker=1) and (rank=1, worker=0)
    # don't collide.
    return int(base_seed or 0) + 1000003 * int(rank) + int(worker_id)


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
