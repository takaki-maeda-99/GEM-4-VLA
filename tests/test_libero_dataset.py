import torch
from torch.utils.data import DataLoader
from vla_project.data.datasets.libero_dataset import SyntheticLIBEROBatchDataset
from vla_project.data.schema import validate_batch


def test_yields_valid_batch():
    ds = SyntheticLIBEROBatchDataset(length=8, prompt_max_len=10)
    dl = DataLoader(ds, batch_size=2, collate_fn=ds.collate_fn)
    batch = next(iter(dl))
    validate_batch(batch)
    assert batch["domain_id"].shape[0] == 2
