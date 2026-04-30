import torch

from vla_project.utils.seed import set_seed
from vla_project.utils.io import load_yaml, save_yaml


def test_set_seed_makes_torch_deterministic(tmp_path):
    set_seed(42)
    a = torch.randn(3)
    set_seed(42)
    b = torch.randn(3)
    assert torch.equal(a, b)


def test_yaml_roundtrip(tmp_path):
    cfg = {"a": 1, "b": [2, 3], "c": {"d": "e"}}
    path = tmp_path / "x.yaml"
    save_yaml(cfg, path)
    assert load_yaml(path) == cfg
