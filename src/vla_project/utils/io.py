from pathlib import Path
from typing import Any

from omegaconf import OmegaConf


def load_yaml(path: str | Path) -> Any:
    return OmegaConf.to_container(OmegaConf.load(str(path)), resolve=True)


def save_yaml(obj: Any, path: str | Path) -> None:
    OmegaConf.save(OmegaConf.create(obj), str(path))
