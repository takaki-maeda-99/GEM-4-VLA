import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode


class SiglipImageTransform(nn.Module):
    """SigLIP-So400m expects 224x224, normalized by SigLIP statistics."""

    MEAN = (0.5, 0.5, 0.5)
    STD = (0.5, 0.5, 0.5)

    def __init__(self, size: int = 224, training: bool = False) -> None:
        super().__init__()
        ops = [T.Resize((size, size), interpolation=InterpolationMode.BICUBIC)]
        if training:
            ops.append(T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.0))
        ops.append(T.Normalize(self.MEAN, self.STD))
        self.transform = T.Compose(ops)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        return self.transform(img.float())
