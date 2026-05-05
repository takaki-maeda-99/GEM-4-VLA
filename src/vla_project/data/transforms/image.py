import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode


class SiglipImageTransform(nn.Module):
    """SigLIP-So400m timm-default transform.

    Pipeline (matches vla-gemma-4 73% baseline + timm SigLIP default):
      Resize(248, bicubic, antialias=True) -> CenterCrop(224) -> Normalize(0.5, 0.5)

    Earlier versions used a direct Resize(224) which gave a different field
    of view than the reference (248-then-crop slightly zooms in, dropping
    edge pixels). For SigLIP features to match the reference, both train
    and eval images must traverse the same crop pipeline.
    """

    MEAN = (0.5, 0.5, 0.5)
    STD = (0.5, 0.5, 0.5)
    RESIZE_SIZE = 248

    def __init__(self, size: int = 224, training: bool = False) -> None:
        super().__init__()
        ops = [
            T.Resize(
                self.RESIZE_SIZE,
                interpolation=InterpolationMode.BICUBIC,
                antialias=True,
            ),
            T.CenterCrop(size),
        ]
        if training:
            ops.append(T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.0))
        ops.append(T.Normalize(self.MEAN, self.STD))
        self.transform = T.Compose(ops)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        return self.transform(img.float())


class DINOv2ImageTransform(nn.Module):
    """DINOv2/ImageNet preprocessing for 224x224 ViT/14 patch tokens."""

    MEAN = (0.485, 0.456, 0.406)
    STD = (0.229, 0.224, 0.225)

    def __init__(self, size: int = 224) -> None:
        super().__init__()
        self.transform = T.Compose([
            T.Resize(
                size,
                interpolation=InterpolationMode.BICUBIC,
                antialias=True,
            ),
            T.CenterCrop(size),
            T.Normalize(self.MEAN, self.STD),
        ])

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        return self.transform(img.float())
