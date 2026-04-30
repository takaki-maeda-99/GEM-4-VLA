import torch
from vla_project.data.transforms.image import SiglipImageTransform


def test_resize_and_normalize_shape():
    t = SiglipImageTransform(size=224, training=False)
    img = torch.zeros(3, 100, 100)
    out = t(img)
    assert out.shape == (3, 224, 224)
    assert out.dtype == torch.float32
