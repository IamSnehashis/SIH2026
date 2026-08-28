"""Smoke tests for the GIS utilities — run with: pytest tests/"""
import numpy as np


def test_normalize_range():
    from src.gis.preprocessor import Normalizer
    arr = np.random.randint(0, 10000, size=(4, 32, 32)).astype(np.float32)
    norm = Normalizer(method="percentile")
    out = norm.fit_transform(arr)
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_model_forward_shape():
    import torch
    from src.dl.models import SRCNN
    model = SRCNN(in_channels=4)
    x = torch.rand(1, 4, 64, 64)
    y = model(x)
    assert y.shape == x.shape
