"""
Unit tests for Spectral Indices, Vectorization & Metrics.
"""
import pytest
import numpy as np
from pathlib import Path
from rasterio.transform import Affine

from src.gis.indices import (
    calculate_ndvi,
    calculate_ndwi,
    calculate_ndbi,
    calculate_evi,
    compute_spectral_indices,
)
from src.gis.vectorizer import vectorize_binary_mask, extract_features_from_indices
from src.gis.metrics import (
    calculate_psnr,
    calculate_ssim,
    calculate_sam,
    calculate_ergas,
    evaluate_super_resolution,
)


def test_spectral_indices():
    # 4-band synthetic array: Blue, Green, Red, NIR
    arr = np.zeros((4, 32, 32), dtype=np.float32)
    arr[0] = 0.1  # Blue
    arr[1] = 0.2  # Green
    arr[2] = 0.1  # Red
    arr[3] = 0.8  # NIR (High vegetation)

    indices = compute_spectral_indices(arr)
    assert "ndvi" in indices
    assert "ndwi" in indices
    assert "evi" in indices

    # NDVI should be strongly positive for high NIR vs Red
    assert indices["ndvi"].mean() > 0.6
    # NDWI should be negative for high NIR vs Green
    assert indices["ndwi"].mean() < 0.0


def test_vectorization():
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[10:30, 10:30] = 1  # 20x20 square = 400 px

    transform = Affine(10.0, 0.0, 500000.0, 0.0, -10.0, 3000000.0)
    geojson_doc = vectorize_binary_mask(mask, transform, crs="EPSG:32643", min_area_pixels=10)

    assert geojson_doc["type"] == "FeatureCollection"
    assert len(geojson_doc["features"]) == 1
    feat = geojson_doc["features"][0]
    assert feat["properties"]["pixel_count"] == 400
    assert feat["properties"]["area_m2"] == 40000.0  # 400 * 100m2


def test_evaluation_metrics():
    target = np.random.uniform(0.1, 0.9, size=(4, 64, 64)).astype(np.float32)
    # Add slight perturbation
    pred = np.clip(target + np.random.normal(0, 0.01, target.shape), 0.0, 1.0).astype(np.float32)

    metrics = evaluate_super_resolution(pred, target, scale_factor=4.0)
    assert metrics["PSNR (dB)"] > 30.0
    assert metrics["SSIM"] > 0.90
    assert metrics["SAM (deg)"] < 5.0
    assert metrics["ERGAS"] < 5.0
