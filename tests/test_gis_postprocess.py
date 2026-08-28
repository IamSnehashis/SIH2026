"""
Unit tests for GIS Post-Processing, Blending & COG.
"""
import pytest
import numpy as np
from pathlib import Path
import rasterio
from rasterio.transform import Affine

from src.gis.postprocessor import SceneMosaicker, generate_blending_weights, postprocess_and_export
from src.gis.raster_utils import compute_sr_geotransform, write_sr_output
from src.gis.cog import export_as_cog


def test_blending_weights():
    w_cos = generate_blending_weights(64, 64, method="cosine")
    assert w_cos.shape == (64, 64)
    assert w_cos.max() <= 1.0
    assert w_cos.min() > 0.0

    w_lin = generate_blending_weights(64, 64, method="linear")
    assert w_lin.shape == (64, 64)


def test_compute_sr_geotransform():
    orig_affine = Affine(10.0, 0.0, 500000.0, 0.0, -10.0, 3000000.0)
    sr_affine = compute_sr_geotransform(orig_affine, scale_factor=4.0)
    assert sr_affine.a == 2.5
    assert sr_affine.e == -2.5
    assert sr_affine.c == 500000.0
    assert sr_affine.f == 3000000.0


def test_scene_mosaicker():
    mosaicker = SceneMosaicker(full_height=128, full_width=128, channels=3, scale_factor=2.0)
    
    patch1 = np.ones((3, 128, 128), dtype=np.float32) * 0.5
    patch2 = np.ones((3, 128, 128), dtype=np.float32) * 0.8
    
    mosaicker.add_tile(patch1, orig_x=0, orig_y=0)
    mosaicker.add_tile(patch2, orig_x=32, orig_y=32)
    
    mosaic = mosaicker.finalize()
    assert mosaic.shape == (3, 256, 256)
    assert not np.isnan(mosaic).any()


def test_postprocess_and_export_geotiff(tmp_path):
    arr = np.random.uniform(0.0, 1.0, size=(4, 128, 128)).astype(np.float32)
    profile = {
        "driver": "GTiff",
        "height": 64,
        "width": 64,
        "count": 4,
        "dtype": "float32",
        "crs": "EPSG:4326",
        "transform": Affine(0.001, 0.0, 78.0, 0.0, -0.001, 27.0),
    }
    out_tif = tmp_path / "sr_out.tif"
    postprocess_and_export(arr, profile, out_tif, scale_factor=2.0)
    
    assert out_tif.exists()
    with rasterio.open(out_tif) as src:
        assert src.width == 128
        assert src.height == 128
        assert src.count == 4
        assert src.transform.a == 0.0005
