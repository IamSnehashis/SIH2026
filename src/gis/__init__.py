"""
GIS Module for Satellite Super-Resolution Mapping (SRM).

Exports:
- raster_utils: read_raster, write_raster, write_sr_output, reproject_raster, get_raster_metadata
- preprocessor: Normalizer, preprocess_scene
- tiling: SceneTiler, TileMetadata
- pair_generator: generate_dataset_pairs_from_scenes, apply_sensor_degradation
- postprocessor: SceneMosaicker, postprocess_and_export, generate_blending_weights
- cog: export_as_cog, build_pyramid_overviews
- indices: compute_spectral_indices, calculate_ndvi, calculate_ndwi, calculate_ndbi, calculate_evi, calculate_savi
- vectorizer: vectorize_binary_mask, extract_features_from_indices
- metrics: evaluate_super_resolution, calculate_psnr, calculate_ssim, calculate_sam, calculate_ergas, calculate_uiqi
- pipeline: GISPipeline
"""
from src.gis.raster_utils import (
    read_raster,
    write_raster,
    write_sr_output,
    reproject_raster,
    get_raster_metadata,
    compute_sr_geotransform,
    check_spatial_compatibility,
)
from src.gis.preprocessor import Normalizer, preprocess_scene
from src.gis.tiling import SceneTiler, TileMetadata
from src.gis.pair_generator import generate_dataset_pairs_from_scenes, apply_sensor_degradation
from src.gis.postprocessor import SceneMosaicker, postprocess_and_export, generate_blending_weights
from src.gis.cog import export_as_cog, build_pyramid_overviews
from src.gis.indices import (
    compute_spectral_indices,
    calculate_ndvi,
    calculate_ndwi,
    calculate_ndbi,
    calculate_evi,
    calculate_savi,
    export_index_geotiff_and_png,
)
from src.gis.vectorizer import vectorize_binary_mask, extract_features_from_indices
from src.gis.metrics import (
    evaluate_super_resolution,
    calculate_psnr,
    calculate_ssim,
    calculate_sam,
    calculate_ergas,
    calculate_uiqi,
    generate_error_heatmap,
)
from src.gis.pipeline import GISPipeline

__all__ = [
    "read_raster",
    "write_raster",
    "write_sr_output",
    "reproject_raster",
    "get_raster_metadata",
    "compute_sr_geotransform",
    "check_spatial_compatibility",
    "Normalizer",
    "preprocess_scene",
    "SceneTiler",
    "TileMetadata",
    "generate_dataset_pairs_from_scenes",
    "apply_sensor_degradation",
    "SceneMosaicker",
    "postprocess_and_export",
    "generate_blending_weights",
    "export_as_cog",
    "build_pyramid_overviews",
    "compute_spectral_indices",
    "calculate_ndvi",
    "calculate_ndwi",
    "calculate_ndbi",
    "calculate_evi",
    "calculate_savi",
    "export_index_geotiff_and_png",
    "vectorize_binary_mask",
    "extract_features_from_indices",
    "evaluate_super_resolution",
    "calculate_psnr",
    "calculate_ssim",
    "calculate_sam",
    "calculate_ergas",
    "calculate_uiqi",
    "generate_error_heatmap",
    "GISPipeline",
]
