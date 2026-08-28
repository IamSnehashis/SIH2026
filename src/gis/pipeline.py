"""
End-to-End GIS Pipeline Orchestrator for Satellite Super-Resolution Mapping.

Connects:
1. GIS Pre-Processing (Raster Ingestion, Normalization, Sliding Window Tiling)
2. ML Super-Resolution Inference (PyTorch Model or Bicubic Benchmark)
3. GIS Post-Processing (Seamless Overlap Blending, Geo-Registration, Spatial Rescaling, COG Export)
4. Downstream GIS Products (NDVI/NDWI Indices, GeoJSON Vector Extraction, SAM/ERGAS/PSNR Metrics)
"""
from pathlib import Path
from typing import Optional, Union, Tuple, Dict, Any, List, Callable
import json
import time
import numpy as np
import torch
import cv2

from src.gis.raster_utils import read_raster, get_raster_metadata, write_sr_output
from src.gis.preprocessor import Normalizer, preprocess_scene
from src.gis.tiling import SceneTiler
from src.gis.postprocessor import SceneMosaicker, postprocess_and_export
from src.gis.cog import export_as_cog
from src.gis.indices import compute_spectral_indices, export_index_geotiff_and_png
from src.gis.vectorizer import extract_features_from_indices
from src.gis.metrics import evaluate_super_resolution, generate_error_heatmap


class GISPipeline:
    """
    High-level orchestrator for full-scene satellite super-resolution workflows.
    """

    def __init__(
        self,
        tile_size: int = 128,
        overlap: int = 16,
        scale_factor: float = 4.0,
        blend_method: str = "cosine",
        norm_method: str = "percentile",
        device: str = "auto",
    ):
        self.tile_size = tile_size
        self.overlap = overlap
        self.scale_factor = scale_factor
        self.blend_method = blend_method
        self.norm_method = norm_method

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

    def run_full_scene(
        self,
        input_scene_path: Union[str, Path],
        output_sr_path: Union[str, Path],
        model: Optional[Any] = None,
        weights_path: Optional[Union[str, Path]] = None,
        bands: Optional[List[int]] = None,
        generate_indices: bool = True,
        generate_vectors: bool = True,
        ground_truth_hr_path: Optional[Union[str, Path]] = None,
        export_cog_format: bool = True,
    ) -> Dict[str, Any]:
        """
        Execute the full GIS super-resolution mapping pipeline on a satellite scene.
        """
        t0 = time.time()
        input_scene_path = Path(input_scene_path)
        output_sr_path = Path(output_sr_path)
        output_sr_path.parent.mkdir(parents=True, exist_ok=True)
        products_dir = output_sr_path.parent / f"{output_sr_path.stem}_products"
        products_dir.mkdir(parents=True, exist_ok=True)

        results = {
            "input_path": str(input_scene_path),
            "output_sr_path": str(output_sr_path),
            "scale_factor": self.scale_factor,
            "tile_size": self.tile_size,
            "overlap": self.overlap,
            "blend_method": self.blend_method,
            "stages": {},
        }

        # -------------------------------------------------------------
        # 1. GIS PRE-PROCESSING
        # -------------------------------------------------------------
        t_pre = time.time()
        arr_norm, profile, normalizer, valid_mask = preprocess_scene(
            input_scene_path, bands=bands, method=self.norm_method
        )
        c, h, w = arr_norm.shape
        normalizer.export_manifest(products_dir / "normalization_manifest.json")

        tiler = SceneTiler(tile_size=self.tile_size, overlap=self.overlap, min_valid_ratio=0.1)
        tiles = tiler.extract_tiles_from_array(
            arr_norm, profile, scene_name=input_scene_path.stem, mask=valid_mask
        )
        tiler.save_tile_catalogue(
            [meta for _, meta in tiles], products_dir, basename="tiling"
        )
        results["stages"]["preprocessing"] = {
            "duration_sec": round(time.time() - t_pre, 3),
            "input_shape": [c, h, w],
            "num_tiles": len(tiles),
        }

        # -------------------------------------------------------------
        # 2. MODEL INFERENCE / PATCH UPSAMPLING
        # -------------------------------------------------------------
        t_inf = time.time()
        mosaicker = SceneMosaicker(
            full_height=h,
            full_width=w,
            channels=c,
            scale_factor=self.scale_factor,
            blend_method=self.blend_method,
        )

        # Load PyTorch model if provided
        py_model = None
        if model is not None:
            py_model = model.to(self.device)
            py_model.eval()
        elif weights_path is not None and Path(weights_path).exists():
            from src.dl.models import build_model
            from src.utils.config import load_config
            cfg = load_config()
            py_model = build_model(cfg).to(self.device)
            py_model.load_state_dict(torch.load(weights_path, map_location=self.device))
            py_model.eval()

        for patch, meta in tiles:
            # Model prediction or bicubic upsampling benchmark
            if py_model is not None:
                # Prepare tensor: if model expects pre-upsampled input
                patch_up = np.zeros((c, int(self.tile_size * self.scale_factor), int(self.tile_size * self.scale_factor)), dtype=np.float32)
                for ch in range(c):
                    patch_up[ch] = cv2.resize(patch[ch], (patch_up.shape[2], patch_up.shape[1]), interpolation=cv2.INTER_CUBIC)
                
                inp_tensor = torch.from_numpy(patch_up).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    pred_tensor = py_model(inp_tensor).squeeze(0).clamp(0.0, 1.0)
                    sr_patch = pred_tensor.cpu().numpy()
            else:
                # High-fidelity bicubic interpolation baseline
                target_th = int(round(self.tile_size * self.scale_factor))
                target_tw = int(round(self.tile_size * self.scale_factor))
                sr_patch = np.zeros((c, target_th, target_tw), dtype=np.float32)
                for ch in range(c):
                    sr_patch[ch] = cv2.resize(patch[ch], (target_tw, target_th), interpolation=cv2.INTER_CUBIC)

            # Add to seamless mosaicker
            mosaicker.add_tile(sr_patch, meta.x, meta.y)

        # Finalize seamless mosaic
        sr_mosaic = mosaicker.finalize()
        results["stages"]["inference"] = {
            "duration_sec": round(time.time() - t_inf, 3),
            "output_shape": list(sr_mosaic.shape),
            "model_type": "PyTorch DL" if py_model is not None else "Bicubic Interpolation Baseline",
        }

        # -------------------------------------------------------------
        # 3. GIS POST-PROCESSING & GEO-REGISTRATION
        # -------------------------------------------------------------
        t_post = time.time()
        raw_sr_tif = products_dir / "sr_uncompressed.tif"
        postprocess_and_export(
            sr_array=sr_mosaic,
            reference_profile=profile,
            output_path=raw_sr_tif,
            scale_factor=self.scale_factor,
            normalizer=normalizer,
            output_dtype="float32",
        )

        if export_cog_format:
            export_as_cog(raw_sr_tif, output_sr_path)
            if raw_sr_tif.exists():
                raw_sr_tif.unlink()
        else:
            raw_sr_tif.rename(output_sr_path)

        sr_meta = get_raster_metadata(output_sr_path)
        results["stages"]["postprocessing"] = {
            "duration_sec": round(time.time() - t_post, 3),
            "sr_resolution": sr_meta["resolution"],
            "sr_bounds": sr_meta["bounds"],
            "sr_crs": sr_meta["crs"],
        }

        # -------------------------------------------------------------
        # 4. DOWNSTREAM GIS ANALYSIS (INDICES & VECTORS)
        # -------------------------------------------------------------
        if generate_indices or generate_vectors:
            t_down = time.time()
            sr_indices = compute_spectral_indices(sr_mosaic)
            results["indices"] = list(sr_indices.keys())

            if generate_indices:
                idx_dir = products_dir / "indices"
                for idx_name, idx_arr in sr_indices.items():
                    cmap = "RdYlGn" if "veg" in idx_name or "ndvi" in idx_name or "evi" in idx_name else "Blues" if "water" in idx_name or "ndwi" in idx_name else "YlOrRd"
                    export_index_geotiff_and_png(idx_arr, sr_meta, idx_dir, index_name=idx_name, colormap_name=cmap)

            if generate_vectors:
                vec_dir = products_dir / "vectors"
                vectors = extract_features_from_indices(
                    sr_indices,
                    transform=sr_meta["transform"],
                    crs=sr_meta["crs"],
                    output_dir=vec_dir
                )
                results["extracted_vectors"] = {k: v["properties"]["total_features"] for k, v in vectors.items()}

            results["stages"]["downstream_analysis"] = {
                "duration_sec": round(time.time() - t_down, 3)
            }

        # -------------------------------------------------------------
        # 5. METRIC EVALUATION (IF GROUND TRUTH HR PROVIDED)
        # -------------------------------------------------------------
        if ground_truth_hr_path and Path(ground_truth_hr_path).exists():
            hr_arr, hr_profile, _ = read_raster(ground_truth_hr_path, bands=bands)
            hr_norm = normalizer.fit_transform(hr_arr)

            # Match dimensions if slight rounding differences
            min_h = min(sr_mosaic.shape[1], hr_norm.shape[1])
            min_w = min(sr_mosaic.shape[2], hr_norm.shape[2])

            metrics = evaluate_super_resolution(
                sr_mosaic[:, :min_h, :min_w],
                hr_norm[:, :min_h, :min_w],
                scale_factor=self.scale_factor
            )
            results["metrics"] = metrics

            # Heatmap
            generate_error_heatmap(
                sr_mosaic[:, :min_h, :min_w],
                hr_norm[:, :min_h, :min_w],
                sr_meta,
                output_dir=products_dir / "evaluation",
            )

        results["total_elapsed_sec"] = round(time.time() - t0, 3)

        # Save summary report
        with open(products_dir / "pipeline_summary.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        return results
