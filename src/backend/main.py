"""
FastAPI Backend & Web GIS Service for Satellite Super-Resolution Mapping.

Endpoints:
- GET  /                -> Interactive Web GIS Viewer Studio
- POST /super-resolve   -> Upload GeoTIFF, returns super-resolved GeoTIFF
- POST /api/run-pipeline-> Executes full scene tiling, super-resolution, blending, index & vector extraction
- POST /api/sample      -> Generates synthetic Sentinel-2 GeoTIFF scene
- GET  /api/metadata    -> Inspects geospatial metadata of any GeoTIFF
- GET  /health          -> Service health check
"""
from pathlib import Path
from typing import Optional, Dict, Any
import shutil
import tempfile
import numpy as np
from fastapi import FastAPI, File, UploadFile, Query, HTTPException, Body
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.gis.raster_utils import read_raster, get_raster_metadata
from src.gis.sample_data import generate_synthetic_satellite_scene
from src.gis.pipeline import GISPipeline
from src.utils.config import load_config

app = FastAPI(
    title="Satellite Super-Resolution Mapping (SRM) GIS Studio",
    description="GIS Pre-Processing & Post-Processing Subsystem for Remote Sensing Imagery",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Web GIS Viewer
viewer_path = Path(__file__).resolve().parent.parent / "viewer"
if viewer_path.exists():
    app.mount("/viewer", StaticFiles(directory=str(viewer_path)), name="viewer")


@app.get("/", response_class=HTMLResponse)
def index_page():
    index_file = viewer_path / "index.html"
    if index_file.exists():
        return HTMLResponse(content=index_file.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>Satellite SRM GIS Studio Backend Running</h1>")


@app.get("/health")
def health_check():
    cfg = load_config()
    weights_path = Path(cfg["paths"]["models_dir"]) / cfg["dl"]["checkpoint_name"]
    return {
        "status": "healthy",
        "model_checkpoint_present": weights_path.exists(),
        "gis_engine": "Rasterio / GDAL",
        "scale_factor": cfg["gis"]["scale_factor"],
    }


class PipelineRequest(BaseModel):
    scale_factor: float = 4.0
    tile_size: int = 128
    overlap: int = 16
    blend_method: str = "cosine"
    input_path: Optional[str] = None


@app.post("/api/run-pipeline")
def run_pipeline_api(req: PipelineRequest = Body(...)):
    """
    Run end-to-end GIS super-resolution pipeline on default or supplied scene.
    """
    cfg = load_config()
    input_file = Path(req.input_path) if req.input_path else Path(cfg["paths"]["raw_dir"]) / "sample_scene.tif"

    if not input_file.exists():
        input_file.parent.mkdir(parents=True, exist_ok=True)
        generate_synthetic_satellite_scene(input_file, height=512, width=512)

    output_sr = Path(cfg["paths"]["outputs_dir"]) / "sr_mosaic.tif"
    weights_path = Path(cfg["paths"]["models_dir"]) / cfg["dl"]["checkpoint_name"]

    pipe = GISPipeline(
        tile_size=req.tile_size,
        overlap=req.overlap,
        scale_factor=req.scale_factor,
        blend_method=req.blend_method,
    )

    results = pipe.run_full_scene(
        input_scene_path=input_file,
        output_sr_path=output_sr,
        weights_path=weights_path if weights_path.exists() else None,
        generate_indices=True,
        generate_vectors=True,
    )

    return JSONResponse(results)


@app.post("/api/sample")
def generate_sample_api():
    """
    Create synthetic 4-band satellite scene at data/raw/sample_scene.tif.
    """
    cfg = load_config()
    raw_dir = Path(cfg["paths"]["raw_dir"])
    out_path = raw_dir / "sample_scene.tif"
    generate_synthetic_satellite_scene(out_path, height=512, width=512)
    meta = get_raster_metadata(out_path)
    return {"message": "Generated synthetic scene", "path": str(out_path), "metadata": meta}


@app.get("/api/metadata")
def inspect_raster_metadata(path: str = Query(..., description="Path to GeoTIFF")):
    p = Path(path)
    if not p.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return get_raster_metadata(p)


@app.post("/super-resolve")
async def super_resolve_upload(file: UploadFile = File(...)):
    """
    Direct file upload -> Super-Resolve -> Return GeoTIFF download.
    """
    cfg = load_config()
    with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp_in:
        shutil.copyfileobj(file.file, tmp_in)
        tmp_in_path = Path(tmp_in.name)

    tmp_out_path = tmp_in_path.parent / f"sr_{tmp_in_path.name}"
    weights_path = Path(cfg["paths"]["models_dir"]) / cfg["dl"]["checkpoint_name"]

    pipe = GISPipeline(
        tile_size=cfg["gis"]["tile_size"],
        overlap=cfg["gis"]["overlap"],
        scale_factor=cfg["gis"]["scale_factor"],
    )

    try:
        pipe.run_full_scene(
            input_scene_path=tmp_in_path,
            output_sr_path=tmp_out_path,
            weights_path=weights_path if weights_path.exists() else None,
            generate_indices=False,
            generate_vectors=False,
        )
        return FileResponse(
            path=str(tmp_out_path),
            filename=f"super_resolved_{file.filename}",
            media_type="image/tiff",
        )
    finally:
        if tmp_in_path.exists():
            tmp_in_path.unlink()
