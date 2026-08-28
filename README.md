# SIH 2026 - Deep Learning Based Satellite Super-Resolution Mapping (SRM)

A comprehensive, production-grade GIS pre-processing and post-processing subsystem for satellite imagery super-resolution.

---

## 🌟 GIS Subsystem Architecture

```
                       ┌──────────────────────────────────────────────┐
                       │           RAW SATELLITE GEOTIFF             │
                       │    (Sentinel-2, Landsat-8, OLI2MSI, etc.)    │
                       └──────────────────────┬───────────────────────┘
                                              │
                    ┌─────────────────────────▼─────────────────────────┐
                    │        1. GIS PRE-PROCESSING ENGINE              │
                    │   • Multi-Sensor Ingestion & CRS Validation       │
                    │   • Radiometric TOA / Reflectance Normalization  │
                    │   • NoData / Cloud Masking & Manifest Logging     │
                    │   • Sliding-Window Scene Tiling with Overlap      │
                    │   • Synthetic Sensor Degradation / Real Pairing   │
                    └─────────────────────────┬─────────────────────────┘
                                              │
                                  [Normalized Tile Patches]
                                              │
                    ┌─────────────────────────▼─────────────────────────┐
                    │        2. MACHINE LEARNING SUPER-RESOLUTION      │
                    │   • SRCNN / EDSR / RCAN / SwinIR Upsampling       │
                    └─────────────────────────┬─────────────────────────┘
                                              │
                               [Super-Resolved Patches]
                                              │
                    ┌─────────────────────────▼─────────────────────────┐
                    │        3. GIS POST-PROCESSING ENGINE             │
                    │   • 2D Cosine / Gaussian Overlap Blending         │
                    │   • Spatial Resolution & Geotransform Rescaling   │
                    │   • Radiometric De-normalization                  │
                    │   • Cloud-Optimized GeoTIFF (COG) & Overviews     │
                    └─────────────────────────┬─────────────────────────┘
                                              │
                    ┌─────────────────────────▼─────────────────────────┐
                    │     4. DOWNSTREAM GIS & REMOTE SENSING PRODUCTS  │
                    │   • Biophysical Indices (NDVI, NDWI, NDBI, EVI)  │
                    │   • Vector Feature Extraction (GeoJSON Polygons)  │
                    │   • Quantitative Metrics (SAM, ERGAS, PSNR, SSIM) │
                    │   • Interactive Web GIS Studio Viewer (Leaflet)   │
                    └───────────────────────────────────────────────────┘
```

---

## 📁 Package Structure

```
srm-satellite-gis/
├── src/
│   ├── gis/
│   │   ├── __init__.py           # Export high-level GIS API
│   │   ├── raster_utils.py       # Raster I/O, CRS reprojection, georeferenced export
│   │   ├── preprocessor.py       # Normalizer (percentile, minmax, reflectance, zscore)
│   │   ├── tiling.py             # Sliding-window tiling with overlap & GeoJSON footprints
│   │   ├── pair_generator.py     # Sensor PSF blur, noise degradation, training pair generation
│   │   ├── postprocessor.py      # Seamless mosaicker with 2D cosine/gaussian overlap blending
│   │   ├── indices.py            # Spectral index engine (NDVI, NDWI, NDBI, EVI, SAVI)
│   │   ├── vectorizer.py         # Vector polygon extraction (GeoJSON water/veg/urban masks)
│   │   ├── metrics.py            # RS fidelity metrics (SAM, ERGAS, UIQI, PSNR, SSIM)
│   │   ├── cog.py                # Cloud-Optimized GeoTIFF & image pyramid generator
│   │   ├── sample_data.py        # Synthetic multi-band satellite scene generator
│   │   ├── pipeline.py           # End-to-end full scene GIS pipeline orchestrator
│   │   └── cli.py                # Unified command-line interface
│   │
│   ├── dl/                       # PyTorch models (SRCNN, EDSR), dataset loader, training
│   ├── backend/                  # FastAPI service with Web GIS studio endpoints
│   ├── viewer/                   # Interactive Leaflet Web GIS studio (split-slider, indices)
│   └── utils/                    # Shared YAML config loader
│
├── data/                         # Raw scenes, tiles, samples
├── outputs/                      # Super-resolved GeoTIFFs & downstream products
└── tests/                        # Full automated test suite (15 unit & integration tests)
```

---

## 🚀 Quickstart & Usage

### 1. Generate Synthetic Sample Scene
```bash
python -m src.gis.cli sample --output data/raw/sample_scene.tif
```

### 2. GIS Pre-Processing (Tiling & Training Pair Generator)
```bash
python -m src.gis.cli preprocess --input data/raw --output data/tiles --scale 4
```

### 3. Run Full End-to-End GIS Pipeline
```bash
python -m src.gis.cli pipeline --input data/raw/sample_scene.tif --output outputs/sr_scene.tif --scale 4 --indices --vectorize
```

### 4. Calculate Spectral Vegetation, Water & Urban Indices
```bash
python -m src.gis.cli indices --input outputs/sr_scene.tif --output outputs/indices
```

### 5. Extract Vector Polygons (GeoJSON)
```bash
python -m src.gis.cli vectorize --input outputs/sr_scene.tif --output outputs/vectors
```

### 6. Evaluate Remote Sensing Metrics vs Ground Truth HR
```bash
python -m src.gis.cli evaluate --pred outputs/sr_scene.tif --target data/raw/ground_truth_hr.tif --scale 4 --output outputs/evaluation
```

### 7. Run Interactive Web GIS Studio
```bash
uvicorn src.backend.main:app --reload --port 8000
```
Open `http://localhost:8000` in your browser to view the interactive map with split comparison slider, live index inspector, and vector layers!

---

## 🧪 Run Automated Tests

```bash
pytest tests/ -v
```
All 15 unit and integration tests verify raster I/O, geotransform scaling, sliding window tiling, overlap blending, spectral indices, vector polygon extraction, metrics (SAM, ERGAS, PSNR, SSIM), and full pipeline execution.
# Deep Learning Based Super Resolution Mapping (SRM) from Medium Resolution Satellite Imagery

## 🚀 Smart India Hackathon 2026 Prototype

A Deep Learning and GIS-based system for enhancing the spatial resolution of medium-resolution satellite imagery using Super Resolution techniques.

The proposed system takes medium-resolution satellite imagery as input, applies a deep learning-based Super Resolution model, and generates an enhanced high-resolution representation while preserving important spatial and geographic information.

---

## 📌 Problem Statement

Medium-resolution satellite imagery is widely available and useful for applications such as:

* Urban planning
* Agriculture monitoring
* Disaster management
* Infrastructure mapping
* Environmental monitoring
* Land-use and land-cover analysis

However, the spatial resolution of freely available satellite imagery can limit detailed analysis.

High-resolution satellite imagery provides better spatial detail but can be expensive, difficult to access, or unavailable for certain regions and time periods.

### Our Objective

Develop a prototype that uses **Deep Learning-based Super Resolution** to enhance medium-resolution satellite imagery and produce a higher-resolution representation that can be visualized and analyzed through a GIS interface.

---

# 🎯 Project Objectives

1. Accept medium-resolution satellite imagery as input.
2. Preprocess and normalize the imagery.
3. Apply a Deep Learning-based Super Resolution model.
4. Generate an enhanced-resolution image.
5. Preserve relevant geospatial information where applicable.
6. Visualize original and enhanced imagery on a GIS-enabled interface.
7. Calculate image-quality metrics such as PSNR and SSIM.
8. Provide an intuitive web-based interface for demonstration.
9. Demonstrate the potential applications of the system in remote sensing and geospatial analysis.

---

# 🏗️ System Architecture

```text
                         USER
                           │
                           ▼
                ┌───────────────────┐
                │    FRONTEND       │
                │                   │
                │ Upload Satellite  │
                │ Image / GeoTIFF   │
                └─────────┬─────────┘
                          │
                          ▼
                ┌───────────────────┐
                │     BACKEND       │
                │    FastAPI        │
                └─────────┬─────────┘
                          │
              ┌───────────┴───────────┐
              │                       │
              ▼                       ▼
      ┌─────────────────┐     ┌─────────────────┐
      │   ML PIPELINE   │     │   GIS PIPELINE  │
      │                 │     │                 │
      │ Preprocessing   │     │ Raster Handling │
      │       ↓         │     │ GeoTIFF         │
      │ SR Model        │     │ Geospatial Data│
      │       ↓         │     │ Map Layers      │
      │ Enhanced Image  │     │                 │
      └────────┬────────┘     └────────┬────────┘
               │                       │
               └───────────┬───────────┘
                           ▼
                ┌─────────────────────┐
                │   RESULT / MAP      │
                │                     │
                │ Original vs SR      │
                │ PSNR / SSIM         │
                │ Resolution Info     │
                └─────────────────────┘
```

---

# 🔄 End-to-End Workflow

```text
Satellite Image
       │
       ▼
Input Validation
       │
       ▼
Image Preprocessing
       │
       ▼
Low Resolution Image
       │
       ▼
Deep Learning
Super Resolution Model
       │
       ▼
Super Resolved Image
       │
       ├──────────────► Quality Evaluation
       │                PSNR / SSIM / RMSE
       │
       ▼
Geospatial Processing
       │
       ▼
GIS Visualization
       │
       ▼
Web Dashboard
```

---

# 🧠 Deep Learning Component

The ML component performs image Super Resolution.

### Pipeline

```text
Input Image
     ↓
Preprocessing
     ↓
Tensor Conversion
     ↓
Super Resolution Model
     ↓
Post-processing
     ↓
Enhanced Image
```

The prototype can use an established Super Resolution architecture such as:

* SRCNN
* ESPCN
* EDSR
* Real-ESRGAN

The final architecture may be selected based on available computational resources, dataset characteristics, and prototype requirements.

---

# 📊 Model Evaluation

The generated Super Resolution output is evaluated against a reference high-resolution image when a suitable reference is available.

### Metrics

#### PSNR — Peak Signal-to-Noise Ratio

Measures reconstruction quality based on pixel-level error.

Higher PSNR generally indicates better reconstruction quality.

#### SSIM — Structural Similarity Index

Measures structural similarity between the generated image and reference image.

SSIM ranges approximately from:

```text
0 → Low similarity
1 → Very high similarity
```

#### RMSE — Root Mean Square Error

Measures the magnitude of reconstruction error.

Lower RMSE indicates lower error.

---

# 🗺️ GIS / Remote Sensing Component

The GIS component handles geospatial imagery and visualization.

Key responsibilities include:

* Raster image processing
* GeoTIFF handling
* Coordinate Reference Systems
* Geospatial metadata
* Raster visualization
* Original vs enhanced map comparison

Potential technologies include:

* QGIS
* Rasterio
* GDAL
* GeoPandas
* Leaflet
* OpenLayers

---

# 🖥️ Web Interface

The frontend provides a simple interface for demonstrating the complete workflow.

### Main Features

* Satellite image upload
* Input image preview
* Processing button
* Processing status
* Super-resolved image preview
* Original vs enhanced comparison
* GIS map visualization
* PSNR / SSIM results
* Download generated output

Example interface:

```text
┌────────────────────────────────────────────┐
│          SATELLITE SRM SYSTEM              │
│                                            │
│  Deep Learning Based Super Resolution      │
│                                            │
│       [ Upload Satellite Image ]           │
│                                            │
│              [ PROCESS ]                   │
│                                            │
├────────────────────────────────────────────┤
│                                            │
│     ORIGINAL          SUPER RESOLVED       │
│                                            │
│      [MAP]                [MAP]            │
│                                            │
├────────────────────────────────────────────┤
│                                            │
│  PSNR: XX dB       SSIM: XX                │
│                                            │
│          [ Download Result ]               │
│                                            │
└────────────────────────────────────────────┘
```

---

# ⚙️ Technology Stack

## Machine Learning

* Python
* PyTorch
* NumPy
* OpenCV
* Pillow
* Scikit-image

## Backend

* Python
* FastAPI
* Uvicorn

## Frontend

* React
* Vite
* JavaScript
* HTML
* CSS

## GIS / Remote Sensing

* QGIS
* Rasterio
* GDAL
* GeoTIFF
* Leaflet / OpenLayers

## Development & Collaboration

* Git
* GitHub
* Visual Studio Code
* Jupyter Notebook (for experimentation)

---

# 📁 Project Structure

```text
SIH-SRM-2026/
│
├── frontend/
│   ├── src/
│   ├── public/
│   ├── package.json
│   └── README.md
│
├── backend/
│   ├── main.py
│   ├── routes/
│   ├── services/
│   ├── uploads/
│   ├── outputs/
│   └── requirements.txt
│
├── ml/
│   ├── model/
│   │   └── sr_model.py
│   │
│   ├── inference/
│   │   └── predict.py
│   │
│   ├── preprocessing/
│   │   └── preprocess.py
│   │
│   ├── evaluation/
│   │   └── metrics.py
│   │
│   ├── weights/
│   │   └── model.pth
│   │
│   └── requirements.txt
│
├── gis/
│   ├── raster/
│   ├── processing/
│   ├── visualization/
│   └── README.md
│
├── evaluation/
│   ├── datasets/
│   ├── results/
│   └── reports/
│
├── docs/
│   ├── architecture/
│   ├── diagrams/
│   └── presentation/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── sample/
│
├── .gitignore
├── README.md
└── LICENSE
```

---

# 👥 Team Responsibilities

| Member   | Role                        | Responsibilities                              |
| -------- | --------------------------- | --------------------------------------------- |
| Snehashis Mandal | ML / Deep Learning          | Model, preprocessing, inference               |
| Aritra Dutta | Frontend                    | Web interface and visualization, GitHub              |
| Asmita Hajra | Backend                     | FastAPI, APIs, ML integration                 |
| Aritra Saha | GIS / Remote Sensing        | Satellite data, GeoTIFF, maps                 |
| Shreya Mahato | Data & Evaluation           | Dataset, PSNR, SSIM, comparison               |
| Priya | Integration & Documentation |  integration, testing, PPT, deployment |

---

# 🔌 API Design

The backend provides communication between the frontend, ML pipeline, and GIS components.

### Upload Image

```http
POST /upload
```

Uploads the satellite image to the backend.

---

### Process Image

```http
POST /process
```

Runs the Super Resolution pipeline.

Example response:

```json
{
    "status": "success",
    "result": "output/sr_image.tif",
    "psnr": 32.4,
    "ssim": 0.91,
    "processing_time": 4.8
}
```

---

### Get Result

```http
GET /result/{id}
```

Returns the generated Super Resolution image.

---

### Get Metrics

```http
GET /metrics/{id}
```

Returns evaluation metrics.

---

# 🛠️ Installation

## 1. Clone the Repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>

cd SIH-SRM-2026
```

---

# 🧠 ML Setup

Create a virtual environment:

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r ml/requirements.txt
```

Example dependencies:

```text
torch
torchvision
numpy
opencv-python
Pillow
scikit-image
rasterio
```

---

# ⚙️ Backend Setup

Navigate to the backend:

```bash
cd backend
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the FastAPI server:

```bash
uvicorn main:app --reload
```

The backend will be available locally at:

```text
http://127.0.0.1:8000
```

---

# 🖥️ Frontend Setup

Navigate to the frontend:

```bash
cd frontend
```

Install dependencies:

```bash
npm install
```

Start the development server:

```bash
npm run dev
```

---

# 🧪 Running the ML Model

Example:

```bash
python ml/inference/predict.py \
    --input data/sample/input.tif \
    --output data/sample/output.tif
```

The generated image will be stored in the specified output location.

---

# 📈 Prototype Demonstration

The final demonstration follows this sequence:

```text
1. Open Web Application
          ↓
2. Upload Satellite Image
          ↓
3. Click "Process"
          ↓
4. Backend receives image
          ↓
5. ML model performs Super Resolution
          ↓
6. Enhanced image is generated
          ↓
7. GIS layer is prepared
          ↓
8. Original and enhanced images displayed
          ↓
9. PSNR / SSIM displayed
          ↓
10. User can download the result
```

---

# 🔬 Example Comparison

The system aims to provide a comparison such as:

```text
              INPUT                  OUTPUT

        Medium Resolution       Super Resolution
             Image                    Image

          ┌──────────┐             ┌──────────┐
          │          │             │          │
          │ Satellite│    ───►     │ Enhanced │
          │  Image   │             │  Image   │
          │          │             │          │
          └──────────┘             └──────────┘

          Lower Spatial            Higher
           Detail                  Detail
```

---

# 🌍 Potential Applications

The technology can support several remote-sensing applications:

### 🏙️ Urban Planning

Improved visualization of:

* Roads
* Buildings
* Urban expansion
* Infrastructure

### 🌾 Agriculture

Potential applications include:

* Crop monitoring
* Field boundary analysis
* Vegetation analysis
* Agricultural planning

### 🌊 Disaster Management

Enhanced imagery may assist in:

* Flood mapping
* Damage assessment
* Disaster response
* Change detection

### 🌳 Environmental Monitoring

Potential use cases include:

* Deforestation monitoring
* Land-use analysis
* Water-body monitoring
* Environmental change detection

---

# ⚠️ Limitations

Super Resolution does not create genuinely captured information that was absent from the original imagery.

The model learns patterns from training data to reconstruct a higher-resolution representation.

Therefore:

* Generated details may contain reconstruction artifacts.
* Results depend on training data.
* Different geographic regions may produce different results.
* Quantitative evaluation requires suitable reference imagery.
* Super-resolved outputs should not automatically be treated as ground truth.

The prototype is intended as a research and decision-support demonstration rather than a replacement for actual high-resolution satellite acquisition.

---

# 🔮 Future Scope

Future versions can include:

* Multi-spectral Super Resolution
* Hyperspectral Super Resolution
* Temporal satellite image fusion
* Cloud removal
* Change detection
* Land-use / land-cover classification
* Object detection
* Edge-device inference
* Larger regional datasets
* Improved geospatial accuracy
* Uncertainty estimation
* Transformer-based Super Resolution
* GAN-based approaches
* Real-time satellite data integration

---

# 🔐 Data & Privacy

The prototype should use publicly available or appropriately licensed satellite datasets.

Do not upload confidential, restricted, or personally identifiable information to the system.

---

# 🧑‍💻 Development Workflow

All team members work through GitHub.

```text
                 GitHub Repository
                        │
        ┌───────────────┼────────────────┐
        │               │                │
       ML           Frontend          Backend
        │               │                │
        └───────────────┼────────────────┘
                        │
                    Integration
                        │
                 Testing / Demo
```

### Branches

```text
main
ml
frontend
backend
gis
evaluation
integration
```

### Basic Git workflow

```bash
git checkout -b feature-name

git add .

git commit -m "Describe your changes"

git push origin feature-name
```

Create a Pull Request before merging major changes into `main`.

---

# 📅 SIH Prototype Timeline

## August 25

* Set up GitHub
* Finalize architecture
* Install dependencies
* Assign responsibilities

## August 26

* ML prototype
* Frontend skeleton
* Backend API skeleton
* GIS setup
* Dataset preparation

## August 27

* First end-to-end integration
* Frontend ↔ Backend
* Backend ↔ Dummy ML
* GIS visualization

## August 28

* Connect actual Super Resolution model
* Generate real outputs
* Fix API integration

## August 29

* GIS integration
* PSNR / SSIM
* Before/after comparison
* Testing

## August 30

* UI polishing
* PPT
* Architecture diagram
* Demo preparation
* Backup video

## August 31

### SIH Demonstration

```text
Problem
   ↓
Solution
   ↓
Architecture
   ↓
Live Demo
   ↓
Results
   ↓
Impact
   ↓
Future Scope
```

---

# 🎯 Minimum Viable Prototype

The project is considered demo-ready when the following workflow works:

```text
┌───────────────────────────────┐
│     Upload Satellite Image    │
└───────────────┬───────────────┘
                ↓
┌───────────────────────────────┐
│       Backend Processing       │
└───────────────┬───────────────┘
                ↓
┌───────────────────────────────┐
│      Deep Learning Model      │
└───────────────┬───────────────┘
                ↓
┌───────────────────────────────┐
│     Super Resolution Output   │
└───────────────┬───────────────┘
                ↓
┌───────────────────────────────┐
│       GIS Visualization       │
└───────────────┬───────────────┘
                ↓
┌───────────────────────────────┐
│   PSNR / SSIM + Comparison    │
└───────────────────────────────┘
```

---

# ⭐ Key Value Proposition

> **Transform medium-resolution satellite imagery into a higher-resolution representation using Deep Learning while enabling geospatial visualization and quantitative evaluation through an integrated web-based platform.**

---

# 📜 License

This project is developed as a prototype for **Smart India Hackathon 2026**.

The license and usage terms for datasets, pretrained models, and third-party libraries must be respected separately.

---

# 👥 Team

**Smart India Hackathon 2026**

### Project

**Deep Learning Based Super Resolution Mapping (SRM) from Medium Resolution Satellite Imagery**

### Team Roles

* 🧠 ML / Deep Learning
* 🖥️ Frontend Development
* ⚙️ Backend Development
* 🗺️ GIS / Remote Sensing
* 📊 Data & Model Evaluation
* 🔗 Integration / Documentation

---

## 🚀 Status

**Prototype — Under Development**

The current development priority is to establish a functional end-to-end pipeline:

```text
Satellite Image
      ↓
Deep Learning
      ↓
Super Resolution
      ↓
GIS
      ↓
Web Interface
      ↓
Evaluation
```

---

**Built for Smart India Hackathon 2026 🇮🇳**
