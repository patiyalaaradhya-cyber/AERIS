# AERIS — AI-Driven Spatio-Temporal Tracking of Extreme Weather Anomalies
**SIH Problem Statement 26078 Master Technical & Architectural Documentation**

---

## 1. Executive Summary

AERIS (**AI-Driven Spatio-Temporal Tracking of Extreme Weather Anomalies in Medium-Range Forecasts**) is a software platform designed to identify, track, and downscale extreme weather events (such as tropical cyclones, heavy precipitation anomalies, and pressure drops) across medium-range numerical weather prediction (NWP) forecasts.

Rather than relying on a single forecasting model, AERIS implements a **Harmonized Multi-Model Forecast Consensus Architecture** that standardizes independent NWP model outputs onto a common spatial-temporal grid before feeding the consensus field into a **Spherical Graph Neural Network (GNN)** for tracking and a **Conditional Diffusion Model** for localized high-resolution spatial downscaling.

---

## 2. System Architecture Overview

```
INDEPENDENT FORECAST SOURCE A          INDEPENDENT FORECAST SOURCE B
 (NCMRWF / NEPS-G 0.5° Grid)            (ECMWF IFS 0.25° -> 0.5° Grid)
             │                                        │
             └───────────────────┬────────────────────┘
                                 │
                   MODEL HARMONIZATION ENGINE
             (Grid Alignment, Unit Normalization: hPa, m/s)
                                 │
                     MULTI-MODEL CONSENSUS MEAN
                 + MODEL SPREAD (σ) CALCULATION
                                 │
               SPHERICAL GNN ANOMALY DETECTION
            (PyTorch Geometric, Geodesic K-NN Graph)
                                 │
           GEODESIC CENTROID & TRAJECTORY TRACKER
            (Displacement, Bearing, Translation Speed)
                                 │
            CONDITIONAL DIFFUSION SPATIAL DOWNSCALING
           (Coarse Forecast -> 5 km High-Res Hazard Map)
                                 │
               POPULATION EXPOSURE IMPACT ENGINE
                 & REAL-TIME GIS FRONTEND (Next.js)
```

---

## 3. Core Science & Implementation Modules

### 3.1 Multi-Model Harmonization & Consensus Engine
- **File**: `weather_core/analysis/multi_model_harmonizer.py`
- **Data Inputs**:
  - **NCMRWF NEPS-G**: Native 0.5° grid (83 × 125 lat-lon points), 12 ensemble members (`data/raw/nepsg/nepsg_amphan_2020.grib2`).
  - **ECMWF IFS**: High-resolution 0.25° grid regridded via bilinear spatial interpolation to the common 0.5° lat-lon grid (`data/raw/era5_amphan_2020.nc`).
- **Common Target Grid**: 83 × 125 Lat-Lon grid across the Bay of Bengal domain (0.37°N – 24.96°N, 80.10°E – 95.00°E).
- **Consensus Metrics**:
  - **Multi-Model Consensus Mean ($\mu$)**: Averaged across independent model fields per lead time step.
  - **Model Spread ($\sigma$)**: Standard deviation across independent models quantifying track and intensity forecast uncertainty.

### 3.2 Spherical GNN & Anomaly Centroid Tracking
- **File**: `weather_core/analysis/nepsg_phase12_pipeline.py` & `weather_core/gnn/models.py`
- **Architecture**: `WeatherGNNPredictor` using PyTorch Geometric spherical graph embeddings ($R = 6371\text{ km}$, $K=4$ nearest neighbors).
- **Geodesic Tracking**: Computes step-by-step centroid displacements, translation speeds ($\text{km/h}$), and navigation bearings ($\text{degrees}$) between consecutive lead times.

### 3.3 Conditional Diffusion Downscaling
- **File**: `weather_core/downscaling/nepsg_phase13_pipeline.py` & `weather_core/downscaling/diffusion.py`
- **Architecture**: Conditional U-Net Diffusion Model conditioned on Spherical GNN 64-dimensional event embeddings.
- **Resolution Output**: Downscales 0.5° (~55 km) coarse forecast inputs to a 5 km research output grid.

---

## 4. Live API & Frontend Reference

### 4.1 REST API Endpoints (FastAPI)
- `GET /api/v1/forecast/multimodel`: Returns active multi-model harmonization status, proof tables, and consensus stats.
- `GET /api/v1/forecast/field`: Serves dynamic raster field values for wind speed, MSL pressure, and precipitation.
- `GET /api/v1/events`: Serves detected extreme event diagnostic objects across lead times.
- `GET /api/v1/events/{event_id}/trajectory`: Returns GeoJSON line-string trajectory coordinates.
- `GET /api/v1/impact/exposure`: Computes localized population exposure estimation.

### 4.2 Dynamic Diagnostic Verification Table (Cyclone Amphan Case Study)

| Lead Step | Event Identifier | Min MSL Pressure | Max Wind Speed | Max Z-Score Anomaly | Footprint Area | Bearing / Direction | Translation Speed |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **+00h** | `AERIS_NCMRWF_AMPHAN_STEP_00H` | $985.2\text{ hPa}$ | $22.2\text{ m/s}$ | **`+6.56 σ`** | **`2,024,823 km²`** | **`0.0° (N/A)`** | **`0.0 km/h`** |
| **+06h** | `AERIS_NCMRWF_AMPHAN_STEP_06H` | $984.1\text{ hPa}$ | $22.1\text{ m/s}$ | **`+6.82 σ`** | **`1,624,443 km²`** | **`273.6° (W)`** | **`187.1 km/h`** |
| **+12h** | `AERIS_NCMRWF_AMPHAN_STEP_12H` | $980.5\text{ hPa}$ | $22.5\text{ m/s}$ | **`+6.51 σ`** | **`1,254,855 km²`** | **`21.8° (N/NE)`** | **`12.0 km/h`** |
| **+18h** | `AERIS_NCMRWF_AMPHAN_STEP_18H` | $978.7\text{ hPa}$ | $24.4\text{ m/s}$ | **`+7.03 σ`** | **`1,184,048 km²`** | **`80.1° (E/NE)`** | **`99.0 km/h`** |
| **+24h** | `AERIS_NCMRWF_AMPHAN_STEP_24H` | $973.2\text{ hPa}$ | $25.8\text{ m/s}$ | **`+6.83 σ`** | **`1,502,813 km²`** | **`273.8° (W)`** | **`88.8 km/h`** |
| **+36h** | `AERIS_NCMRWF_AMPHAN_STEP_36H` | $987.7\text{ hPa}$ | $23.5\text{ m/s}$ | **`+4.41 σ`** | **`10,700,097 km²`** | **`17.6° (N/NE)`** | **`58.3 km/h`** |
| **+42h** | `AERIS_NCMRWF_AMPHAN_STEP_42H` | $981.8\text{ hPa}$ | $26.1\text{ m/s}$ | **`+6.69 σ`** | **`2,069,178 km²`** | **`2.8° (N)`** | **`89.1 km/h`** |
| **+48h** | `AERIS_NCMRWF_AMPHAN_STEP_48H` | $977.8\text{ hPa}$ | $29.8\text{ m/s}$ | **`+7.25 σ`** | **`1,533,013 km²`** | **`11.2° (N)`** | **`11.2 km/h`** |
| **+54h** | `AERIS_NCMRWF_AMPHAN_STEP_54H` | $977.1\text{ hPa}$ | $33.1\text{ m/s}$ | **`+8.10 σ`** | **`1,519,541 km²`** | **`11.0° (N)`** | **`11.3 km/h`** |
| **+60h** | `AERIS_NCMRWF_AMPHAN_STEP_60H` | $977.2\text{ hPa}$ | $34.0\text{ m/s}$ | **`+7.65 σ`** | **`1,803,434 km²`** | **`21.2° (N/NE)`** | **`6.0 km/h`** |
| **+66h** | `AERIS_NCMRWF_AMPHAN_STEP_66H` | $973.7\text{ hPa}$ | $33.8\text{ m/s}$ | **`+9.15 σ`** | **`1,621,284 km²`** | **`10.9° (N)`** | **`11.3 km/h`** |
| **+72h** | `AERIS_NCMRWF_AMPHAN_STEP_72H` | $974.3\text{ hPa}$ | $33.9\text{ m/s}$ | **`+9.55 σ`** | **`1,457,257 km²`** | **`0.0° (N)`** | **`11.1 km/h`** |

---

## 5. Automated Test Suite Verification

- **Command**: `python -m unittest discover -s tests`
- **Pass Rate**: **`78 / 78 PASSED (100% OK)`**
- **Test Modules Covered**:
  - `test_anomaly.py`: Extreme weather anomaly thresholding and spatial cluster detection.
  - `test_climatology.py`: Long-term baseline mean and standard deviation computation.
  - `test_gnn.py`: Spherical graph construction and GNN forward inference passes.
  - `test_diffusion.py`: Conditional diffusion noise schedule and downscaling inference.
  - `test_nepsg_phase12.py`: Harmonized multi-model consensus tracking pipeline execution.
  - `test_nepsg_phase13.py`: Spatial downscaling pipeline evaluation.
  - `test_api.py`: FastAPI endpoint responses, REST contracts, and status handlers.

---

## 6. Repository File & Directory Structure

```
D:\SIH26078_AERIS\
├── AERIS_MASTER_DOCUMENTATION.md
├── START_AERIS.bat
├── .gitignore
├── backend/
│   ├── main.py                     # FastAPI application entrypoint
│   ├── api/v1/router.py            # API routers & endpoints
│   ├── schemas/models.py           # Pydantic data schemas
│   └── services/                   # Business logic layers (event, tracking, dataset)
├── frontend/
│   ├── app/                        # Next.js app pages (page.tsx, layout.tsx, globals.css)
│   ├── components/                 # React UI components (GISMap, EventDetailsPanel, etc.)
│   ├── lib/api.ts                  # Axios/Fetch API client functions
│   └── package.json                # Frontend dependencies
├── weather_core/
│   ├── analysis/                   # Multi-Model Harmonizer & Phase 12/13 pipelines
│   ├── anomaly/                    # Anomaly engine & spatial region detection
│   ├── climatology/                # Baseline statistical engine
│   ├── downscaling/                # Conditional Diffusion model & physics losses
│   ├── gnn/                        # PyTorch Geometric Spherical GNN models
│   ├── impact/                     # Population exposure estimation engine
│   ├── ingestion/                  # GRIB2 & NetCDF decoder modules
│   └── tracking/                   # Geodesic centroid tracker & extrapolation
├── scripts/                        # Launcher, demo runners, and evaluation utilities
├── tests/                          # 78 unit test suites
└── data/
    └── processed/                  # Lightweight GeoJSON & JSON demo manifests
```
