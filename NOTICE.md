# FastGeoCLIP — Modifications notice

FastGeoCLIP is derived from [GeoCLIP](https://github.com/VicenteVivan/geo-clip) by Vicente Vivanco (Copyright (c) 2024, MIT-licensed). The original LICENSE file is preserved in this repository unchanged.

This document summarises substantive changes vs. the original code. The full attribution requirement of the MIT License is satisfied by the retained LICENSE; this NOTICE is informational.

## Summary of changes

### 1. Vision-only, offline CLIP

The original `ImageEncoder` downloaded the full text+vision `CLIPModel` from HuggingFace at runtime. The fork uses `CLIPVisionModelWithProjection` loaded from a local `clip/clip-vit-large-vision/` directory (prepared once via `reduce_clip_size.py`). The path is configurable via `CLIP_MODEL_PATH`. This removes the text tower (unused for image→GPS), shrinks the artifact, and makes the service air-gapped friendly.

### 2. Pre-computed and cached GPS embeddings

The original `predict()` encoded all 100K gallery locations on every call. The fork computes the embeddings once in `__init__`, caches them to disk, and reuses them on subsequent starts. The cache directory is configurable via `CACHE_DIR`. Cache filenames embed an 8-hex fingerprint of (gallery CSV, `location_encoder_weights.pth`) so swapping either artifact rebuilds the cache automatically instead of silently loading stale embeddings.

### 3. FAISS nearest-neighbor search

The original used a full O(N) dot-product across the 100K gallery per request. The fork uses FAISS to narrow to `k_neighbors` candidates, then runs exact cosine scoring on those only. The index type is configurable via `FAISS_INDEX_TYPE` (`flatl2` — exact, default; `hnsw` or `ivf` — approximate). The built index is cached to disk alongside the embeddings.

### 4. Training code removed

The momentum GPS queue, the contrastive `forward()`, and the entire `train/` directory (dataloader, train loop, eval) were removed. This fork is inference-only.

### 5. Simplified RFF module

Replaced the 3-class `rff/` package with a single-class `rff.py` exposing only `GaussianEncoding` (the only encoding actually used by the location encoder). Math matches upstream exactly: `γ(v) = (cos(2π·v B^T), sin(2π·v B^T))`. Verified by comparing the location-encoder output element-wise against upstream's implementation — max absolute difference 3.7e-6, i.e. floating-point noise.

### 6. Extracted geographic utilities

`equal_earth_projection` and its constants moved from `location_encoder.py` / `misc.py` to a dedicated `geo_utils.py` with named constants (`EQUAL_EARTH_A1`, etc.).

### 7. Stdlib CSV instead of pandas

`load_gps_data` now uses the stdlib `csv` module instead of `pandas`. Drops the pandas dependency from the runtime image.

### 8. Package renamed

The Python package directory was renamed from `geoclip/model/` to `fastgeoclip/` for clarity at the import site (`from fastgeoclip import FastGeoCLIP`). The deployment-only directories `clip/` (vision tower) and `data/` (GeoPackage) live at the repo root, separate from the package.

### 9. FastAPI service layer

Added `app.py`: a FastAPI service exposing `/predict` and `/lookup`. PyTorch inference runs in a thread (`asyncio.to_thread`) so the event loop stays free for concurrent calls. Uploads are size-capped via `MAX_UPLOAD_MB`.

### 10. Docker deployment

Added `Dockerfile` and an example `docker-compose.example.yml`. The runtime image installs CPU-only PyTorch and `faiss-cpu`, bakes in the fine-tuned weights + GPS gallery, and mounts the CLIP tower + GeoPackage at runtime. Reverse-geocoding uses the local GeoPackage rather than calling an external service.

## What was *not* changed

The fine-tuned weights (`image_encoder_mlp_weights.pth`, `location_encoder_weights.pth`, `logit_scale_weights.pth`) and the 100K GPS gallery CSV are the original GeoCLIP artifacts, redistributed unmodified inside `fastgeoclip/weights/` and `fastgeoclip/gps_gallery/` — same as upstream.

The model architecture, the Equal Earth projection math, the random Fourier feature layer (math), and the contrastive prediction logic are unchanged.
