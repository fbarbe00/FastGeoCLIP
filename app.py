import asyncio
import io
import os
import time
from datetime import datetime
from functools import lru_cache
import warnings

import geopandas as gpd
import torch
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from pydantic import BaseModel
from contextlib import asynccontextmanager
from PIL import Image
from shapely.geometry import Point

from fastgeoclip.FastGeoCLIP import FastGeoCLIP

# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------

DATA_GPKG = os.getenv("DATA_GPKG", "/app/data/admin1_clean.gpkg")
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_MB", "25")) * 1024 * 1024
warnings.filterwarnings("ignore", message=".*geographic CRS.*")


# ----------------------------------------------------------------------------
# APP + MODEL SETUP
# ----------------------------------------------------------------------------

model = None
device = "cpu"
regions = None
sindex = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, regions, sindex
    torch.set_grad_enabled(False)
    torch.set_num_threads(int(os.getenv("NUM_THREADS", "4")))
    k_neighbors = int(os.getenv("K_NEIGHBORS", "1000"))
    index_type = os.getenv("FAISS_INDEX_TYPE", "flatl2")

    # Load models
    print("[startup] loading FastGeoCLIP model...")
    model = FastGeoCLIP(k_neighbors=k_neighbors, index_type=index_type)
    model.to(device).eval()
    print(f"[startup] FastGeoCLIP configured with k_neighbors={k_neighbors}, index={index_type}")

    # Load geolocation regions (optional — /lookup will be disabled if missing).
    if os.path.isfile(DATA_GPKG):
        print("[startup] loading regions...")
        t0 = time.time()
        try:
            regions = gpd.read_file(DATA_GPKG, layer="regions")
            sindex = regions.sindex
            print(f"[startup] regions loaded in {time.time() - t0:.3f}s")
        except Exception as exc:
            print(f"[startup] failed to load {DATA_GPKG}: {exc}; /lookup will be disabled")
            regions = None
            sindex = None
    else:
        print(f"[startup] {DATA_GPKG} not found; /lookup will be disabled. "
              f"See README for how to prepare admin1_clean.gpkg.")

    yield

app = FastAPI(
    title="GeoCLIP API",
    version="1.0",
    lifespan=lifespan,
)

# ----------------------------------------------------------------------------
# GEOLOCATION UTILITIES
# ----------------------------------------------------------------------------

# tolerance in degrees (~0.01° ≈ 1.1km at equator)
TOL = 0.03
# Maximum distance threshold in degrees (~0.1° ≈ 11km at equator)
# This prevents matching ocean points to very distant land regions
MAX_DIST_DEG = 0.1

@lru_cache(maxsize=10_000)
def _lookup_cached(lat_r: float, lon_r: float):
    pt = Point(lon_r, lat_r)

    # First: try exact contains (fast)
    idxs = list(sindex.intersection(pt.bounds))
    if idxs:
        cand = regions.iloc[idxs]
        hit = cand[cand.contains(pt)]
        if not hit.empty:
            row = hit.iloc[0]
            return row["name_en"], row["admin"], row["iso_a2"]

    # Second: try nearest within tolerance
    tol_bounds = (
        lon_r - TOL, lat_r - TOL,
        lon_r + TOL, lat_r + TOL
    )

    idxs = list(sindex.intersection(tol_bounds))
    if not idxs:
        return None, None, None

    cand = regions.iloc[idxs].copy()  # Add .copy() to avoid SettingWithCopyWarning

    # compute distances and pick closest
    cand["dist"] = cand.geometry.distance(pt)
    nearest = cand.sort_values("dist").iloc[0]

    # Only return a match if it's reasonably close (within MAX_DIST_DEG)
    # This prevents ocean points from being matched to distant land regions
    if nearest["dist"] > MAX_DIST_DEG:
        return None, None, None

    return nearest["name_en"], nearest["admin"], nearest["iso_a2"]


def lookup_geo(lat: float, lon: float):
    """Lookup region and country for a given lat/lon coordinate."""
    region, country, iso_code = _lookup_cached(round(lat, 4), round(lon, 4))
    return region, country, iso_code

# ----------------------------------------------------------------------------
# RESPONSE MODELS
# ----------------------------------------------------------------------------

class Prediction(BaseModel):
    rank: int
    latitude: float
    longitude: float
    probability: float

class PredictionResponse(BaseModel):
    predictions: list[Prediction]
    processing_time_ms: float

class LookupRequest(BaseModel):
    lat: float
    lon: float

class LookupResponse(BaseModel):
    region: str | None
    country: str | None
    iso_code: str | None

# ----------------------------------------------------------------------------
# PREDICTION ENDPOINT
# ----------------------------------------------------------------------------

@app.post("/predict", response_model=PredictionResponse)
async def predict(
    file: UploadFile = File(...),
    top_k: int = Query(5, ge=1, le=100),
):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    start = datetime.now()

    # Read in chunks with a hard cap to avoid OOM/DoS from oversized uploads.
    # Content-Length is not trusted; enforce the limit while reading.
    chunks = []
    total = 0
    while True:
        chunk = await file.read(1024 * 1024)
        if not chunk:
            break
        total += len(chunk)
        if total > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail=f"Image exceeds max size of {MAX_UPLOAD_BYTES // (1024 * 1024)}MB")
        chunks.append(chunk)
    image_bytes = b"".join(chunks)

    try:
        # Fail fast on non-image bytes (Image.open is lazy; .verify() consumes
        # the parser and raises on corruption without decoding the full image).
        try:
            Image.open(io.BytesIO(image_bytes)).verify()
        except Exception:
            raise HTTPException(status_code=400, detail="File is not a valid image")

        # Run CPU-bound inference in a thread so the event loop stays unblocked
        # (allowing /lookup requests to be served concurrently)
        gps, probs = await asyncio.to_thread(
            model.predict, io.BytesIO(image_bytes), top_k
        )

        predictions = [
            Prediction(
                rank=i + 1,
                latitude=float(lat),
                longitude=float(lon),
                probability=float(prob),
            )
            for i, ((lat, lon), prob) in enumerate(zip(gps, probs))
        ]

        elapsed = (datetime.now() - start).total_seconds() * 1000

        return PredictionResponse(
            predictions=predictions,
            processing_time_ms=elapsed,
        )

    except HTTPException:
        raise
    except Exception as e:
        # Don't leak internal exception text (file paths, torch internals) to
        # the caller; surface a generic message and log the detail server-side.
        print(f"[predict] internal error: {type(e).__name__}: {e}", flush=True)
        raise HTTPException(status_code=500, detail="Prediction failed")

# ----------------------------------------------------------------------------
# GEOLOCATION ENDPOINT
# ----------------------------------------------------------------------------

@app.post("/lookup", response_model=LookupResponse)
def lookup(req: LookupRequest):
    """Lookup region and country for a given latitude/longitude."""
    if regions is None:
        raise HTTPException(
            status_code=503,
            detail="/lookup is disabled: GeoPackage not loaded. See README for setup.",
        )
    region, country, iso_code = lookup_geo(req.lat, req.lon)
    return {"region": region, "country": country, "iso_code": iso_code}

# ----------------------------------------------------------------------------
# RUN
# ----------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
