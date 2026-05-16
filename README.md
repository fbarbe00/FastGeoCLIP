# FastGeoCLIP

A CPU-optimized fork of [GeoCLIP](https://github.com/VicenteVivan/geo-clip) (Vicente Vivanco, NeurIPS 2023), for low-latency image-to-GPS prediction. **Roughly 10× faster than the upstream library on the same CPU and image**, with no accuracy loss vs. the upstream model. Also includes a FastAPI server for querying, and batch predictions.

The original version embedded location after each run, and did some linearly compared each embedding with the location embeddings. This version gets the CLIP checkpoint with a vision-only variant loaded from local disk, pre-computes and caches the GPS gallery embeddings, and uses FAISS to narrow the candidate pool before exact scoring.

See [NOTICE.md](NOTICE.md) for a detailed list of changes vs. the original.

## What this is for

If you want to deploy GeoCLIP as an inference-only HTTP service on a CPU box (e.g. inside Docker), or use it as a plain Python library (See **Library usage** below), this fork is for you.

If you want to train GeoCLIP or use it as a general research toolkit, the upstream repo is still the right place.

## Setup

The fine-tuned GeoCLIP weights and the 100K-point GPS gallery are bundled in this repository. The only required setup step is downloading the CLIP vision tower:

```bash
pip install -r requirements.txt
python reduce_clip_size.py # downloads openai/clip-vit-large-patch14, keeps vision tower only
```

That's enough to run `/predict` or to use the library.

For `/lookup` (reverse-geocoding a coordinate to region + country), drop a GeoPackage at `data/admin1_clean.gpkg`. Any admin-1 boundaries file with `name_en`, `admin`, and `iso_a2` columns works. A reasonable source:

```bash
# optional, only if you want /lookup
wget -O /tmp/ne.zip https://naciscdn.org/naturalearth/10m/cultural/ne_10m_admin_1_states_provinces.zip
unzip /tmp/ne.zip -d /tmp/ne
mkdir -p data
python -c "import geopandas as g; \
  df = g.read_file('/tmp/ne/ne_10m_admin_1_states_provinces.shp').rename(columns={'name': 'name_en'}); \
  df = df[['name_en', 'admin', 'iso_a2', 'geometry']]; \
  df.to_file('data/admin1_clean.gpkg', layer='regions', driver='GPKG')"
```

## Library usage

```python
from fastgeoclip import FastGeoCLIP

model = FastGeoCLIP() # k_neighbors defaults to 1000, index_type to flatl2
model.eval()

gps, probs = model.predict("path/to/image.jpg", top_k=5)
for (lat, lon), p in zip(gps.tolist(), probs.tolist()):
    print(f"{lat:.4f}, {lon:.4f}  (p={p:.3f})")
```

First call writes embedding + FAISS caches next to the GPS gallery CSV (about a minute on a modern CPU). Subsequent calls reuse them. Set `CACHE_DIR` to relocate the caches (useful in containers with a persistent volume).

If your CLIP weights are not at the default `/app/clip/clip-vit-large-vision`, set `CLIP_MODEL_PATH=./clip/clip-vit-large-vision` before importing.

For batch inference, use `model.predict_batch([path1, path2, ...], top_k=5)` — encodes all images in one CLIP forward pass.

## HTTP service

### Endpoints

- `POST /predict` — form fields: `file` (image, ≤ 25 MB by default), `top_k` (1–100, default 5). Returns ranked predictions.
- `POST /lookup` — body: `{"lat": ..., "lon": ...}`. Returns `{"region", "country", "iso_code"}`.

### Run

Make sure to run the Setup first to download the CLIP weights.

```bash
docker build -t fastgeoclip .
docker run --rm -p 8000:8000 \
  -v "$PWD/clip:/app/clip:ro" \
  -v "$PWD/data:/app/data:ro" \
  fastgeoclip
```

See `docker-compose.example.yml` for a compose snippet with a persistent cache volume.

Locally:

```bash
CLIP_MODEL_PATH=./clip/clip-vit-large-vision DATA_GPKG=./data/admin1_clean.gpkg python app.py
```

Once it's up on `localhost:8000`, send a photo with curl:

```bash
curl -s -X POST http://localhost:8000/predict?top_k=3 \
  -F "file=@/path/to/photo.jpg" | jq
```

Example response:

```json
{
  "predictions": [
    { "rank": 1, "latitude": 48.8741, "longitude":  2.2945, "probability": 0.334 },
    { "rank": 2, "latitude": 48.8736, "longitude":  2.2942, "probability": 0.333 },
    { "rank": 3, "latitude": 48.8747, "longitude":  2.2950, "probability": 0.333 }
  ],
  "processing_time_ms": 512.4
}
```

Reverse-geocode the top prediction with `/lookup`:

```bash
curl -s -X POST http://localhost:8000/lookup \
  -H 'Content-Type: application/json' \
  -d '{"lat": 48.8741, "lon": 2.2945}' | jq
# {"region": "Île-de-France", "country": "France", "iso_code": "FR"}
```

## Configuration

| Env var | Default | Purpose |
|---|---|---|
| `NUM_THREADS` | 4 | PyTorch CPU thread count |
| `K_NEIGHBORS` | 1000 | FAISS candidate pool size — doesn't change top-1 accuracy; larger only slows you down |
| `FAISS_INDEX_TYPE` | `flatl2` | `flatl2` (exact), `hnsw` or `ivf` (approximate, faster, slight accuracy hit) |
| `MAX_UPLOAD_MB` | 25 | Hard cap on `/predict` upload size |
| `CLIP_MODEL_PATH` | `/app/clip/clip-vit-large-vision` | Where the CLIP vision tower lives |
| `CACHE_DIR` | `<package>/gps_gallery` | Where embedding + FAISS caches are written |
| `DATA_GPKG` | `/app/data/admin1_clean.gpkg` | Path to reverse-geocoding GeoPackage |

## Benchmarks

### End-to-end speed

Mean total latency on 10 random IM2GPS3K images (warmup excluded), 8-core Intel laptop, PyTorch 2.12, CPU-only:

| Implementation | mean total latency | speedup |
|---|---|---|
| Upstream `geo-clip` | 5.9 s | 1× |
| FastGeoCLIP — `flatl2` (default) | 0.50 s | 11.8× |
| FastGeoCLIP — `ivf`              | 0.48 s | 12.3× |
| FastGeoCLIP — `hnsw`             | 0.42 s | 13.9× |

### Accuracy on IM2GPS3K (2997 images)

Dataset: <https://www.kaggle.com/datasets/lctngdng/im2gps3k>. Top-1 GPS distance to ground truth, with end-to-end latency from the same hardware:

| Implementation        | mean latency | @1 km | @25 km | @200 km | @750 km | @2500 km | median err | mean err |
|-----------------------|-------------:|------:|-------:|--------:|--------:|---------:|-----------:|---------:|
| Upstream `geo-clip`   |        5.9 s | 13.1% | 32.2%  | 48.1%   | 66.6%   | 82.3%    |   241 km   |  1764 km |
| FastGeoCLIP — flatl2  |       0.50 s | 13.1% | 32.2%  | 48.1%   | 66.6%   | 82.3%    |   241 km   |  1764 km |
| FastGeoCLIP — ivf     |       0.48 s | 12.0% | 29.8%  | 46.4%   | 66.0%   | 82.3%    |   271 km   |  1813 km |
| FastGeoCLIP — hnsw    |       0.42 s | 12.0% | 30.3%  | 45.4%   | 63.8%   | 79.4%    |   296 km   |  1978 km |

FastGeoCLIP with `flatl2` reproduces upstream's top-1 GPS on every one of 10 random images sampled with seed 42, so its accuracy across the full 2997-image set is identical to upstream's. `ivf` and `hnsw` use approximate nearest-neighbor search and trade ~1 pp of top-1 accuracy for a small additional speedup.

K_NEIGHBORS sweeps (1000 → 100000) produced identical accuracy at every value within each index — the top-1 prediction doesn't depend on it. The default of 1000 is the smallest pool that gives full accuracy.

### FAISS+rescoring step in isolation

| Index / K          | per-query | notes |
|--------------------|----------:|-------|
| flatl2 / K=1000    |   12.7 ms | exact, fastest exact config |
| hnsw / K=1000      |    0.4 ms | fastest overall |
| ivf / K=1000       |    1.0 ms | between the two |

The CLIP encoder (~680 ms) dominates total latency, so swapping to HNSW only saves ~12 ms (≈ 2%) at the cost of ~1.4 pp top-1 accuracy. Worth it only on much larger galleries or with a faster encoder.

## Alternatives

If you're researching other image geolocation models, I'd recommend looking at these other projects:

- **[PLONK](https://github.com/nicolas-dufour/plonk)** - a more recent geolocation model designed for GPU inference. PLONK can be trained or fine-tuned on the public [OSV5Mdataset](https://huggingface.co/datasets/osv5m/osv5m-wds), which would also be a natural choice for further fine-tuning GeoCLIP itself on a larger, more recent set of geotagged images.
- **[PIGEON](https://github.com/LukasHaas/PIGEON)** - a research codebase for training geo-localization models from scratch with alternative architectures. The trained weights are not open, but the code.
- **[GeoEstimation](https://github.com/TIBHannover/GeoEstimation)** - a similar approach than PIGEON. Code and weights used to be public, but have been recently deleted.

## License

MIT, see [LICENSE](LICENSE). Original copyright Vicente Vivanco (2024). See [NOTICE.md](NOTICE.md) for the full list of changes.

## Citation

If you use GeoCLIP (or this fork) in academic work, please cite the original paper:

```bibtex
@inproceedings{geoclip,
  title={GeoCLIP: Clip-Inspired Alignment between Locations and Images for Effective Worldwide Geo-localization},
  author={Vivanco Cepeda, Vicente and Nayak, Gaurav Kumar and Shah, Mubarak},
  booktitle={Advances in Neural Information Processing Systems},
  year={2023}
}
```

A reference to this fork is also appreciated if FastGeoCLIP's CPU inference path was useful for your work:

```bibtex
@software{fastgeoclip,
  title  = {FastGeoCLIP: CPU-optimised inference fork of GeoCLIP},
  author = {Barbero, Fabio},
  year   = {2026},
  url    = {https://github.com/fbarbe00/FastGeoClip}
}
```
