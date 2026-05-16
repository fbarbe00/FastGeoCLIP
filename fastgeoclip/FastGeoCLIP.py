import hashlib
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from .data_utils import load_gps_data
from .image_encoder import ImageEncoder
from .location_encoder import LocationEncoder

file_dir = os.path.dirname(os.path.realpath(__file__))
# Cache directory for generated embeddings + FAISS index. Defaults to the
# bundled gps_gallery/ so a fresh checkout works without configuration; override
# with CACHE_DIR to point at a persistent volume in production deployments.
cache_dir = os.environ.get("CACHE_DIR", os.path.join(file_dir, "gps_gallery"))

# Supported FAISS index types. flatl2 is exact; ivf and hnsw are approximate
# and only worth the swap on much larger galleries (millions). All three are
# correct on this 100K gallery; the difference is build-time + memory.
SUPPORTED_INDEX_TYPES = ("flatl2", "ivf", "hnsw")
DEFAULT_INDEX_TYPE = os.environ.get("FAISS_INDEX_TYPE", "flatl2").lower()

try:
    import faiss
except ImportError:
    print("FAISS not installed. Run: pip install faiss-cpu")
    faiss = None


def _artifact_fingerprint(*paths):
    """Short fingerprint of (size, mtime) tuples — used to bust caches when
    the weights or GPS gallery change. Avoids hashing 40 MB of weights on
    every import; size+mtime is enough to catch real artifact swaps."""
    parts = []
    for path in paths:
        st = os.stat(path)
        parts.append(f"{st.st_size}-{int(st.st_mtime)}")
    return hashlib.sha1("|".join(parts).encode()).hexdigest()[:8]


class FastGeoCLIP(nn.Module):
    """GeoCLIP with FAISS approximate nearest neighbor search"""

    def __init__(self, from_pretrained=True, k_neighbors=1000, index_type=None):
        super().__init__()
        if faiss is None:
            raise ImportError("FAISS is required. Install with: pip install faiss-cpu")

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        # Use vision-only model (path overridable via CLIP_MODEL_PATH; see image_encoder.py)
        self.image_encoder = ImageEncoder()
        # LocationEncoder defaults to from_pretrained=True; we re-load below
        # via _load_weights, so suppress its self-load to avoid touching the
        # weights file twice on every startup.
        self.location_encoder = LocationEncoder(from_pretrained=False)

        self.gps_csv = os.path.join(file_dir, "gps_gallery", "coordinates_100K.csv")
        self.gps_gallery = load_gps_data(self.gps_csv)

        self.weights_folder = os.path.join(file_dir, "weights")
        if from_pretrained:
            self._load_weights()

        # Fingerprint the artifacts that determine the embeddings + FAISS cache
        # contents. Without this, swapping location_encoder_weights.pth or the
        # CSV would silently reuse stale caches and corrupt predictions.
        self._artifact_fp = _artifact_fingerprint(
            self.gps_csv,
            os.path.join(self.weights_folder, "location_encoder_weights.pth"),
        )

        self.device = "cpu"
        self.k_neighbors = k_neighbors
        self.index_type = (index_type or DEFAULT_INDEX_TYPE).lower()
        if self.index_type not in SUPPORTED_INDEX_TYPES:
            raise ValueError(
                f"Unknown index_type {self.index_type!r}; expected one of {SUPPORTED_INDEX_TYPES}"
            )

        # Load embeddings and initialize reusable, normalized variant.
        self.gps_embeddings = self._load_or_compute_gps_embeddings()
        self.gps_embeddings_normalized = F.normalize(self.gps_embeddings, dim=1)

        # Load/build FAISS index and cache it for faster startup.
        self.faiss_index = self._load_or_build_faiss_index()

    def to(self, device):
        self.device = device
        self.image_encoder.to(device)
        self.location_encoder.to(device)
        self.logit_scale.data = self.logit_scale.data.to(device)
        self.gps_embeddings = self.gps_embeddings.to(device)
        self.gps_embeddings_normalized = self.gps_embeddings_normalized.to(device)
        return super().to(device)

    def _load_weights(self):
        load = lambda p: torch.load(p, map_location="cpu", weights_only=True)
        self.image_encoder.mlp.load_state_dict(
            load(f"{self.weights_folder}/image_encoder_mlp_weights.pth")
        )
        self.location_encoder.load_state_dict(
            load(f"{self.weights_folder}/location_encoder_weights.pth")
        )
        self.logit_scale = nn.Parameter(
            load(f"{self.weights_folder}/logit_scale_weights.pth")
        )

    def _load_or_compute_gps_embeddings(self):
        """Load pre-computed GPS embeddings or compute and cache them"""
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, f"embeddings_100K_{self._artifact_fp}.pth")

        if os.path.exists(cache_path):
            print(f"Loading cached GPS embeddings ({self._artifact_fp})...")
            return torch.load(cache_path, map_location="cpu", weights_only=True)

        print("Computing GPS embeddings (this may take a minute)...")
        with torch.inference_mode():
            batch_size = 1000
            embeddings_list = []
            for i in range(0, len(self.gps_gallery), batch_size):
                batch_gps = self.gps_gallery[i:i+batch_size]
                embeddings_list.append(self.location_encoder(batch_gps))
                if (i // batch_size + 1) % 10 == 0:
                    print(f"  Processed {i + batch_size}/{len(self.gps_gallery)} locations")
            embeddings = torch.cat(embeddings_list, dim=0)

        torch.save(embeddings, cache_path)
        return embeddings

    def _build_faiss_index(self):
        """Build a FAISS index of the configured type over the normalized gallery."""
        print(f"Building FAISS index ({self.index_type})...")
        embeddings_np = self.gps_embeddings_normalized.cpu().numpy().astype('float32')
        d = embeddings_np.shape[1]

        if self.index_type == "flatl2":
            # Exact. L2 on normalized vectors is equivalent to cosine.
            index = faiss.IndexFlatL2(d)
            index.add(embeddings_np)
        elif self.index_type == "ivf":
            # Inverted-file index. nlist tuned for 100K gallery; recall drops at
            # nprobe=1 but stays >0.95 at nprobe=8.
            nlist = 256
            quantizer = faiss.IndexFlatL2(d)
            index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)
            index.train(embeddings_np)
            index.add(embeddings_np)
            index.nprobe = 8
        elif self.index_type == "hnsw":
            # Hierarchical NSW. M=32 is a reasonable default for 100K-scale.
            index = faiss.IndexHNSWFlat(d, 32, faiss.METRIC_L2)
            index.hnsw.efConstruction = 80
            index.hnsw.efSearch = 64
            index.add(embeddings_np)
        else:  # pragma: no cover — guarded in __init__
            raise ValueError(f"Unsupported index_type {self.index_type!r}")

        print(f"FAISS index built ({self.index_type}, {index.ntotal} vectors)")
        return index

    def _load_or_build_faiss_index(self):
        """Load FAISS index from cache, or build and persist it."""
        os.makedirs(cache_dir, exist_ok=True)
        # Cache key encodes BOTH the index type and the artifact fingerprint so
        # swapping weights/CSV or index type rebuilds cleanly.
        cache_path = os.path.join(
            cache_dir, f"embeddings_100K_{self._artifact_fp}_{self.index_type}.faiss"
        )

        if os.path.exists(cache_path):
            try:
                print(f"Loading cached FAISS index ({self.index_type})...")
                index = faiss.read_index(cache_path)
                if index.ntotal == len(self.gps_gallery) and index.d == self.gps_embeddings_normalized.shape[1]:
                    if self.index_type == "ivf":
                        index.nprobe = 8
                    elif self.index_type == "hnsw":
                        index.hnsw.efSearch = 64
                    print(f"FAISS index loaded ({self.index_type}, {index.ntotal} vectors)")
                    return index
                print("Cached FAISS index metadata mismatch, rebuilding...")
            except Exception as exc:
                print(f"Failed to load cached FAISS index ({exc}), rebuilding...")

        index = self._build_faiss_index()
        try:
            faiss.write_index(index, cache_path)
            print(f"FAISS index cached at {cache_path}")
        except Exception as exc:
            print(f"Failed to cache FAISS index ({exc})")
        return index

    def _encode_images(self, images):
        """Encode one or more PIL images to normalized image features. Internal."""
        # HF processor accepts a list directly; one call is cheaper than a list
        # comprehension that hits the Python loop per image.
        pixel_values = self.image_encoder.image_processor(
            images=images, return_tensors="pt"
        )["pixel_values"].to(self.device)
        features = self.image_encoder(pixel_values)
        return F.normalize(features, dim=1)

    @torch.inference_mode()
    def predict(self, image_path, top_k, k_neighbors=None):
        """Predict top k GPS coordinates for a single image."""
        if k_neighbors is None:
            k_neighbors = self.k_neighbors

        image_features = self._encode_images([Image.open(image_path)])

        image_features_np = image_features.cpu().numpy().astype('float32')
        distances, indices = self.faiss_index.search(
            image_features_np, k=min(k_neighbors, len(self.gps_gallery))
        )

        neighbor_indices = torch.from_numpy(indices[0]).long()
        neighbor_embeddings = self.gps_embeddings_normalized[neighbor_indices]

        logit_scale = self.logit_scale.exp()
        neighbor_logits = logit_scale * (image_features @ neighbor_embeddings.t())

        top_logits, top_indices_in_neighbors = torch.topk(neighbor_logits, k=top_k, dim=1)
        final_indices = neighbor_indices[top_indices_in_neighbors[0]]

        top_pred_gps = self.gps_gallery[final_indices]
        top_pred_prob = torch.softmax(top_logits[0], dim=0)
        return top_pred_gps, top_pred_prob

    @torch.inference_mode()
    def predict_batch(self, image_paths, top_k, k_neighbors=None):
        """Predict top k GPS coordinates for a list of images.

        Runs the image preprocessor and CLIP encoder once over the whole batch,
        which amortises per-call Python overhead. Useful when scoring many
        images back-to-back (e.g. a multi-player upload round).
        """
        if k_neighbors is None:
            k_neighbors = self.k_neighbors

        images = [Image.open(p) for p in image_paths]
        image_features = self._encode_images(images)
        n = image_features.shape[0]

        feats_np = image_features.cpu().numpy().astype('float32')
        k = min(k_neighbors, len(self.gps_gallery))
        _, indices = self.faiss_index.search(feats_np, k=k)

        results = []
        logit_scale = self.logit_scale.exp()
        for i in range(n):
            neighbor_indices = torch.from_numpy(indices[i]).long()
            neighbor_embeddings = self.gps_embeddings_normalized[neighbor_indices]
            neighbor_logits = logit_scale * (image_features[i:i+1] @ neighbor_embeddings.t())
            top_logits, top_in_neighbors = torch.topk(neighbor_logits, k=top_k, dim=1)
            final_indices = neighbor_indices[top_in_neighbors[0]]
            results.append((self.gps_gallery[final_indices], torch.softmax(top_logits[0], dim=0)))
        return results
