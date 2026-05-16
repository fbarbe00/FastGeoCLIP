#!/usr/bin/env python3
"""
Download openai/clip-vit-large-patch14 and save a vision-only variant.

The text tower of CLIP is unused for image→GPS prediction, so we keep only the
vision encoder + projection + image preprocessor. Output is suitable for
loading with `local_files_only=True` in air-gapped deployments.

Run once before building the Docker image, or once per host for local dev.
"""
import os
import sys

from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor

SRC = os.environ.get("CLIP_SRC", "openai/clip-vit-large-patch14")
# Default to writing next to this script so the output is predictable
# regardless of the caller's CWD. Override with CLIP_DST=/some/path.
DEFAULT_DST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "clip", "clip-vit-large-vision")
DST = os.environ.get("CLIP_DST", DEFAULT_DST)

try:
    os.makedirs(DST, exist_ok=True)
except PermissionError as e:
    sys.exit(
        f"Cannot create {DST}: {e}\n"
        f"Set CLIP_DST to a writable path, e.g. CLIP_DST=~/clip-vit-large-vision python reduce_clip_size.py"
    )

print(f"Downloading vision tower from {SRC}…")
model = CLIPVisionModelWithProjection.from_pretrained(SRC)

print("Downloading preprocessor…")
processor = CLIPImageProcessor.from_pretrained(SRC)

print(f"Saving to {DST}/")
model.save_pretrained(DST, safe_serialization=True)
processor.save_pretrained(DST)

print(f"Done. The container expects this directory at /app/clip/clip-vit-large-vision.")
print(f"Override the in-container path with CLIP_MODEL_PATH.")
