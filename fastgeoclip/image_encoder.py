import os
import torch
import torch.nn as nn
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor

DEFAULT_CLIP_PATH = os.environ.get("CLIP_MODEL_PATH", "/app/clip/clip-vit-large-vision")

class ImageEncoder(nn.Module):
    def __init__(self, model_path=DEFAULT_CLIP_PATH):
        super().__init__()
        if not os.path.isdir(model_path) or not os.path.isfile(os.path.join(model_path, "config.json")):
            raise FileNotFoundError(
                f"CLIP vision tower not found at {model_path!r}. "
                f"Run `python reduce_clip_size.py` to download it, or set "
                f"CLIP_MODEL_PATH to an existing directory containing "
                f"config.json + model.safetensors + preprocessor_config.json."
            )

        self.CLIP = CLIPVisionModelWithProjection.from_pretrained(
            model_path,
            local_files_only=True
        )

        # Use CLIPImageProcessor instead of AutoProcessor
        self.image_processor = CLIPImageProcessor.from_pretrained(
            model_path,
            local_files_only=True,
            use_fast=True
        )
        
        # Your original dimensions that worked
        self.mlp = nn.Sequential(
            nn.Linear(768, 768),
            nn.ReLU(),
            nn.Linear(768, 512)
        )
        
        for param in self.CLIP.parameters():
            param.requires_grad = False
    
    def preprocess_image(self, image):
        x = self.image_processor(images=image, return_tensors="pt")["pixel_values"]
        return x
    
    def forward(self, x):
        with torch.inference_mode():
            x = self.CLIP(pixel_values=x).image_embeds
            x = self.mlp(x)
        return x
