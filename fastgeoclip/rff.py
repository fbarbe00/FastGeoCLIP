"""
Random Fourier Features encoding used by the GeoCLIP location encoder.

Single-purpose simplification of the upstream `rff/` package: we only need
GaussianEncoding, so the other variants (BasicEncoding, PositionalEncoding)
and the separate functional module are gone.

Math matches upstream exactly:
    γ(v) = ( cos(2π · v B^T) , sin(2π · v B^T) )
with B sampled from N(0, σ²) of shape (encoded_size, input_size).

The order (cos, sin) and the transpose on B both matter — the downstream
linear layers were trained against this convention, and swapping either
would silently produce nonsense predictions.
"""

import math
import torch
import torch.nn as nn


class GaussianEncoding(nn.Module):
    def __init__(self, sigma: float = 1.0, input_size: int = 2, encoded_size: int = 256):
        super().__init__()
        b = torch.randn(encoded_size, input_size) * sigma
        self.b = nn.Parameter(b, requires_grad=False)

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        vp = 2 * math.pi * (v @ self.b.T)
        return torch.cat([torch.cos(vp), torch.sin(vp)], dim=-1)
