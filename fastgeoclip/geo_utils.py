"""
Shared geographic utilities for GeoCLIP models.
Contains coordinate projections and transformations used across multiple models.
"""

import torch
import torch.nn as nn

# Equal Earth Projection Constants
# Reference: https://en.wikipedia.org/wiki/Equal_Earth_projection
EQUAL_EARTH_A1 = 1.340264
EQUAL_EARTH_A2 = -0.081106
EQUAL_EARTH_A3 = 0.000893
EQUAL_EARTH_A4 = 0.003796
EQUAL_EARTH_SF = 66.50336


def equal_earth_projection(L):
    """
    Project latitude/longitude to Equal Earth projection.

    Equal Earth is a compromise projection designed to minimize overall distortion.
    It produces a visually appealing map while maintaining relatively equal area representation.

    Args:
        L: Tensor of shape (batch_size, 2) with [latitude, longitude] in degrees

    Returns:
        Tensor of shape (batch_size, 2) with projected x, y coordinates
    """
    latitude = L[:, 0]
    longitude = L[:, 1]

    # Convert degrees to radians
    latitude_rad = torch.deg2rad(latitude)
    longitude_rad = torch.deg2rad(longitude)

    # Equal Earth projection formulas
    sin_theta = (torch.sqrt(torch.tensor(3.0)) / 2) * torch.sin(latitude_rad)
    theta = torch.asin(sin_theta)

    denominator = 3 * (
        9 * EQUAL_EARTH_A4 * theta**8 +
        7 * EQUAL_EARTH_A3 * theta**6 +
        3 * EQUAL_EARTH_A2 * theta**2 +
        EQUAL_EARTH_A1
    )

    x = (2 * torch.sqrt(torch.tensor(3.0)) * longitude_rad * torch.cos(theta)) / denominator
    y = (
        EQUAL_EARTH_A4 * theta**9 +
        EQUAL_EARTH_A3 * theta**7 +
        EQUAL_EARTH_A2 * theta**3 +
        EQUAL_EARTH_A1 * theta
    )

    return (torch.stack((x, y), dim=1) * EQUAL_EARTH_SF) / 180
