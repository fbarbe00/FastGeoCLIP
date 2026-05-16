"""
Data loading utilities for GeoCLIP.
Handles GPS coordinate loading and caching.
"""

import os
import csv
import torch

file_dir = os.path.dirname(os.path.realpath(__file__))


def load_gps_data(csv_file: str) -> torch.Tensor:
    """
    Load GPS coordinates from CSV file efficiently.

    Args:
        csv_file: Path to CSV file with LAT and LON columns

    Returns:
        Tensor of shape (N, 2) with [lat, lon] coordinates
    """
    lat_lon = []
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            lat_lon.append([float(row['LAT']), float(row['LON'])])

    return torch.tensor(lat_lon, dtype=torch.float32)
