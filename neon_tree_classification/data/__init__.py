"""
NEON Tree Classification Data Module

Provides dataset and datamodule classes for multi-modal tree crown data,
plus data handling and preprocessing utilities.
"""

from .dataset import (
    NeonCrownDataset,
    normalize_rgb,
    normalize_hsi,
    normalize_lidar,
    fill_nodata_lidar,
    RGBLoader,
    HSILoader,
    LiDARLoader,
)

from .datamodule import NeonCrownDataModule
from .shapefile_processor import ShapefileProcessor
from .neon_downloader import NEONDownloader

__all__ = [
    # Main dataset classes
    "NeonCrownDataset",
    "NeonCrownDataModule",
    # Normalization functions
    "normalize_rgb",
    "normalize_hsi",
    "normalize_lidar",
    "fill_nodata_lidar",
    # Data loaders
    "RGBLoader",
    "HSILoader",
    "LiDARLoader",
    # Utilities
    "ShapefileProcessor",
    "NEONDownloader",
]
