"""
NEON Tree Classification Package

A clean, modular approach to multi-modal tree species classification.
"""

__version__ = "0.1.0"
__author__ = "Ritesh Chowdhry"

# Import key classes for easy access
from .data.dataset import NeonCrownDataset
from .data.datamodule import NeonCrownDataModule

# Import model factory functions
from .models.rgb_models import create_rgb_model
from .models.hsi_models import create_hsi_model
from .models.lidar_models import create_lidar_model

# Import Lightning modules
from .models.lightning_modules import (
    BaseTreeClassifier,
    RGBClassifier,
    HSIClassifier,
    LiDARClassifier,
)

__all__ = [
    "NeonCrownDataset",
    "NeonCrownDataModule",
    "create_rgb_model",
    "create_hsi_model",
    "create_lidar_model",
    "BaseTreeClassifier",
    "RGBClassifier",
    "HSIClassifier",
    "LiDARClassifier",
]
