"""
NEON Tree Classification Package

A comprehensive package for downloading, processing, and classifying tree species
from NEON airborne data including RGB, hyperspectral imagery, and LiDAR.
"""

__version__ = "0.1.0"
__author__ = "Ritesh Chowdhry"

# Import key classes for easy access
from .data.shapefile_processor import ShapefileProcessor
from .models.architectures import HsiPixelClassifier

__all__ = [
    "ShapefileProcessor", 
    "HsiPixelClassifier",
]
