"""
Tile processing utilities for NEON data.

This module contains functions for processing NEON tiles and extracting
crown data from RGB, HSI, and LiDAR modalities.
"""

import os
import re
from typing import Tuple, Optional


def extract_neon_coords_from_filename(filename: str, modality: str) -> Tuple[Optional[int], Optional[int]]:
    """
    Extract coordinates from NEON filenames based on modality.
    
    Args:
        filename: Full path to the file
        modality: 'RGB', 'HSI', or 'LiDAR'
    
    Returns:
        Tuple of (easting, northing) or (None, None) if not found
    """
    basename = os.path.basename(filename)
    
    if modality == 'RGB':
        # NEON RGB: 2019_BART_5_314000_4878000_image.tif
        match = re.search(r'(\d+)_(\d+)_image\.tif$', basename)
    elif modality == 'HSI':
        # NEON HSI: NEON_D01_BART_DP3_317000_4882000_reflectance.h5
        match = re.search(r'(\d+)_(\d+)_reflectance\.h5$', basename)
    elif modality == 'LiDAR':
        # NEON LiDAR: NEON_D01_BART_DP3_319000_4881000_CHM.tif
        match = re.search(r'(\d+)_(\d+)_CHM\.tif$', basename)
    else:
        return None, None
    
    if match:
        return int(match.group(1)), int(match.group(2))
    return None, None


# TODO: Move more functions from process_tiles_to_crowns.py 
# This is a minimal start 
