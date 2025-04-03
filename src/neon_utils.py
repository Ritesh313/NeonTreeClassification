import os
import re
import glob
import numpy as np
import pandas as pd
import rasterio
import matplotlib.pyplot as plt

# Function to extract coordinates from filenames
def extract_rgb_coords(filename):
    # Example: 2022_HARV_7_725000_4697000_image.tif
    match = re.search(r'(\d+)_(\d+)_image\.tif$', os.path.basename(filename))
    if match:
        return int(match.group(1)), int(match.group(2))
    return None, None

def extract_hsi_coords(filename):
    # Example: NEON_D01_HARV_DP3_737000_4700000_bidirectional_reflectance.h5
    match = re.search(r'(\d+)_(\d+)_bidirectional', os.path.basename(filename))
    if match:
        return int(match.group(1)), int(match.group(2))
    return None, None

if __name__ == "__main__":
    # Define paths to rgb and hsi directories
    rgb_dir = ''
    hsi_dir = ''

    # Get all RGB and HSI files
    rgb_files = glob.glob(os.path.join(rgb_dir, '*.tif'))
    hsi_files = glob.glob(os.path.join(hsi_dir, '*.h5'))

    print(f"Found {len(rgb_files)} RGB tiles and {len(hsi_files)} HSI tiles")

    # Extract coordinates and create lookup dictionaries
    rgb_lookup = {}
    for rgb_file in rgb_files:
        x, y = extract_rgb_coords(rgb_file)
        if x is not None and y is not None:
            rgb_lookup[(x, y)] = rgb_file

    hsi_lookup = {}
    for hsi_file in hsi_files:
        x, y = extract_hsi_coords(hsi_file)
        if x is not None and y is not None:
            hsi_lookup[(x, y)] = hsi_file

    # Find direct matches between RGB and HSI tiles
    direct_matches = []
    for (rgb_x, rgb_y), rgb_file in rgb_lookup.items():
        if (rgb_x, rgb_y) in hsi_lookup:
            direct_matches.append({
                'rgb_file': rgb_file,
                'hsi_file': hsi_lookup[(rgb_x, rgb_y)],
                'easting': rgb_x,
                'northing': rgb_y
            })

    print(f"Found {len(direct_matches)} direct matches between RGB and HSI tiles")