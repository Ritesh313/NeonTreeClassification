#!/usr/bin/env python3
"""
NEON HSI H5 to TIF Conversion Script

Converts NEON HSI H5 files to GeoTIFF format with simple, reliable processing.
Removes water absorption bands and applies light compression.

Usage:
    python hsi_convert_h5_to_tif.py -i /path/to/hsi/ -o /path/to/output/
    python hsi_convert_h5_to_tif.py -i /path/to/hsi/ -o /path/to/output/ -p 16

Author: Ritesh Chowdhry
"""

import os
import sys
import argparse
import glob
import time
import re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Tuple, List

import h5py
import numpy as np
import rasterio
from rasterio.transform import Affine
from tqdm import tqdm


def get_good_bands() -> np.ndarray:
    """
    Get indices of bands to keep (removes water absorption bands).

    CRITICAL: Must remove bands in descending order to avoid index shifting!

    Returns:
        ndarray: Band indices to keep
    """
    # Start with all bands (0-424)
    band_indices = np.arange(425)

    # Remove water absorption bands in DESCENDING ORDER to avoid index shifting
    # Bands around 2400nm (bands 419-424) - REMOVE FIRST
    band_indices = np.delete(band_indices, np.arange(419, 425))
    # Bands around 1900nm (bands 283-315) - REMOVE SECOND
    band_indices = np.delete(band_indices, np.arange(283, 315))
    # Bands around 1400nm (bands 192-210) - REMOVE LAST
    band_indices = np.delete(band_indices, np.arange(192, 210))

    return band_indices


def safe_decode(value) -> str:
    """
    Safely decode bytes to string, handling both bytes and string inputs.

    Args:
        value: Input value (bytes or str)

    Returns:
        str: Decoded string
    """
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def convert_hsi_h5_to_tif_simple(
    h5_file: str, output_dir: str, overwrite: bool = False
) -> Optional[str]:
    """
    Convert a single HSI H5 file to GeoTIFF format using simple, reliable approach.

    Args:
        h5_file: Path to input H5 file
        output_dir: Output directory for TIF file
        overwrite: Whether to overwrite existing files

    Returns:
        str: Path to output TIF file, or None if failed
    """
    try:
        # Simple: just replace .h5 with .tif
        input_basename = Path(h5_file).stem  # Remove .h5
        output_filename = f"{input_basename}.tif"
        output_path = os.path.join(output_dir, output_filename)

        # Check if file exists and skip if not overwriting
        if os.path.exists(output_path) and not overwrite:
            return output_path

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Simple approach - load all data at once (like your working version)
        with h5py.File(h5_file, "r") as hdf:
            # Get site name (first key in HDF file)
            sitename = list(hdf.keys())[0]

            # Get reflectance group
            refl_group = hdf[sitename]["Reflectance"]

            # Load ALL data at once - simple and fast
            refl_data = refl_group["Reflectance_Data"][:]

            # Get metadata with safe decoding
            epsg_raw = refl_group["Metadata"]["Coordinate_System"]["EPSG Code"][()]
            epsg = safe_decode(epsg_raw)
            epsg = epsg.split("'")[1] if "'" in epsg else epsg

            map_info_raw = refl_group["Metadata"]["Coordinate_System"]["Map_Info"][()]
            map_info = safe_decode(map_info_raw).split(",")

            # Extract spatial metadata
            pixel_width = float(map_info[5])
            pixel_height = float(map_info[6])
            x_min = float(map_info[3])
            y_max = float(map_info[4])

            no_data_val = float(
                refl_group["Reflectance_Data"].attrs["Data_Ignore_Value"]
            )

            # Remove water absorption bands (CORRECTED ORDER)
            good_bands = get_good_bands()
            refl_data = refl_data[:, :, good_bands]

            # Rearrange dimensions for rasterio (bands, rows, cols)
            refl_data = np.moveaxis(refl_data, 2, 0)

            # Create geotransform
            transform = Affine.translation(x_min, y_max) * Affine.scale(
                pixel_width, -pixel_height
            )

        # Simple write - no tiling, no chunking, light compression
        with rasterio.open(
            output_path,
            "w",
            driver="GTiff",
            height=refl_data.shape[1],
            width=refl_data.shape[2],
            count=refl_data.shape[0],
            dtype=np.float32,
            crs=f"EPSG:{epsg}",
            transform=transform,
            nodata=no_data_val,
            compress="deflate",  # Light, reliable compression
            predictor=2,  # Good for float data
        ) as dst:
            # Single write operation - let rasterio handle it optimally
            dst.write(refl_data.astype(np.float32))

        return output_path

    except Exception as e:
        print(f"✗ Error converting {h5_file}: {e}")
        # Clean up partial file
        if "output_path" in locals() and os.path.exists(output_path):
            try:
                os.remove(output_path)
            except:
                pass
        return None


def convert_batch(
    h5_files: List[str], output_dir: str, overwrite: bool = False
) -> Tuple[int, int]:
    """
    Convert a batch of HSI H5 files to TIF format (sequential).

    Returns:
        tuple: (successful_conversions, failed_conversions)
    """
    successful = 0
    failed = 0

    for h5_file in tqdm(h5_files, desc="Converting HSI files", unit="file"):
        result = convert_hsi_h5_to_tif_simple(
            h5_file=h5_file,
            output_dir=output_dir,
            overwrite=overwrite,
        )

        if result:
            successful += 1
        else:
            failed += 1

    return successful, failed


def convert_parallel(
    h5_files: List[str], output_dir: str, max_workers: int = 16, overwrite: bool = False
) -> Tuple[int, int]:
    """
    Convert HSI H5 files to TIF format using parallel processing.

    Args:
        max_workers: Number of parallel workers (default: 16 for your setup)

    Returns:
        tuple: (successful_conversions, failed_conversions)
    """
    successful = 0
    failed = 0

    print(f"🚀 Starting parallel conversion with {max_workers} workers...")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_file = {
            executor.submit(
                convert_hsi_h5_to_tif_simple, h5_file, output_dir, overwrite
            ): h5_file
            for h5_file in h5_files
        }

        # Process results with progress bar
        for future in tqdm(
            as_completed(future_to_file),
            total=len(h5_files),
            desc="Converting HSI files",
            unit="file",
        ):
            h5_file = future_to_file[future]
            try:
                result = future.result()
                if result:
                    successful += 1
                else:
                    failed += 1
            except Exception as e:
                print(f"✗ Error processing {h5_file}: {e}")
                failed += 1

    return successful, failed


def main():
    parser = argparse.ArgumentParser(
        description="Convert NEON HSI H5 files to GeoTIFF format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Convert all H5 files in directory (sequential)
    python hsi_convert_h5_to_tif.py -i /path/to/hsi/ -o /path/to/output/

    # Use maximum parallel processing (recommended)
    python hsi_convert_h5_to_tif.py -i /path/to/hsi/ -o /path/to/output/ -p 16

    # Overwrite existing files
    python hsi_convert_h5_to_tif.py -i /path/to/hsi/ -o /path/to/output/ -p 16 --overwrite
        """,
    )

    parser.add_argument(
        "-i",
        "--input-dir",
        required=True,
        help="Input directory containing HSI H5 files",
    )
    parser.add_argument(
        "-o", "--output-dir", required=True, help="Output directory for TIF files"
    )
    parser.add_argument(
        "-p",
        "--parallel",
        type=int,
        default=1,
        help="Number of parallel workers (default: 1, recommended: 16)",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing TIF files"
    )

    args = parser.parse_args()

    # Validate input directory
    if not os.path.exists(args.input_dir):
        print(f"✗ Input directory does not exist: {args.input_dir}")
        sys.exit(1)

    # Find all H5 files
    h5_pattern = os.path.join(args.input_dir, "*.h5")
    h5_files = glob.glob(h5_pattern)

    if not h5_files:
        print(f"✗ No H5 files found in: {args.input_dir}")
        sys.exit(1)

    print(f"NEON HSI H5 to TIF Conversion (Simple & Fast)")
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Found {len(h5_files)} H5 files")
    print(f"Parallel workers: {args.parallel}")
    print(f"Overwrite existing: {args.overwrite}")
    print("=" * 50)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Start conversion
    start_time = time.time()

    if args.parallel > 1:
        successful, failed = convert_parallel(
            h5_files=h5_files,
            output_dir=args.output_dir,
            max_workers=args.parallel,
            overwrite=args.overwrite,
        )
    else:
        successful, failed = convert_batch(
            h5_files=h5_files,
            output_dir=args.output_dir,
            overwrite=args.overwrite,
        )

    # Print summary
    elapsed_time = time.time() - start_time

    print("\n" + "=" * 50)
    print("CONVERSION COMPLETED")
    print(f"✓ Successfully converted: {successful} files")
    if failed > 0:
        print(f"✗ Failed conversions: {failed} files")
    print(f"Total time: {elapsed_time:.1f} seconds")
    if successful > 0:
        print(f"Average time per file: {elapsed_time/successful:.1f} seconds")
    print(f"Output directory: {args.output_dir}")

    if failed > 0:
        print(f"\n{failed} files failed to convert. Check error messages above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
