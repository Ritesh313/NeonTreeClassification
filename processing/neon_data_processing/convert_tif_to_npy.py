#!/usr/bin/env python3
"""
Convert TIF files to NPY format with validation for NEON tree crown data.

This script:
1. Reads existing crop_metadata.csv
2. Validates and converts TIF files to NPY format
3. Creates data_npy directory with same modality structure
4. Updates metadata with NPY paths and validity flags

Usage:
    python convert_tif_to_npy.py /path/to/cropped_crowns_modality_organized /path/to/crop_metadata.csv
"""

import os
import pandas as pd
import numpy as np
import rasterio
from pathlib import Path
import argparse
from tqdm import tqdm


def validate_tif_file(tif_path, modality, expected_bands=None):
    """
    Simple binary validation for tree crown files.
    Either fully valid or invalid - no thresholds for small crowns.
    """
    try:
        if not Path(tif_path).exists():
            return False, "File not found"

        with rasterio.open(tif_path) as src:
            # Check band count
            if expected_bands and src.count != expected_bands:
                return (
                    False,
                    f"Wrong band count: expected {expected_bands}, got {src.count}",
                )

            # Check minimum size based on modality
            if modality == "rgb":
                min_size = 3  # RGB at higher resolution
            else:  # hsi, lidar
                min_size = 2  # HSI/LiDAR at lower resolution, allow 2x2

            if src.height < min_size or src.width < min_size:
                return (
                    False,
                    f"Too small: {src.height}x{src.width} (min: {min_size}x{min_size})",
                )

            # Read first band to check for corruption
            try:
                data = src.read(1)
            except Exception as e:
                return False, f"Cannot read data: {str(e)}"

            # For HSI/LiDAR: Allow some nodata pixels (crown masking), but not too many
            # For RGB: Should have minimal nodata
            if modality in ["hsi", "lidar"]:
                # HSI/LiDAR can have nodata pixels from crown masking - check if we have SOME valid data
                if src.nodata is not None:
                    valid_pixels = np.sum(data != src.nodata)
                else:
                    valid_pixels = np.sum((data != -9999) & ~np.isnan(data))

                total_pixels = data.size
                valid_ratio = valid_pixels / total_pixels

                # Require at least 10% valid pixels for HSI/LiDAR (very lenient for small crowns)
                if valid_ratio < 0.1:
                    return False, f"Too few valid pixels: {valid_ratio:.1%}"
            else:
                # For RGB: stricter nodata check
                if src.nodata is not None:
                    has_nodata = np.any(data == src.nodata)
                else:
                    # Common nodata values
                    has_nodata = np.any((data == -9999) | np.isnan(data))

                if has_nodata:
                    return False, "Contains nodata pixels"

        return True, "Valid"

    except Exception as e:
        return False, f"Validation error: {str(e)}"


def convert_tif_to_npy(tif_path, npy_path):
    """Convert TIF file to NPY format with nodata replacement."""
    try:
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(npy_path), exist_ok=True)

        with rasterio.open(tif_path) as src:
            data = src.read().astype(np.float32)  # Shape: [bands, height, width]

            # Replace nodata values with 0
            if src.nodata is not None:
                data[data == src.nodata] = 0.0
            else:
                # Handle common nodata values
                data[data == -9999.0] = 0.0
                data[np.isnan(data)] = 0.0

        # Save as numpy array
        np.save(npy_path, data)
        return True, "Converted successfully"

    except Exception as e:
        return False, f"Conversion failed: {str(e)}"


def process_modality_files(base_tif_dir, base_npy_dir, modality, expected_bands=None):
    """Process all files for a specific modality."""
    tif_dir = os.path.join(base_tif_dir, modality)
    npy_dir = os.path.join(base_npy_dir, modality)

    if not os.path.exists(tif_dir):
        print(f"⚠️ Warning: {modality} directory not found: {tif_dir}")
        return {}

    # Get all TIF files
    tif_files = list(Path(tif_dir).glob("*.tif"))

    results = {}
    print(f"\n📁 Processing {len(tif_files)} {modality.upper()} files...")

    for tif_file in tqdm(tif_files, desc=f"Converting {modality}"):
        crown_id = tif_file.stem  # filename without extension
        tif_path = str(tif_file)
        npy_path = os.path.join(npy_dir, f"{crown_id}.npy")

        # Validate TIF file
        is_valid, message = validate_tif_file(tif_path, modality, expected_bands)

        if is_valid:
            # Convert to NPY
            converted, conv_message = convert_tif_to_npy(tif_path, npy_path)
            if not converted:
                is_valid = False
                message = conv_message

        results[crown_id] = {
            f"{modality}_tif_path": tif_path,
            f"{modality}_npy_path": npy_path if is_valid else None,
            f"{modality}_valid": is_valid,
            f"{modality}_error": message if not is_valid else None,
        }

    valid_count = sum(1 for r in results.values() if r[f"{modality}_valid"])
    print(
        f"✅ {modality.upper()}: {valid_count}/{len(tif_files)} files valid and converted"
    )

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Convert TIF files to NPY with validation"
    )
    parser.add_argument(
        "tif_base_dir", help="Base directory containing modality-organized TIF files"
    )
    parser.add_argument(
        "metadata_csv",
        nargs="?",
        help="Path to existing crop_metadata.csv (optional: will auto-detect if not provided)",
    )
    parser.add_argument(
        "--output-csv", help="Output CSV path (default: adds _npy suffix)"
    )
    parser.add_argument(
        "--npy-dir", help="NPY output directory name (default: cropped_crowns_npy)"
    )

    args = parser.parse_args()

    # Set up paths
    tif_base_dir = Path(args.tif_base_dir)
    npy_base_dir = tif_base_dir.parent / (args.npy_dir or "cropped_crowns_npy")

    # Auto-detect metadata CSV if not provided
    if args.metadata_csv:
        metadata_csv = args.metadata_csv
    else:
        # Look for crop_metadata.csv in the TIF directory or its parent
        potential_paths = [
            tif_base_dir / "crop_metadata.csv",
            tif_base_dir.parent / "crop_metadata.csv",
        ]

        metadata_csv = None
        for path in potential_paths:
            if path.exists():
                metadata_csv = str(path)
                print(f"📍 Auto-detected metadata CSV: {metadata_csv}")
                break

        if not metadata_csv:
            print(f"⚠️ No metadata CSV found in expected locations:")
            for path in potential_paths:
                print(f"   - {path}")
            print(f"   Will create new metadata from TIF files only")
            metadata_csv = str(
                tif_base_dir / "crop_metadata.csv"
            )  # Use this path for output

    if args.output_csv:
        output_csv = args.output_csv
    else:
        # Save CSV in the NPY directory with _npy suffix
        csv_path = Path(metadata_csv)
        output_csv = npy_base_dir / f"{csv_path.stem}_npy{csv_path.suffix}"

    print(f"🔄 NEON TIF → NPY Conversion")
    print(f"📂 Input TIF directory: {tif_base_dir}")
    print(f"📂 Output NPY directory: {npy_base_dir}")
    print(f"📄 Input metadata: {metadata_csv}")
    print(f"📄 Output metadata: {output_csv}")

    # Create NPY base directory
    os.makedirs(npy_base_dir, exist_ok=True)

    # Process each modality
    modality_results = {}

    # RGB - 3 bands expected
    rgb_results = process_modality_files(
        tif_base_dir, npy_base_dir, "rgb", expected_bands=3
    )
    modality_results.update(rgb_results)

    # HSI - 369 bands expected (based on your data)
    hsi_results = process_modality_files(
        tif_base_dir, npy_base_dir, "hsi", expected_bands=369
    )
    for crown_id, result in hsi_results.items():
        if crown_id in modality_results:
            modality_results[crown_id].update(result)
        else:
            modality_results[crown_id] = result

    # LiDAR - 1 band expected
    lidar_results = process_modality_files(
        tif_base_dir, npy_base_dir, "lidar", expected_bands=1
    )
    for crown_id, result in lidar_results.items():
        if crown_id in modality_results:
            modality_results[crown_id].update(result)
        else:
            modality_results[crown_id] = result

    # Load existing metadata if it exists
    if os.path.exists(metadata_csv):
        print(f"\n📖 Loading existing metadata from {metadata_csv}")
        existing_df = pd.read_csv(metadata_csv)
        print(f"Found {len(existing_df)} existing records")

        # Check if this is original long format or already processed NPY format
        is_long_format = "modality" in existing_df.columns
        if is_long_format:
            print(
                "📋 Original metadata is in long format - will create new NPY structure"
            )
            existing_df = pd.DataFrame()  # Don't merge, create fresh
        else:
            print("📋 Existing metadata appears to be NPY format - will merge")
    else:
        print(f"\n⚠️ Metadata file not found, creating new one")
        existing_df = pd.DataFrame()

    # Convert results to DataFrame
    results_df = pd.DataFrame.from_dict(modality_results, orient="index")
    results_df.index.name = "crown_id"
    results_df = results_df.reset_index()

    # Add all_modalities_valid flag
    results_df["all_modalities_valid"] = (
        results_df.get("rgb_valid", False)
        & results_df.get("hsi_valid", False)
        & results_df.get("lidar_valid", False)
    )

    # Merge with existing metadata if available
    if not existing_df.empty and "crown_id" in existing_df.columns:
        # Merge on crown_id, keeping all existing columns
        final_df = existing_df.merge(
            results_df, on="crown_id", how="outer", suffixes=("_old", "")
        )

        # Clean up any duplicate columns from merge
        for col in final_df.columns:
            if col.endswith("_old"):
                base_col = col[:-4]
                if base_col in final_df.columns:
                    final_df = final_df.drop(columns=[col])
    else:
        final_df = results_df

    # Save updated metadata
    final_df.to_csv(output_csv, index=False)

    # Print summary
    total_samples = len(final_df)
    rgb_valid = final_df.get("rgb_valid", pd.Series(dtype=bool)).sum()
    hsi_valid = final_df.get("hsi_valid", pd.Series(dtype=bool)).sum()
    lidar_valid = final_df.get("lidar_valid", pd.Series(dtype=bool)).sum()
    all_valid = final_df["all_modalities_valid"].sum()

    print(f"\n📊 CONVERSION SUMMARY")
    print(f"Total samples: {total_samples}")
    print(f"RGB valid: {rgb_valid} ({rgb_valid/total_samples*100:.1f}%)")
    print(f"HSI valid: {hsi_valid} ({hsi_valid/total_samples*100:.1f}%)")
    print(f"LiDAR valid: {lidar_valid} ({lidar_valid/total_samples*100:.1f}%)")
    print(f"All modalities valid: {all_valid} ({all_valid/total_samples*100:.1f}%)")

    print(f"\n✅ Conversion complete!")
    print(f"📄 Updated metadata saved to: {output_csv}")
    print(f"📂 NPY files saved to: {npy_base_dir}")


if __name__ == "__main__":
    main()
