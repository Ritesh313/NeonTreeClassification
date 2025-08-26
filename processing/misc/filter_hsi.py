#!/usr/bin/env python3
"""
Simple HSI corruption filter - removes samples with corrupted HSI files.

Usage:
    python filter_hsi.py input.csv output.csv [--bands 369] [--nodata-threshold 90]
"""

import pandas as pd
import rasterio
import numpy as np
from pathlib import Path
import argparse


def is_hsi_valid(hsi_path, expected_bands=369, nodata_threshold=90):
    """Check if HSI file is valid."""
    try:
        if not Path(hsi_path).exists():
            return False

        with rasterio.open(hsi_path) as src:
            if src.count != expected_bands or src.height < 2 or src.width < 2:
                return False

            band_data = src.read(1)
            if src.nodata is not None:
                nodata_pct = np.sum(band_data == src.nodata) / band_data.size * 100
            else:
                nodata_pct = (
                    np.sum((band_data == -9999) | np.isnan(band_data))
                    / band_data.size
                    * 100
                )

            return nodata_pct <= nodata_threshold
    except:
        return False


def filter_corrupted_hsi(input_csv, output_csv, bands=369, nodata_threshold=90):
    """Filter out samples with corrupted HSI files."""
    print(f"Loading dataset from {input_csv}...")
    df = pd.read_csv(input_csv)
    print(f"Loaded {len(df)} samples")

    valid_indices = []
    for i, row in df.iterrows():
        if (i + 1) % 1000 == 0:
            print(f"Processed {i + 1}/{len(df)} samples")

        if is_hsi_valid(row["hsi_path"], bands, nodata_threshold):
            valid_indices.append(i)

    clean_df = df.iloc[valid_indices].reset_index(drop=True)
    clean_df.to_csv(output_csv, index=False)

    print(
        f"Filtered {len(df)} -> {len(clean_df)} samples ({len(df) - len(clean_df)} removed)"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Filter corrupted HSI files from dataset"
    )
    parser.add_argument("input_csv", help="Input CSV file")
    parser.add_argument("output_csv", help="Output CSV file")
    parser.add_argument(
        "--bands", type=int, default=369, help="Expected number of bands (default: 369)"
    )
    parser.add_argument(
        "--nodata-threshold",
        type=float,
        default=90,
        help="Max NoData percentage (default: 90)",
    )

    args = parser.parse_args()
    filter_corrupted_hsi(
        args.input_csv, args.output_csv, args.bands, args.nodata_threshold
    )
