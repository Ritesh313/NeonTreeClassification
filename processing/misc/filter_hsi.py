#!/usr/bin/env python3
"""
Simple HSI corruption filter - removes samples with corrupted HSI files.
"""

import pandas as pd
import rasterio
import numpy as np
from pathlib import Path


def filter_corrupted_hsi(input_csv, output_csv):
    """
    Simple function to filter out samples with corrupted HSI files.

    Args:
        input_csv: Path to input dataset CSV
        output_csv: Path to save filtered dataset CSV
    """
    # Load dataset
    df = pd.read_csv(input_csv)

    # Check each HSI file
    valid_indices = []
    for i, row in df.iterrows():
        hsi_path = row["hsi_path"]
        if is_hsi_valid(hsi_path):
            valid_indices.append(i)

    # Filter to keep only valid samples
    clean_df = df.iloc[valid_indices].reset_index(drop=True)

    # Save clean dataset
    clean_df.to_csv(output_csv, index=False)

    print(
        f"Filtered {len(df)} -> {len(clean_df)} samples ({len(df) - len(clean_df)} removed)"
    )


def is_hsi_valid(hsi_path):
    """Check if HSI file is valid (not corrupted)."""
    try:
        if not Path(hsi_path).exists():
            return False

        with rasterio.open(hsi_path) as src:
            # Basic checks
            if src.count != 369:  # Should have 369 bands
                return False
            if src.height < 2 or src.width < 2:  # Should have reasonable dimensions
                return False

            # Check for excessive NoData
            band_data = src.read(1)  # Check first band
            if src.nodata is not None:
                nodata_pct = np.sum(band_data == src.nodata) / band_data.size * 100
            else:
                # Check for common nodata values
                nodata_pct = (
                    np.sum((band_data == -9999) | np.isnan(band_data))
                    / band_data.size
                    * 100
                )

            if nodata_pct > 90:  # Reject if >90% NoData
                return False

        return True
    except:
        return False


if __name__ == "__main__":
    input_csv = "/blue/azare/riteshchowdhry/Macrosystems/Data_files/hand_annotated_neon/curated_tiles_20250822/cropped_crowns_modality_organized/training_data_filtered.csv"
    output_csv = "training_data_clean.csv"

    filter_corrupted_hsi(input_csv, output_csv)
