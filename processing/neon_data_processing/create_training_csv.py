#!/usr/bin/env python3
"""
Create a training CSV that combines crown crop paths with species labels.

This script combines cropped crown data with NEON VST species labels to create
a training-ready CSV for machine learning models.

Usage:
    python create_training_csv.py \
        --crop_metadata /path/to/crop_metadata.csv \
        --vst_labels /path/to/neon_vst_data.csv \
        --output /path/to/training_data.csv
"""

import argparse
import pandas as pd
import re
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Create training CSV for NEON tree classification"
    )
    parser.add_argument(
        "--crop_metadata", required=True, help="Path to crop metadata CSV"
    )
    parser.add_argument("--vst_labels", required=True, help="Path to VST labels CSV")
    parser.add_argument("--output", required=True, help="Output CSV path")
    parser.add_argument(
        "--verify_files", action="store_true", help="Verify image files exist"
    )

    args = parser.parse_args()

    print("🌲 Creating training CSV for NEON tree classification...")

    # Load crop metadata
    print("📁 Loading crop metadata...")
    crops_df = pd.read_csv(args.crop_metadata)
    print(f"Total crop entries: {len(crops_df)}")

    # Check if this is new NPY format or old format
    is_npy_format = "rgb_npy_path" in crops_df.columns

    if is_npy_format:
        print("🆕 Detected NPY metadata format")
        # Filter for valid crowns (all modalities valid)
        if "all_modalities_valid" in crops_df.columns:
            crops_df = crops_df[crops_df["all_modalities_valid"] == True]
            print(f"Valid crops (all modalities): {len(crops_df)}")

        # Extract individual ID from crown_id (format: SITE_YEAR_INDIVIDUAL_INDEX)
        def extract_individual_from_crown_id(crown_id):
            try:
                # Split by underscore and take the individual part
                parts = crown_id.split("_")
                if len(parts) >= 4:
                    # Format: SITE_YEAR_INDIVIDUAL_INDEX
                    individual = parts[2]  # Third part is individual ID
                    return individual
                return None
            except:
                return None

        crops_df["individual"] = crops_df["crown_id"].apply(
            extract_individual_from_crown_id
        )

        # Create the format expected by rest of script
        complete_crowns = crops_df[
            ["crown_id", "individual", "rgb_npy_path", "hsi_npy_path", "lidar_npy_path"]
        ].copy()
        complete_crowns = complete_crowns.rename(
            columns={
                "rgb_npy_path": "rgb_path",
                "hsi_npy_path": "hsi_path",
                "lidar_npy_path": "lidar_path",
            }
        )

        # Extract site and year from crown_id if needed
        def extract_site_year(crown_id):
            try:
                parts = crown_id.split("_")
                if len(parts) >= 2:
                    return parts[0], int(parts[1])  # site, year
                return None, None
            except:
                return None, None

        complete_crowns[["site", "year"]] = crops_df["crown_id"].apply(
            lambda x: pd.Series(extract_site_year(x))
        )

    else:
        print("📜 Detected original metadata format")
        # Filter successful crops (original logic)
        if "success" in crops_df.columns:
            crops_df = crops_df[crops_df["success"] == True]
            print(f"Successful crops: {len(crops_df)}")

        # Pivot to get paths for each modality (original logic)
        print("🔄 Pivoting crop data by modality...")
        index_cols = ["crown_id", "individual"]
        if "site" in crops_df.columns:
            index_cols.append("site")
        if "year" in crops_df.columns:
            index_cols.append("year")
        # Include spatial coordinates if available
        if "tile_x" in crops_df.columns:
            index_cols.append("tile_x")
        if "tile_y" in crops_df.columns:
            index_cols.append("tile_y")

        crops_pivot = crops_df.pivot_table(
            index=index_cols,
            columns="modality",
            values="output_path",
            aggfunc="first",
        ).reset_index()

        # Rename columns
        crops_pivot.columns.name = None
        if "hsi" in crops_pivot.columns:
            crops_pivot = crops_pivot.rename(columns={"hsi": "hsi_path"})
        if "lidar" in crops_pivot.columns:
            crops_pivot = crops_pivot.rename(columns={"lidar": "lidar_path"})
        if "rgb" in crops_pivot.columns:
            crops_pivot = crops_pivot.rename(columns={"rgb": "rgb_path"})

        complete_crowns = crops_pivot
    print(f"Unique crowns: {len(complete_crowns)}")

    # Filter crowns with all 3 modalities
    path_cols = [
        col
        for col in ["rgb_path", "hsi_path", "lidar_path"]
        if col in complete_crowns.columns
    ]

    if not is_npy_format:
        complete_crowns = complete_crowns.dropna(subset=path_cols)

    print(f"Crowns with all modalities: {len(complete_crowns)}")

    # Validate NEON IDs
    print("✅ Validating NEON IDs...")
    neon_pattern = re.compile(r"^NEON\.PLA\.D\d{2}\.[A-Z]{4}\.\d+[A-Z]?$")
    valid_ids = complete_crowns["individual"].apply(
        lambda x: isinstance(x, str) and bool(neon_pattern.fullmatch(x))
    )
    complete_crowns = complete_crowns[valid_ids].reset_index(drop=True)
    print(f"Valid NEON IDs: {len(complete_crowns)}")

    # Load labels
    print("🏷️ Loading species labels...")
    labels_df = pd.read_csv(args.vst_labels)
    print(f"Total label entries: {len(labels_df)}")

    # Keep most recent per individual
    if "date" in labels_df.columns:
        labels_df = labels_df.sort_values(["individualID", "date"]).drop_duplicates(
            "individualID", keep="last"
        )
    else:
        labels_df = labels_df.drop_duplicates("individualID", keep="last")
    print(f"Unique individuals: {len(labels_df)}")

    # Merge with labels
    print("🔗 Merging with species labels...")
    label_cols = ["individualID", "taxonID", "scientificName"]
    optional_cols = [
        "siteID",
        "height",
        "stemDiameter",
        "canopyPosition",
        "plantStatus",
    ]
    for col in optional_cols:
        if col in labels_df.columns:
            label_cols.append(col)

    final_df = complete_crowns.merge(
        labels_df[label_cols],
        left_on="individual",
        right_on="individualID",
        how="inner",
    )
    print(f"Crowns with labels: {len(final_df)}")

    # Clean up column names
    final_df = final_df.rename(
        columns={
            "individualID": "individual_id",
            "taxonID": "species",  # Changed from 'species_code' to 'species'
            "scientificName": "species_name",
            "siteID": "label_site",  # to distinguish from crop site if different
            "tile_x": "easting",  # Use tile coordinates as easting
            "tile_y": "northing",  # Use tile coordinates as northing
        }
    )

    # Add plot column (extract from crown_id or set as unknown)
    if "plot" not in final_df.columns:
        # Try to extract plot from crown_id if pattern exists, otherwise use 'unknown'
        final_df["plot"] = "unknown"  # Default value
        # You could add plot extraction logic here if your crown_id contains plot info

    # Verify files exist if requested
    if args.verify_files:
        print("📂 Verifying file paths...")
        for col in path_cols:
            exists = final_df[col].apply(lambda x: Path(x).exists())
            missing = (~exists).sum()
            if missing > 0:
                print(f"⚠️ {missing} {col.replace('_path', '').upper()} files missing")

        # Keep only rows where all files exist
        all_exist = True
        for col in path_cols:
            all_exist = all_exist & final_df[col].apply(lambda x: Path(x).exists())
        final_df = final_df[all_exist]
        print(f"Final count with existing files: {len(final_df)}")

    # Sort by site and species
    sort_cols = []
    for col in ["site", "species_code", "crown_id"]:
        if col in final_df.columns:
            sort_cols.append(col)
    if sort_cols:
        final_df = final_df.sort_values(sort_cols).reset_index(drop=True)

    # Print summary
    print(f"\n📊 Dataset Summary:")
    print(f"Total samples: {len(final_df)}")
    if "species_code" in final_df.columns:
        print(f"Unique species: {final_df['species_code'].nunique()}")
    if "site" in final_df.columns:
        print(f"Sites: {final_df['site'].nunique()}")
    if "year" in final_df.columns:
        print(f"Years: {sorted(final_df['year'].unique())}")

    # Save CSV
    print(f"\n💾 Saving to: {args.output}")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(args.output, index=False)

    print("✅ Training CSV created successfully!")


if __name__ == "__main__":
    main()
