#!/usr/bin/env python3
"""
Convert NEON VST data CSV to the format expected by the neon_downloader_clean.py script.

This script takes the hand-annotated NEON VST data and converts it to a GeoPackage
format that is compatible with the existing downloader infrastructure.

The downloader expects columns: individual, siteID, plotID, year, center_easting, center_northing, geometry
The VST data has: individualID, siteID, plotID, date, itcEasting, itcNorthing (and lat/lon)

Usage:
    python convert_vst_to_downloader_format.py --input /path/to/neon_vst_data_2024.csv --output vst_crown_metadata.gpkg
"""

import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import argparse
from datetime import datetime
import os


def convert_vst_to_downloader_format(input_csv: str, output_gpkg: str):
    """
    Convert VST CSV data to downloader-compatible GeoPackage format.

    Args:
        input_csv: Path to the input VST CSV file
        output_gpkg: Path to the output GeoPackage file
    """
    print(f"🔄 Converting VST data from {input_csv}")

    # Read the VST data
    df = pd.read_csv(input_csv)
    print(f"📊 Loaded {len(df)} records from VST data")

    # Extract year from date column
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year

    # Rename columns to match downloader expectations
    column_mapping = {
        "individualID": "individual",
        "siteID": "siteID",  # Already correct
        "plotID": "plotID",  # Already correct
        "year": "year",  # Extracted above
        "itcEasting": "center_easting",
        "itcNorthing": "center_northing",
    }

    # Create a new DataFrame with the required columns
    converted_df = df.rename(columns=column_mapping)

    # Add source_file column (not in original data, but expected by downloader)
    converted_df["source_file"] = "neon_vst_data_2024.csv"

    # Create geometry from coordinates using lat/lon for proper CRS
    geometry = [
        Point(lon, lat) for lon, lat in zip(df["itcLongitude"], df["itcLatitude"])
    ]

    # Create GeoDataFrame with WGS84 (EPSG:4326) initially
    gdf = gpd.GeoDataFrame(
        converted_df[
            [
                "individual",
                "siteID",
                "plotID",
                "year",
                "center_easting",
                "center_northing",
                "source_file",
            ]
        ],
        geometry=geometry,
        crs="EPSG:4326",
    )

    print(f"📍 Created geometry for {len(gdf)} records")
    print(f"📅 Year range: {gdf['year'].min()} to {gdf['year'].max()}")
    print(f"🏔️  Sites: {sorted(gdf['siteID'].unique())}")

    # Summary statistics
    site_counts = gdf.groupby(["siteID", "year"]).size().reset_index(name="count")
    print(f"\n📈 Records per site/year:")
    for _, row in site_counts.head(10).iterrows():  # Show first 10 for brevity
        print(f"  {row['siteID']} {row['year']}: {row['count']} crowns")
    if len(site_counts) > 10:
        print(f"  ... and {len(site_counts) - 10} more site/year combinations")

    # Save to GeoPackage
    print(f"\n💾 Saving to {output_gpkg}")
    gdf.to_file(output_gpkg, driver="GPKG")

    print(f"✅ Conversion complete!")
    print(f"📁 Output saved to: {output_gpkg}")
    print(f"🎯 Ready for use with neon_downloader_clean.py")

    return output_gpkg


def main():
    parser = argparse.ArgumentParser(
        description="Convert NEON VST data CSV to downloader-compatible GeoPackage format"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="/blue/azare/riteshchowdhry/Macrosystems/Data_files/hand_annotated_neon/neon_vst_data_2024.csv",
        help="Input VST CSV file path",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/blue/azare/riteshchowdhry/Macrosystems/code/NeonTreeClassification/neon_tree_classification/data/shapefile_processing/vst_crown_metadata.gpkg",
        help="Output GeoPackage file path",
    )

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"❌ Input file not found: {args.input}")
        return 1

    try:
        convert_vst_to_downloader_format(args.input, args.output)
        return 0
    except Exception as e:
        print(f"❌ Error during conversion: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
