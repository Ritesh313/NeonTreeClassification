#!/usr/bin/env python3
"""
Download NEON data for all modalities (RGB, HSI, LiDAR).

This script downloads NEON data products for multiple modalities
based on site and date specifications.

Usage:
    python scripts/download_neon_all_modalities.py

Author: Ritesh Chowdhry
"""

import argparse
import os
import numpy as np
import pandas as pd
from pathlib import Path
import rpy2
import math
import re
import sys
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from rpy2.rinterface_lib.callbacks import logger as rpy2_logger
import logging
from datetime import datetime

# Configure R logging
rpy2_logger.setLevel(logging.ERROR)

# Import R packages with error handling
try:
    base = importr('base')
    utils = importr('utils')
    stats = importr('stats')
    neonUtilities = importr('neonUtilities')
    print("✅ R packages loaded successfully")
except Exception as e:
    print(f"❌ Error loading R packages: {e}")
    print("Please ensure R is installed and neonUtilities package is available")
    print("Install neonUtilities with: install.packages('neonUtilities') in R")
    sys.exit(1)

# Global data product codes
RGB_PRODUCT = 'DP3.30010.001'
HSI_PRE2022_PRODUCT = 'DP3.30006.001'  # Pre-2022 HSI without BRDF
HSI_2022_PRODUCT = 'DP3.30006.002'     # 2022+ HSI with BRDF
LIDAR_PRODUCT = 'DP3.30015.001'


def get_hsi_product_code(year):
    """
    Get the correct HSI product code based on year
    
    Args:
        year: Year as string or int (handles formats like "2019" or "2019 (1)")
        
    Returns:
        str: Appropriate HSI product code
    """
    # Extract numeric year from formats like "2019 (1)" or just "2019"
    year_match = re.search(r'(\d{4})', str(year))
    if year_match:
        year_int = int(year_match.group(1))
    else:
        year_int = int(year)  # fallback for simple integer years
        
    if year_int < 2022:
        return HSI_PRE2022_PRODUCT
    else:
        return HSI_2022_PRODUCT

def rgb_data_download(easting, northing, site, year, output_dir):
    """
    Download RGB data from NEON for a specific site and year.
    
    Args:
        easting: List of easting coordinates
        northing: List of northing coordinates
        site: NEON site code (e.g., 'BART')
        year: Year of data to download (e.g., '2019')
        output_dir: Directory to save the downloaded data
    """
    print(f"  Downloading RGB data for {site} {year}...")
    print(f"    Product code: {RGB_PRODUCT}")
    print(f"    Tiles: {len(easting)} tiles")
    print(f"    Output: {output_dir}")
    
    # Convert Python lists to R vectors to fix rpy2 conversion error
    r_easting = robjects.FloatVector(easting)
    r_northing = robjects.FloatVector(northing)
    
    neonUtilities.byTileAOP(
        dpID=RGB_PRODUCT, 
        site=site, 
        year=year,
        check_size=False,
        easting=r_easting, 
        northing=r_northing,
        include_provisional=True,
        savepath=output_dir
    )
    print(f"    ✓ RGB download completed")


def hsi_data_download(easting, northing, site, year, output_dir):
    """
    Download HSI data from NEON for a specific site and year.
    Uses appropriate product code based on year (pre-2022 vs 2022+).
    
    Args:
        easting: List of easting coordinates
        northing: List of northing coordinates
        site: NEON site code (e.g., 'BART')
        year: Year of data to download (e.g., '2019')
        output_dir: Directory to save the downloaded data
    """
    hsi_product = get_hsi_product_code(year)
    print(f"  Downloading HSI data for {site} {year}...")
    print(f"    Product code: {hsi_product}")
    print(f"    Tiles: {len(easting)} tiles")
    print(f"    Output: {output_dir}")
    
    # Convert Python lists to R vectors to fix rpy2 conversion error
    r_easting = robjects.FloatVector(easting)
    r_northing = robjects.FloatVector(northing)
    
    neonUtilities.byTileAOP(
        dpID=hsi_product, 
        site=site, 
        year=year,
        check_size=False,
        easting=r_easting, 
        northing=r_northing,
        include_provisional=True,
        savepath=output_dir
    )
    print(f"    ✓ HSI download completed")


def lidar_chm_data_download(easting, northing, site, year, output_dir):
    """
    Download LiDAR CHM data from NEON for a specific site and year.
    
    Args:
        easting: List of easting coordinates
        northing: List of northing coordinates
        site: NEON site code (e.g., 'BART')
        year: Year of data to download (e.g., '2019')
        output_dir: Directory to save the downloaded data
    """
    print(f"  Downloading LiDAR data for {site} {year}...")
    print(f"    Product code: {LIDAR_PRODUCT}")
    print(f"    Tiles: {len(easting)} tiles")
    print(f"    Output: {output_dir}")
    
    # Convert Python lists to R vectors to fix rpy2 conversion error
    r_easting = robjects.FloatVector(easting)
    r_northing = robjects.FloatVector(northing)
    
    neonUtilities.byTileAOP(
        dpID=LIDAR_PRODUCT, 
        site=site, 
        year=year,
        check_size=False,
        easting=r_easting, 
        northing=r_northing,
        include_provisional=True,
        savepath=output_dir
    )
    print(f"    ✓ LiDAR download completed")

def download_site_data(site, year, eastings, northings, output_base_dir, modalities=['RGB', 'HSI', 'LiDAR']):
    """
    Download all requested modalities for a specific site and year.
    
    Args:
        site: NEON site code (e.g., 'BART')
        year: Year of data to download (e.g., '2019')
        eastings: List of easting coordinates
        northings: List of northing coordinates
        output_base_dir: Base directory to save downloaded data
        modalities: List of modalities to download ['RGB', 'HSI', 'LiDAR']
    """
    print(f"\n{'='*60}")
    print(f"DOWNLOADING DATA FOR {site} {year}")
    print(f"{'='*60}")
    print(f"Modalities: {', '.join(modalities)}")
    print(f"Coordinates: {len(eastings)} easting, {len(northings)} northing values")
    print(f"Output directory: {output_base_dir}")
    
    # Validate coordinates
    if len(eastings) != len(northings):
        raise ValueError(f"Easting and northing coordinate lists must have same length: {len(eastings)} vs {len(northings)}")
    
    if any(math.isnan(x) for x in eastings + northings):
        raise ValueError("Found NaN values in coordinates")
    
    # Create modality directories
    download_dirs = {}
    for modality in modalities:
        modality_dir = os.path.join(output_base_dir, modality, f"{site}_{year}")
        Path(modality_dir).mkdir(parents=True, exist_ok=True)
        download_dirs[modality] = modality_dir
    
    # Download functions mapping
    download_functions = {
        'RGB': rgb_data_download,
        'HSI': hsi_data_download,
        'LiDAR': lidar_chm_data_download
    }
    
    # Download each modality
    success_count = 0
    for modality in modalities:
        if modality not in download_functions:
            print(f"⚠️  Unknown modality: {modality}. Skipping.")
            continue
            
        try:
            download_functions[modality](eastings, northings, site, year, download_dirs[modality])
            success_count += 1
        except Exception as e:
            print(f"❌ Error downloading {modality} data for {site} {year}: {e}")
    
    print(f"\n✅ Download completed: {success_count}/{len(modalities)} modalities successful")
    return success_count == len(modalities)

def process_csv_and_download_data(csv_path, output_base_dir, modalities=['RGB', 'HSI', 'LiDAR'], 
                                  filter_site=None, filter_year=None):
    """
    Process a CSV file containing tree crown metadata and download corresponding 
    NEON data tiles for specified modalities.
    
    Args:
        csv_path: Path to the CSV file with tree crown metadata
        output_base_dir: Base directory to save downloaded data
        modalities: List of modalities to download ['RGB', 'HSI', 'LiDAR']
        filter_site: Optional site code to download only specific site (e.g., 'BART')
        filter_year: Optional year to download only specific year (e.g., 2019)
    """
    print(f"📄 PROCESSING CSV: {csv_path}")
    print(f"🎯 TARGET MODALITIES: {', '.join(modalities)}")
    print(f"📁 OUTPUT DIRECTORY: {output_base_dir}")
    
    # Read the CSV file
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    print(f"📊 Total records in CSV: {len(df)}")
    
    # Apply filters if specified
    if filter_site:
        df = df[df['site'] == filter_site]
        print(f"🔍 Filtered to site {filter_site}: {len(df)} records")
    
    if filter_year:
        df = df[df['year'] == filter_year]
        print(f"🔍 Filtered to year {filter_year}: {len(df)} records")
    
    if len(df) == 0:
        print("❌ No records found after filtering. Exiting.")
        return
    
    # Clean up column names
    df.columns = [col.lower().strip() for col in df.columns]
    
    # Find coordinate columns with improved detection
    easting_cols = [col for col in df.columns if re.search(r'(center_easting|easting|utm_x|x_coord)', col, re.IGNORECASE)]
    northing_cols = [col for col in df.columns if re.search(r'(center_northing|northing|utm_y|y_coord)', col, re.IGNORECASE)]
    
    if not easting_cols or not northing_cols:
        print(f"Available columns: {list(df.columns)}")
        raise ValueError("Could not find easting/northing columns in the CSV file")
    
    easting_col = easting_cols[0]
    northing_col = northing_cols[0]
    
    print(f"📍 Using coordinate columns: {easting_col} and {northing_col}")
    
    # Ensure required columns exist
    required_cols = ['site', 'year']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in CSV file")
    
    # Convert year to string
    df['year'] = df['year'].astype(str)
    
    # Clean data: Remove rows with NaN coordinates
    original_count = len(df)
    df = df.dropna(subset=[easting_col, northing_col])
    if len(df) < original_count:
        print(f"⚠️  Removed {original_count - len(df)} rows with missing coordinates")
    
    # Group by site and year
    site_year_groups = df.groupby(['site', 'year'])
    sites_years = list(site_year_groups.groups.keys())
    
    print(f"🏗️  Found {len(sites_years)} site-year combinations:")
    for site, year in sites_years:
        count = len(site_year_groups.get_group((site, year)))
        print(f"  • {site} {year}: {count} records")
    
    # Process each site-year group
    successful_downloads = 0
    total_downloads = len(sites_years)
    
    for (site, year), group in site_year_groups:
        print(f"\n📍 Processing site: {site}, year: {year} ({len(group)} records)")
        
        # Extract and validate coordinates
        eastings = group[easting_col].values
        northings = group[northing_col].values
        
        # Ensure coordinates are numeric
        try:
            eastings = np.array([float(x) for x in eastings])
            northings = np.array([float(x) for x in northings])
        except ValueError as e:
            print(f"❌ Error converting coordinates for {site} {year}: {e}. Skipping.")
            continue
        
        # Remove NaN values
        valid_indices = ~(np.isnan(eastings) | np.isnan(northings))
        if not np.all(valid_indices):
            print(f"⚠️  Found {np.sum(~valid_indices)} invalid coordinates. Removing.")
            eastings = eastings[valid_indices]
            northings = northings[valid_indices]
        
        if len(eastings) == 0:
            print(f"❌ No valid coordinates found for {site} {year}. Skipping.")
            continue
        
        # Calculate NEON tile coordinates
        tile_size = 1000  # 1km tiles
        tile_eastings = np.unique(np.round(eastings / tile_size) * tile_size)
        tile_northings = np.unique(np.round(northings / tile_size) * tile_size)
        
        # Generate all tile coordinate pairs
        tile_pairs = []
        for e in tile_eastings:
            for n in tile_northings:
                tile_pairs.append((e, n))
        
        download_eastings = [p[0] for p in tile_pairs]
        download_northings = [p[1] for p in tile_pairs]
        
        print(f"📊 Will download {len(tile_pairs)} tiles")
        print(f"   Easting range: {min(tile_eastings):.0f} to {max(tile_eastings):.0f}")
        print(f"   Northing range: {min(tile_northings):.0f} to {max(tile_northings):.0f}")
        
        # Download the data
        success = download_site_data(
            site=site,
            year=year,
            eastings=download_eastings,
            northings=download_northings,
            output_base_dir=output_base_dir,
            modalities=modalities
        )
        
        if success:
            successful_downloads += 1
    
    # Final summary
    print(f"\n🏁 DOWNLOAD SUMMARY")
    print(f"{'='*50}")
    print(f"Successful downloads: {successful_downloads}/{total_downloads}")
    print(f"Data saved to: {output_base_dir}")
    
    if successful_downloads == total_downloads:
        print("🎉 All downloads completed successfully!")
    elif successful_downloads > 0:
        print("⚠️  Some downloads failed. Check logs above.")
    else:
        print("❌ All downloads failed. Check your configuration.")
    
    return successful_downloads

def main():
    """Main function with command line arguments for flexibility."""
    parser = argparse.ArgumentParser(description='Download NEON data for multiple sites and years')
    parser.add_argument('--mode', choices=['test', 'full', 'custom'], default='test',
                       help='Download mode: test (BART 2019), full (all data), or custom')
    parser.add_argument('--site', help='Specific site to download (for custom mode)')
    parser.add_argument('--year', help='Specific year to download (for custom mode)')
    parser.add_argument('--csv-path', 
                       default='/blue/azare/riteshchowdhry/Macrosystems/Data_files/hand_annotated_neon/clean_annotations/neon_sites_coordinates_CORRECTED_CLEAN.csv',
                       help='Path to crown coordinate CSV file')
    parser.add_argument('--output-dir',
                       default='/blue/azare/riteshchowdhry/Macrosystems/Data_files/hand_annotated_neon/neon_tiles',
                       help='Base output directory')
    parser.add_argument('--modalities', nargs='+', default=['RGB', 'HSI', 'LiDAR'],
                       help='Modalities to download')
    
    args = parser.parse_args()
    
    if args.mode == 'test':
        print("🧪 TEST MODE: Downloading BART 2019 only")
        process_csv_and_download_data(
            csv_path=args.csv_path,
            output_base_dir=args.output_dir + '_test', 
            modalities=args.modalities,
            filter_site='BART',
            filter_year='2019'
        )
    elif args.mode == 'full':
        print("🏗️  FULL MODE: Processing all sites and years")
        process_csv_and_download_data(
            csv_path=args.csv_path,
            output_base_dir=args.output_dir + '_full',
            modalities=args.modalities
        )
    elif args.mode == 'custom':
        print(f"🎯 CUSTOM MODE: Site={args.site}, Year={args.year}")
        process_csv_and_download_data(
            csv_path=args.csv_path,
            output_base_dir=args.output_dir + '_custom',
            modalities=args.modalities,
            filter_site=args.site,
            filter_year=args.year
        )

if __name__ == "__main__":
    main()
