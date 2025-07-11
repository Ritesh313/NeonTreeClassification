#!/usr/bin/env python3
"""
NEON Tree Crown Processing Pipeline

This script processes NEON HSI tiles and crown annotation data to create
training datasets for tree classification models.

Usage:
    python scripts/process_tiles_to_crowns.py

Author: Ritesh Chowdhry
"""

import os
import glob
import re
import h5py
import numpy as np
import pandas as pd
import geopandas as gpd
import warnings
import rasterio
from rasterio.transform import Affine
from rasterio.mask import mask
import cv2

# Suppress warnings
warnings.filterwarnings("ignore")

def extract_coordinates(filename):
    """
    Extract coordinates from NEON tile filenames.
    Handles various NEON file naming conventions.
    """
    basename = os.path.basename(filename)
    
    if basename.startswith('NEON_'):
        # NEON RGB: 2019_BART_5_314000_4878000_image.tif
        match = re.search(r'(\d+)_(\d+)_image\.tif$', basename)
        if match:
            return int(match.group(1)), int(match.group(2))
        # NEON HSI: NEON_D01_BART_DP3_317000_4882000_reflectance.h5
        match = re.search(r'(\d+)_(\d+)_reflectance\.h5$', basename)
        if match:
            return int(match.group(1)), int(match.group(2))
        # NEON LiDAR: NEON_D01_BART_DP3_319000_4881000_CHM.tif
        match = re.search(r'(\d+)_(\d+)_CHM\.tif$', basename)
        if match:
            return int(match.group(1)), int(match.group(2))
    
    return None

def organize_downloaded_tiles(tiles_base_dir):
    """
    Organize tiles by coordinate pairs across all modalities.
    
    Args:
        tiles_base_dir: Base directory containing tile subdirectories
        
    Returns:
        dict: Nested dictionary with coordinate pairs as keys
    """
    print("ORGANIZING DOWNLOADED TILES")
    print("="*50)
    
    # Find tile directories
    tile_inventory = {}
    
    # Look for RGB tiles (in RGB subdirectory)
    rgb_dir = os.path.join(tiles_base_dir, 'RGB')
    rgb_files = []
    if os.path.exists(rgb_dir):
        for root, dirs, files in os.walk(rgb_dir):
            rgb_files.extend([os.path.join(root, f) for f in files if f.endswith('.tif')])
    
    # Look for HSI tiles (in HSI subdirectory)  
    hsi_dir = os.path.join(tiles_base_dir, 'HSI')
    hsi_files = []
    if os.path.exists(hsi_dir):
        for root, dirs, files in os.walk(hsi_dir):
            hsi_files.extend([os.path.join(root, f) for f in files if f.endswith('.h5')])
    
    # Look for LiDAR tiles (in LiDAR subdirectory)
    lidar_dir = os.path.join(tiles_base_dir, 'LiDAR') 
    lidar_files = []
    if os.path.exists(lidar_dir):
        for root, dirs, files in os.walk(lidar_dir):
            lidar_files.extend([os.path.join(root, f) for f in files if f.endswith('.tif')])
    
    print(f"Found {len(rgb_files)} RGB files")
    print(f"Found {len(hsi_files)} HSI files") 
    print(f"Found {len(lidar_files)} LiDAR files")
    
    # Process each modality
    for file_path in rgb_files:
        coords = extract_coordinates(file_path)
        if coords:
            x, y = coords
            if (x, y) not in tile_inventory:
                tile_inventory[(x, y)] = {}
            tile_inventory[(x, y)]['rgb'] = file_path
            
    for file_path in hsi_files:
        coords = extract_coordinates(file_path)
        if coords:
            x, y = coords
            if (x, y) not in tile_inventory:
                tile_inventory[(x, y)] = {}
            tile_inventory[(x, y)]['hsi'] = file_path
            
    for file_path in lidar_files:
        coords = extract_coordinates(file_path)
        if coords:
            x, y = coords
            if (x, y) not in tile_inventory:
                tile_inventory[(x, y)] = {}
            tile_inventory[(x, y)]['lidar'] = file_path
    
    # Print summary
    print(f"\nTILE INVENTORY:")
    print(f"Total unique coordinate pairs: {len(tile_inventory)}")
    
    # Count by modality
    rgb_count = sum(1 for tile_data in tile_inventory.values() if 'rgb' in tile_data)
    hsi_count = sum(1 for tile_data in tile_inventory.values() if 'hsi' in tile_data)
    lidar_count = sum(1 for tile_data in tile_inventory.values() if 'lidar' in tile_data)
    
    print(f"  RGB tiles: {rgb_count}")
    print(f"  HSI tiles: {hsi_count}")
    print(f"  LiDAR tiles: {lidar_count}")
    
    # Count tiles with all 3 modalities
    complete_tiles = sum(1 for tile_data in tile_inventory.values() 
                        if all(mod in tile_data for mod in ['rgb', 'hsi', 'lidar']))
    print(f"  All 3 modalities: {complete_tiles}")
    
    return tile_inventory

def convert_hsi_tile_to_tif(h5_file, output_dir, site, year, coords):
    """
    Convert a single HSI H5 file to GeoTIFF format
    Adapted from your existing function
    """
    os.makedirs(output_dir, exist_ok=True)
    output_file = f"{year}_{site}_{coords}_HSI.tif"
    
    # Extract metadata and reflectance data
    with h5py.File(h5_file, 'r') as hdf:
        # Get sitename from file attributes
        sitename = list(hdf.keys())[0]
        
        # Get reflectance data and metadata
        refl_group = hdf[sitename]['Reflectance']
        refl_data = refl_group['Reflectance_Data'][:]
        
        # Get metadata
        epsg = str(refl_group['Metadata']['Coordinate_System']['EPSG Code'][()])
        epsg = epsg.split("'")[1] if "'" in epsg else epsg
        
        map_info = str(refl_group['Metadata']['Coordinate_System']['Map_Info'][()])
        map_info = map_info.split(",")
        
        # Get resolution
        pixel_width = float(map_info[5])
        pixel_height = float(map_info[6])
        
        # Get corner coordinates
        x_min = float(map_info[3])
        y_max = float(map_info[4])
        
        # Delete water absorption bands
        band_indices = np.r_[0:425]
        band_indices = np.delete(band_indices, np.r_[419:425])
        band_indices = np.delete(band_indices, np.r_[283:315])
        band_indices = np.delete(band_indices, np.r_[192:210])
        refl_data = refl_data[:, :, band_indices]
        
        # Scale factor
        scale_factor = float(refl_group['Reflectance_Data'].attrs['Scale_Factor'])
        no_data_val = float(refl_group['Reflectance_Data'].attrs['Data_Ignore_Value'])
    
    # Create geotransform for the raster
    transform = Affine.translation(x_min, y_max) * Affine.scale(pixel_width, -pixel_height)
    
    # Rearrange dimensions for rasterio (bands, rows, cols)
    refl_data = np.moveaxis(refl_data, 2, 0)
    
    # Write to GeoTIFF
    output_path = os.path.join(output_dir, output_file)
    with rasterio.open(
        output_path,
        'w',
        driver='GTiff',
        height=refl_data.shape[1],
        width=refl_data.shape[2],
        count=refl_data.shape[0],
        dtype=refl_data.dtype,
        crs=f'EPSG:{epsg}',
        transform=transform,
        nodata=no_data_val
    ) as dst:
        dst.write(refl_data)
    
    return output_path

def convert_all_hsi_tiles(tile_inventory, site, year, output_base_dir):
    """
    Convert all HSI tiles to TIF format
    
    Args:
        tile_inventory: Dictionary from organize_downloaded_tiles
        site: NEON site code
        year: Year string
        output_base_dir: Base directory for converted files
    
    Returns:
        dict: Updated tile inventory with HSI TIF paths
    """
    print(f"CONVERTING HSI TILES TO TIF FORMAT")
    print("="*50)
    
    hsi_tif_dir = os.path.join(output_base_dir, 'HSI_converted', f'{site}_{year}')
    os.makedirs(hsi_tif_dir, exist_ok=True)
    
    converted_count = 0
    updated_inventory = tile_inventory.copy()
    
    for (tile_x, tile_y), tile_data in tile_inventory.items():
        if 'hsi' in tile_data:
            h5_file = tile_data['hsi']
            coords = f"{tile_x}_{tile_y}"
            
            # Check if TIF already exists
            output_file = f"{year}_{site}_{coords}_HSI.tif"
            output_path = os.path.join(hsi_tif_dir, output_file)
            
            if os.path.exists(output_path):
                print(f"  Already exists: {coords}")
                updated_inventory[(tile_x, tile_y)]['hsi_tif'] = output_path
                converted_count += 1
                continue
            
            try:
                tif_file = convert_hsi_tile_to_tif(
                    h5_file=h5_file,
                    output_dir=hsi_tif_dir,
                    site=site,
                    year=year,
                    coords=coords
                )
                
                # Update inventory with TIF path
                updated_inventory[(tile_x, tile_y)]['hsi_tif'] = tif_file
                converted_count += 1
                print(f"  Converted: {coords}")
                
            except Exception as e:
                print(f"  Error converting {coords}: {e}")
    
    print(f"Successfully converted {converted_count} HSI tiles to TIF format")
    return updated_inventory

def load_crown_data(shapefile_path, site, year):
    """
    Load crown polygons from shapefile for specific site and year
    
    Args:
        shapefile_path: Path to cleaned CSV with crown coordinates
        site: NEON site code
        year: Year as integer
    
    Returns:
        GeoDataFrame: Crown polygons with metadata
    """
    print(f"LOADING CROWN DATA FOR {site} {year}")
    print("="*50)
    
    # Load the cleaned CSV
    df = pd.read_csv(shapefile_path)
    
    # Filter for specific site and year (handle both "2019" and "2019 (1)" formats)
    year_str = str(year)
    site_data = df[(df['site'] == site) & (df['year'].astype(str).str.startswith(year_str))].copy()
    
    print(f"Found {len(site_data)} crown records for {site} {year}")
    
    if len(site_data) == 0:
        return None
    
    # Convert to GeoDataFrame using the coordinate bounds
    # This is a simplified approach - for full processing you'd want to load the actual shapefiles
    geometries = []
    for _, row in site_data.iterrows():
        from shapely.geometry import box
        geom = box(row['min_easting'], row['min_northing'], 
                   row['max_easting'], row['max_northing'])
        geometries.append(geom)
    
    site_data['geometry'] = geometries
    crown_gdf = gpd.GeoDataFrame(site_data, crs=site_data.iloc[0]['target_crs'])
    
    return crown_gdf

def process_crown_tile_intersections(tile_inventory, crown_gdf, site, year):
    """
    Find intersections between tiles and crown polygons
    
    Args:
        tile_inventory: Dictionary of tile paths by coordinate
        crown_gdf: GeoDataFrame of crown polygons
        site: NEON site code
        year: Year string
        
    Returns:
        dict: Mapping of tiles to overlapping crowns
    """
    print(f"FINDING TILE-CROWN INTERSECTIONS")
    print("="*50)
    
    intersections = {}
    
    for (tile_x, tile_y), tile_data in tile_inventory.items():
        # Create tile polygon (1km x 1km)
        from shapely.geometry import box
        tile_polygon = box(tile_x, tile_y, tile_x + 1000, tile_y + 1000)
        
        # Find intersecting crowns
        intersecting_crowns = crown_gdf[crown_gdf.intersects(tile_polygon)]
        
        if len(intersecting_crowns) > 0:
            intersections[(tile_x, tile_y)] = {
                'tile_data': tile_data,
                'crowns': intersecting_crowns,
                'num_crowns': len(intersecting_crowns)
            }
            print(f"  Tile {tile_x}_{tile_y}: {len(intersecting_crowns)} crowns")
    
    print(f"Found {len(intersections)} tiles with crown overlaps")
    return intersections

def crop_tile_to_crown(tile_path, crown_polygon, output_path, tile_type='RGB'):
    """
    Crop a tile to crown polygon boundaries
    
    Args:
        tile_path: Path to tile file
        crown_polygon: Shapely polygon of crown boundary
        output_path: Where to save cropped tile
        tile_type: Type of tile ('RGB', 'HSI', 'LiDAR')
        
    Returns:
        bool: Success status
    """
    try:
        if not os.path.exists(tile_path):
            print(f"Warning: Tile file not found: {tile_path}")
            return False
            
        with rasterio.open(tile_path) as src:
            # Crop the raster to crown polygon
            out_image, out_transform = mask(src, [crown_polygon], crop=True)
            out_meta = src.meta.copy()
            
            # Update metadata
            out_meta.update({
                "driver": "GTiff",
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": out_transform
            })
            
            # Write cropped tile
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with rasterio.open(output_path, "w", **out_meta) as dest:
                dest.write(out_image)
        
        return True
        
    except Exception as e:
        print(f"Error cropping tile {tile_path}: {e}")
        return False

def process_all_crowns(tile_inventory, crown_gdf, site, year, output_base_dir):
    """
    Process all crown-tile intersections and create cropped datasets
    
    Args:
        tile_inventory: Dictionary of tile paths
        crown_gdf: GeoDataFrame of crown polygons  
        site: NEON site code
        year: Year string
        output_base_dir: Base output directory
        
    Returns:
        dict: Processing results summary
    """
    print(f"PROCESSING ALL CROWN-TILE INTERSECTIONS")
    print("="*50)
    
    # Find intersections
    intersections = process_crown_tile_intersections(tile_inventory, crown_gdf, site, year)
    
    if len(intersections) == 0:
        print("No tile-crown intersections found!")
        return {}
    
    # Process each intersection
    results = {
        'total_intersections': len(intersections),
        'successful_crops': 0,
        'failed_crops': 0,
        'crown_datasets': []
    }
    
    crown_output_dir = os.path.join(output_base_dir, 'crown_datasets', f'{site}_{year}')
    
    for (tile_x, tile_y), intersection_data in intersections.items():
        tile_data = intersection_data['tile_data']
        crowns = intersection_data['crowns']
        
        for crown_idx, crown_row in crowns.iterrows():
            crown_id = f"{site}_{crown_row['plot']}_{crown_row['year']}_{crown_idx}"
            crown_dir = os.path.join(crown_output_dir, crown_id)
            
            # Process each modality
            success_count = 0
            
            # RGB
            if 'rgb' in tile_data:
                rgb_output = os.path.join(crown_dir, f"{crown_id}_RGB.tif")
                if crop_tile_to_crown(tile_data['rgb'], crown_row['geometry'], rgb_output, 'RGB'):
                    success_count += 1
            
            # HSI (converted TIF)
            if 'hsi_tif' in tile_data:
                hsi_output = os.path.join(crown_dir, f"{crown_id}_HSI.tif") 
                if crop_tile_to_crown(tile_data['hsi_tif'], crown_row['geometry'], hsi_output, 'HSI'):
                    success_count += 1
            
            # LiDAR
            if 'lidar' in tile_data:
                lidar_output = os.path.join(crown_dir, f"{crown_id}_LiDAR.tif")
                if crop_tile_to_crown(tile_data['lidar'], crown_row['geometry'], lidar_output, 'LiDAR'):
                    success_count += 1
            
            if success_count > 0:
                results['successful_crops'] += 1
                results['crown_datasets'].append({
                    'crown_id': crown_id,
                    'tile_coords': (tile_x, tile_y),
                    'modalities': success_count,
                    'output_dir': crown_dir
                })
            else:
                results['failed_crops'] += 1
    
    print(f"Successfully processed {results['successful_crops']} crown datasets")
    print(f"Failed to process {results['failed_crops']} crown datasets")
    
    return results

def main_processing_pipeline(tiles_base_dir, crown_csv_path, site, year, output_base_dir):
    """
    Run the complete tile-to-crown processing pipeline
    
    Args:
        tiles_base_dir: Directory containing downloaded tile subdirectories
        crown_csv_path: Path to cleaned crown coordinate CSV
        site: NEON site code (e.g., 'BART')
        year: Year string (e.g., '2019')
        output_base_dir: Base directory for all outputs
        
    Returns:
        dict: Processing results
    """
    print(f"🚀 STARTING FULL TILE PROCESSING PIPELINE")
    print(f"Site: {site}, Year: {year}")
    print("="*60)
    
    # Step 1: Organize downloaded tiles
    tile_inventory = organize_downloaded_tiles(tiles_base_dir)
    
    if len(tile_inventory) == 0:
        print("❌ No tiles found! Check tiles_base_dir path.")
        return
    
    # Step 2: Convert HSI tiles to TIF format
    tile_inventory = convert_all_hsi_tiles(tile_inventory, site, year, output_base_dir)
    
    # Step 3: Load crown data
    crown_gdf = load_crown_data(crown_csv_path, site, int(year))
    
    if crown_gdf is None:
        print("No crown data found. Exiting.")
        return
    
    # Step 4: Process all crowns
    results = process_all_crowns(tile_inventory, crown_gdf, site, year, output_base_dir)
    
    print(f"\nPIPELINE COMPLETED!")
    print(f"Check output directory: {output_base_dir}")
    
    return results

# Configuration - can be modified for different sites/years or made into command line args
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Process NEON tiles and crown data')
    parser.add_argument('--site', default='BART', help='NEON site code (e.g., BART, SERC)')
    parser.add_argument('--year', default='2019', help='Year to process')
    parser.add_argument('--tiles-dir', 
                       default='/blue/azare/riteshchowdhry/Macrosystems/Data_files/hand_annotated_neon/neon_tiles_test',
                       help='Base directory containing tile subdirectories')
    parser.add_argument('--crown-csv',
                       default='/blue/azare/riteshchowdhry/Macrosystems/Data_files/hand_annotated_neon/clean_annotations/neon_sites_coordinates_CORRECTED_CLEAN.csv',
                       help='Path to cleaned crown coordinate CSV')
    parser.add_argument('--output-dir',
                       default='/blue/azare/riteshchowdhry/Macrosystems/Data_files/hand_annotated_neon/processed_crowns', 
                       help='Base output directory')
    
    args = parser.parse_args()
    
    print("🚀 RUNNING FULL TILE PROCESSING PIPELINE")
    print(f"Site: {args.site}, Year: {args.year}")
    print(f"Tiles: {args.tiles_dir}")
    print(f"Crowns: {args.crown_csv}")
    print(f"Output: {args.output_dir}")
    
    results = main_processing_pipeline(
        tiles_base_dir=args.tiles_dir,
        crown_csv_path=args.crown_csv, 
        site=args.site,
        year=args.year,
        output_base_dir=args.output_dir
    )
