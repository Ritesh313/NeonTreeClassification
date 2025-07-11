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
from rasterio.windows import from_bounds
from shapely.geometry import box
import json
from pathlib import Path

def extract_neon_coords_from_filename(filename, modality):
    """
    Extract coordinates from NEON filenames based on modality
    
    Args:
        filename: Full path to the file
        modality: 'RGB', 'HSI', or 'LiDAR'
    
    Returns:
        tuple: (easting, northing) or (None, None) if not found
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

def organize_downloaded_tiles(base_dir, site, year):
    """
    Organize and inventory downloaded NEON tiles
    
    Args:
        base_dir: Base directory containing RGB/, HSI/, LiDAR/ folders
        site: NEON site code (e.g., 'BART')
        year: Year string (e.g., '2019')
    
    Returns:
        dict: Organized tile inventory with coordinates as keys
    """
    print(f"ORGANIZING TILES FOR {site} {year}")
    print("="*50)
    
    # Define paths
    rgb_dir = os.path.join(base_dir, 'RGB', f'{site}_{year}')
    hsi_dir = os.path.join(base_dir, 'HSI', f'{site}_{year}')
    lidar_dir = os.path.join(base_dir, 'LiDAR', f'{site}_{year}')
    
    # Find all files
    rgb_files = []
    hsi_files = []
    lidar_files = []
    
    # Search for files in subdirectories (NEON downloads create nested structures)
    for root, dirs, files in os.walk(rgb_dir):
        rgb_files.extend([os.path.join(root, f) for f in files if f.endswith('.tif')])
    
    for root, dirs, files in os.walk(hsi_dir):
        hsi_files.extend([os.path.join(root, f) for f in files if f.endswith('.h5')])
    
    for root, dirs, files in os.walk(lidar_dir):
        lidar_files.extend([os.path.join(root, f) for f in files if f.endswith('.tif')])
    
    print(f"Found files:")
    print(f"  RGB: {len(rgb_files)} files")
    print(f"  HSI: {len(hsi_files)} files")
    print(f"  LiDAR: {len(lidar_files)} files")
    
    # Create coordinate-based inventory
    tile_inventory = {}
    
    # Process RGB files
    for rgb_file in rgb_files:
        x, y = extract_neon_coords_from_filename(rgb_file, 'RGB')
        if x and y:
            if (x, y) not in tile_inventory:
                tile_inventory[(x, y)] = {}
            tile_inventory[(x, y)]['rgb'] = rgb_file
    
    # Process HSI files
    for hsi_file in hsi_files:
        x, y = extract_neon_coords_from_filename(hsi_file, 'HSI')
        if x and y:
            if (x, y) not in tile_inventory:
                tile_inventory[(x, y)] = {}
            tile_inventory[(x, y)]['hsi'] = hsi_file
    
    # Process LiDAR files
    for lidar_file in lidar_files:
        x, y = extract_neon_coords_from_filename(lidar_file, 'LiDAR')
        if x and y:
            if (x, y) not in tile_inventory:
                tile_inventory[(x, y)] = {}
            tile_inventory[(x, y)]['lidar'] = lidar_file
    
    print(f"\nTILE INVENTORY:")
    print(f"Total unique coordinate pairs: {len(tile_inventory)}")
    
    # Count modality availability
    rgb_count = sum(1 for tile in tile_inventory.values() if 'rgb' in tile)
    hsi_count = sum(1 for tile in tile_inventory.values() if 'hsi' in tile)
    lidar_count = sum(1 for tile in tile_inventory.values() if 'lidar' in tile)
    all_three = sum(1 for tile in tile_inventory.values() if all(mod in tile for mod in ['rgb', 'hsi', 'lidar']))
    
    print(f"  RGB tiles: {rgb_count}")
    print(f"  HSI tiles: {hsi_count}")
    print(f"  LiDAR tiles: {lidar_count}")
    print(f"  All 3 modalities: {all_three}")
    
    return tile_inventory

def find_tile_for_crown(crown_coords, tile_inventory):
    """
    Find which tile contains a specific crown
    
    Args:
        crown_coords: (easting, northing) of crown center
        tile_inventory: Dictionary from organize_downloaded_tiles
    
    Returns:
        tuple: (tile_easting, tile_northing) or (None, None) if not found
    """
    crown_x, crown_y = crown_coords
    
    # NEON tiles are 1km x 1km, starting at multiples of 1000
    tile_x = int(crown_x // 1000) * 1000
    tile_y = int(crown_y // 1000) * 1000
    
    if (tile_x, tile_y) in tile_inventory:
        return tile_x, tile_y
    
    return None, None

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
    
    # Create crown polygons from coordinates
    # For now, create square bounding boxes around crown centers
    # You can modify this to use actual polygon shapes from shapefiles later
    
    crown_size = 20  # 20 meter crown bounding box (adjust as needed)
    
    geometries = []
    for idx, row in site_data.iterrows():
        center_x = row['center_easting']
        center_y = row['center_northing']
        
        # Create bounding box
        minx = center_x - crown_size/2
        maxx = center_x + crown_size/2
        miny = center_y - crown_size/2
        maxy = center_y + crown_size/2
        
        geometries.append(box(minx, miny, maxx, maxy))
    
    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame(site_data, geometry=geometries)
    
    # Set appropriate UTM CRS - handle the utm_zone value safely
    try:
        utm_zone = site_data.iloc[0]['utm_zone']
        if pd.isna(utm_zone) or utm_zone == 'error':
            # Default to BART zone if utm_zone is problematic
            gdf.crs = "EPSG:32619"  
        else:
            gdf.crs = f"EPSG:326{utm_zone}"
    except:
        # Fallback to BART zone
        gdf.crs = "EPSG:32619"
    
    print(f"Created {len(gdf)} crown polygons with {crown_size}m bounding boxes")
    
    return gdf

def crop_crown_from_tile(tile_path, crown_polygon, crown_id, output_dir, modality, site, year):
    """
    Crop a single crown from a tile using bounding box
    
    Args:
        tile_path: Path to the tile file
        crown_polygon: Shapely polygon of the crown
        crown_id: Unique identifier for the crown
        output_dir: Directory to save cropped files
        modality: 'RGB', 'HSI', or 'LiDAR'
        site: NEON site code
        year: Year string
    
    Returns:
        str: Path to cropped file or None if failed
    """
    if not os.path.exists(tile_path):
        return None
    
    try:
        with rasterio.open(tile_path) as src:
            # Get bounds of the crown polygon
            bounds = crown_polygon.bounds  # (minx, miny, maxx, maxy)
            
            # Create window from bounds
            window = from_bounds(*bounds, src.transform)
            
            # Read the data for the window
            cropped_data = src.read(window=window)
            
            # Calculate transform for the cropped area
            cropped_transform = rasterio.windows.transform(window, src.transform)
            
            # Create output filename
            crown_coords = f"{int(crown_polygon.centroid.x)}_{int(crown_polygon.centroid.y)}"
            output_filename = f"{site}_{year}_{crown_coords}_{crown_id}_{modality}.tif"
            output_path = os.path.join(output_dir, output_filename)
            
            # Write cropped data
            with rasterio.open(
                output_path,
                'w',
                driver='GTiff',
                height=cropped_data.shape[1],
                width=cropped_data.shape[2],
                count=cropped_data.shape[0],
                dtype=cropped_data.dtype,
                crs=src.crs,
                transform=cropped_transform,
                nodata=src.nodata
            ) as dst:
                dst.write(cropped_data)
            
            return output_path
            
    except Exception as e:
        print(f"Error cropping {modality} for crown {crown_id}: {e}")
        return None

def process_all_crowns(tile_inventory, crown_gdf, site, year, output_base_dir):
    """
    Process all crowns and create individual crop files
    
    Args:
        tile_inventory: Dictionary from organize_downloaded_tiles
        crown_gdf: GeoDataFrame with crown polygons
        site: NEON site code
        year: Year string
        output_base_dir: Base directory for output
    
    Returns:
        dict: Summary of processing results
    """
    print(f"CROPPING ALL CROWNS FOR {site} {year}")
    print("="*50)
    
    # Create output directories
    rgb_output_dir = os.path.join(output_base_dir, 'crops', 'RGB')
    hsi_output_dir = os.path.join(output_base_dir, 'crops', 'HSI')
    lidar_output_dir = os.path.join(output_base_dir, 'crops', 'LiDAR')
    
    for output_dir in [rgb_output_dir, hsi_output_dir, lidar_output_dir]:
        os.makedirs(output_dir, exist_ok=True)
    
    results = {
        'total_crowns': len(crown_gdf),
        'successful_rgb': 0,
        'successful_hsi': 0,
        'successful_lidar': 0,
        'failed_crowns': [],
        'crop_files': []
    }
    
    for idx, crown_row in crown_gdf.iterrows():
        crown_id = f"crown_{idx:03d}"
        crown_polygon = crown_row.geometry
        crown_coords = (crown_polygon.centroid.x, crown_polygon.centroid.y)
        
        print(f"Processing {crown_id} at coordinates {crown_coords}")
        
        # Find tile containing this crown
        tile_coords = find_tile_for_crown(crown_coords, tile_inventory)
        
        if tile_coords is None:
            print(f"  No tile found for {crown_id}")
            results['failed_crowns'].append(crown_id)
            continue
        
        tile_data = tile_inventory[tile_coords]
        crown_files = {}
        
        # Crop RGB
        if 'rgb' in tile_data:
            rgb_crop = crop_crown_from_tile(
                tile_path=tile_data['rgb'],
                crown_polygon=crown_polygon,
                crown_id=crown_id,
                output_dir=rgb_output_dir,
                modality='RGB',
                site=site,
                year=year
            )
            if rgb_crop:
                crown_files['rgb'] = rgb_crop
                results['successful_rgb'] += 1
                print(f"  ✓ RGB cropped")
        
        # Crop HSI (use converted TIF if available)
        hsi_tile = tile_data.get('hsi_tif', tile_data.get('hsi'))
        if hsi_tile and hsi_tile.endswith('.tif'):
            hsi_crop = crop_crown_from_tile(
                tile_path=hsi_tile,
                crown_polygon=crown_polygon,
                crown_id=crown_id,
                output_dir=hsi_output_dir,
                modality='HSI',
                site=site,
                year=year
            )
            if hsi_crop:
                crown_files['hsi'] = hsi_crop
                results['successful_hsi'] += 1
                print(f"  ✓ HSI cropped")
        
        # Crop LiDAR
        if 'lidar' in tile_data:
            lidar_crop = crop_crown_from_tile(
                tile_path=tile_data['lidar'],
                crown_polygon=crown_polygon,
                crown_id=crown_id,
                output_dir=lidar_output_dir,
                modality='LiDAR',
                site=site,
                year=year
            )
            if lidar_crop:
                crown_files['lidar'] = lidar_crop
                results['successful_lidar'] += 1
                print(f"  ✓ LiDAR cropped")
        
        if crown_files:
            # Save metadata
            metadata = {
                'crown_id': crown_id,
                'site': site,
                'year': year,
                'center_coords': [crown_polygon.centroid.x, crown_polygon.centroid.y],
                'bounds': list(crown_polygon.bounds),
                'tile_coords': tile_coords,
                'source_file': crown_row['filename'],
                'modalities': list(crown_files.keys()),
                'crop_files': crown_files
            }
            results['crop_files'].append(metadata)
    
    # Print summary
    print(f"\nCROPPING SUMMARY:")
    print(f"Total crowns processed: {results['total_crowns']}")
    print(f"Successful RGB crops: {results['successful_rgb']}")
    print(f"Successful HSI crops: {results['successful_hsi']}")
    print(f"Successful LiDAR crops: {results['successful_lidar']}")
    print(f"Failed crowns: {len(results['failed_crowns'])}")
    
    # Save results to JSON
    results_file = os.path.join(output_base_dir, f'{site}_{year}_cropping_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {results_file}")
    
    return results

def main_processing_pipeline(tiles_base_dir, crown_csv_path, site, year, output_base_dir):
    """
    Main pipeline to process tiles to crown crops
    
    Args:
        tiles_base_dir: Base directory with downloaded tiles
        crown_csv_path: Path to cleaned crown CSV
        site: NEON site code
        year: Year string
        output_base_dir: Base directory for all outputs
    """
    print(f"STARTING TILE-TO-CROWN PROCESSING PIPELINE")
    print(f"Site: {site}, Year: {year}")
    print("="*60)
    
    # Step 1: Organize downloaded tiles
    tile_inventory = organize_downloaded_tiles(tiles_base_dir, site, year)
    
    if not tile_inventory:
        print("No tiles found. Exiting.")
        return
    
    # Step 2: Convert HSI tiles to TIF
    tile_inventory = convert_all_hsi_tiles(tile_inventory, site, year, output_base_dir)
    
    # Step 3: Load crown data
    crown_gdf = load_crown_data(crown_csv_path, site, int(year))
    
    if crown_gdf is None or len(crown_gdf) == 0:
        print("No crown data found. Exiting.")
        return
    
    # Step 4: Process all crowns
    results = process_all_crowns(tile_inventory, crown_gdf, site, year, output_base_dir)
    
    print(f"\nPIPELINE COMPLETED!")
    print(f"Check output directory: {output_base_dir}")
    
    return results

# Test the pipeline with the downloaded BART data
if __name__ == "__main__":
    # Configuration for BART 2019 test case
    tiles_base_dir = "/blue/azare/riteshchowdhry/Macrosystems/Data_files/hand_annotated_neon/neon_tiles_test"
    crown_csv_path = "/blue/azare/riteshchowdhry/Macrosystems/Data_files/hand_annotated_neon/clean_annotations/neon_sites_coordinates_CORRECTED_CLEAN.csv"
    output_base_dir = "/blue/azare/riteshchowdhry/Macrosystems/Data_files/hand_annotated_neon/processed_crowns"
    
    site = "BART"
    year = "2019"
    
    print("🚀 RUNNING FULL TILE PROCESSING PIPELINE")
    print("="*60)
    
    # Run the full pipeline
    results = main_processing_pipeline(
        tiles_base_dir=tiles_base_dir,
        crown_csv_path=crown_csv_path,
        site=site,
        year=year,
        output_base_dir=output_base_dir
    )