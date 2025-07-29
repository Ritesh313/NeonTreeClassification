import os
import glob
import re
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.mask import mask
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import box

# Function definition
def create_tile_inventory(flat_dir, hsi_converted_dir=None):
    """
    Create an inventory of all available tiles across the three modalities.
    
    Args:
        flat_dir: Base directory containing RGB, LiDAR, and HSI subdirectories
        hsi_converted_dir: Directory with converted HSI TIF files (defaults to flat_dir/HSI_converted)
    
    Returns:
        dict: Inventory of tiles by coordinates and modality
    """
    if hsi_converted_dir is None:
        hsi_converted_dir = os.path.join(flat_dir, 'HSI_converted')
        
    inventory = {}
    
    # Define subdirectory paths
    rgb_dir = os.path.join(flat_dir, 'RGB')
    lidar_dir = os.path.join(flat_dir, 'LiDAR')
    
    # Find RGB tiles
    rgb_files = glob.glob(os.path.join(rgb_dir, "*_image.tif"))
    print(f"Found {len(rgb_files)} RGB files")
    
    # Find LiDAR tiles
    lidar_files = glob.glob(os.path.join(lidar_dir, "*_CHM.tif"))
    print(f"Found {len(lidar_files)} LiDAR files")
    
    # Find HSI tiles (already converted to TIF)
    hsi_files = glob.glob(os.path.join(hsi_converted_dir, "*_reflectance.tif"))
    print(f"Found {len(hsi_files)} HSI (TIF) files")
    
    # Function to extract coordinates from filenames
    def extract_coordinates(filename):
        basename = os.path.basename(filename)
        
        # RGB pattern: YYYY_SITE_X_XXXXXX_YYYYYY_image.tif
        match = re.search(r'_(\d+)_(\d+)_image\.tif$', basename)
        if match:
            return int(match.group(1)), int(match.group(2))
        
        # LiDAR pattern: NEON_D01_SITE_DP3_XXXXXX_YYYYYY_YYYY_CHM.tif
        match = re.search(r'_(\d+)_(\d+)_\d+_CHM\.tif$', basename)
        if match:
            return int(match.group(1)), int(match.group(2))
        
        # HSI pattern: NEON_D01_SITE_DP3_XXXXXX_YYYYYY_YYYY_reflectance.tif
        match = re.search(r'_(\d+)_(\d+)_\d+_reflectance\.tif$', basename)
        if match:
            return int(match.group(1)), int(match.group(2))
        
        return None
    
    # Process RGB files
    for file_path in rgb_files:
        coords = extract_coordinates(file_path)
        if coords:
            x, y = coords
            if (x, y) not in inventory:
                inventory[(x, y)] = {}
            inventory[(x, y)]['rgb'] = file_path
    
    # Process LiDAR files
    for file_path in lidar_files:
        coords = extract_coordinates(file_path)
        if coords:
            x, y = coords
            if (x, y) not in inventory:
                inventory[(x, y)] = {}
            inventory[(x, y)]['lidar'] = file_path
    
    # Process HSI files
    for file_path in hsi_files:
        coords = extract_coordinates(file_path)
        if coords:
            x, y = coords
            if (x, y) not in inventory:
                inventory[(x, y)] = {}
            inventory[(x, y)]['hsi'] = file_path
    
    # Print summary
    print(f"\nTile Inventory Summary:")
    print(f"Total unique coordinate pairs: {len(inventory)}")
    
    # Count by modality
    rgb_count = sum(1 for data in inventory.values() if 'rgb' in data)
    hsi_count = sum(1 for data in inventory.values() if 'hsi' in data)
    lidar_count = sum(1 for data in inventory.values() if 'lidar' in data)
    
    print(f"  RGB tiles: {rgb_count}")
    print(f"  HSI tiles: {hsi_count}")
    print(f"  LiDAR tiles: {lidar_count}")
    
    # Count tiles with all 3 modalities
    complete_tiles = sum(1 for data in inventory.values() 
                       if all(mod in data for mod in ['rgb', 'hsi', 'lidar']))
    print(f"  Complete tiles (all 3 modalities): {complete_tiles}")
    
    # Optional: print a sample of file patterns to verify coordinate extraction
    if rgb_files:
        print(f"\nSample RGB filename: {os.path.basename(rgb_files[0])}")
    if lidar_files:
        print(f"Sample LiDAR filename: {os.path.basename(lidar_files[0])}")
    if hsi_files:
        print(f"Sample HSI filename: {os.path.basename(hsi_files[0])}")
    
    return inventory

def load_crown_data(crown_csv_path, shapefile_dir, site=None, year=None): 
    """ 
    Load crown data from CSV and corresponding shapefiles
    Args:
        crown_csv_path: Path to CSV with crown metadata
        shapefile_dir: Directory containing crown shapefiles
        site: Optional filter for specific NEON site
        year: Optional filter for specific year
        
    Returns:
        GeoDataFrame: Crown polygons with metadata
    """
    print(f"Loading crown data from {crown_csv_path}")

    # Load the CSV for metadata
    df = pd.read_csv(crown_csv_path)
    print(f"CSV contains {len(df)} crown records")

    # Filter for specific site and year if provided
    if site is not None:
        df = df[df['site'] == site]
        print(f"Filtered to {len(df)} crowns for site {site}")

    if year is not None:
        year_str = str(year)
        df = df[df['year'].astype(str) == year_str]
        print(f"Filtered to {len(df)} crowns for year {year}")

    if len(df) == 0:
        print("No crown records found with the specified filters.")
        return None

    # Create a progress counter
    total_crowns = len(df)
    processed = 0
    loaded = 0
    failed = 0

    # List to store crown records with geometry
    crown_records = []

    # Process each crown record
    for idx, row in df.iterrows():
        processed += 1
        
        # Get shapefile name and construct path
        shapefile_name = row['filename']
        shapefile_path = os.path.join(shapefile_dir, shapefile_name)
        
        # Try to load the shapefile
        try:
            # Check if file exists
            if not os.path.exists(shapefile_path):
                # Try without .shp extension if it was included
                if shapefile_path.endswith('.shp') and not os.path.exists(shapefile_path):
                    shapefile_path = shapefile_path[:-4]
                # If still doesn't exist, fall back to bounding box
                if not os.path.exists(shapefile_path + '.shp'):
                    raise FileNotFoundError(f"Shapefile not found: {shapefile_path}")
            
            # Load shapefile
            crown_shp = gpd.read_file(shapefile_path)
            
            # Get CRS from CSV
            target_crs = row['target_crs']
            
            # Ensure shapefile has the correct CRS
            if crown_shp.crs is None:
                crown_shp.crs = target_crs
            elif str(crown_shp.crs) != target_crs:
                crown_shp = crown_shp.to_crs(target_crs)
            
            # Add metadata from CSV to each polygon
            for crown_idx, crown_geom in enumerate(crown_shp.geometry):
                crown_record = row.to_dict()
                crown_record['geometry'] = crown_geom
                crown_record['crown_idx'] = crown_idx
                crown_records.append(crown_record)
                
            loaded += 1
            
            # Show progress
            if processed % 50 == 0 or processed == total_crowns:
                print(f"Processed {processed}/{total_crowns} crowns. Loaded: {loaded}, Failed: {failed}")
                
        except Exception as e:
            # Fall back to bounding box from CSV
            print(f"Error loading shapefile {shapefile_path}: {e}")
            print("Falling back to bounding box from CSV")
            
            # Create bounding box from coordinates
            bbox = box(
                row['min_easting'], 
                row['min_northing'], 
                row['max_easting'], 
                row['max_northing']
            )
            
            # Add record with bounding box
            crown_record = row.to_dict()
            crown_record['geometry'] = bbox
            crown_record['crown_idx'] = 0  # Just one box
            crown_records.append(crown_record)
            
            failed += 1

    # Create GeoDataFrame from crown records
    if crown_records:
        crown_gdf = gpd.GeoDataFrame(crown_records)
        print(f"Created GeoDataFrame with {len(crown_gdf)} crown polygons")
        
        # Set CRS based on first record's target_crs
        # This assumes all crowns have the same target CRS
        crown_gdf.crs = crown_gdf.iloc[0]['target_crs']
        
        return crown_gdf
    else:
        print("No crown polygons could be loaded.")
        return None

def create_tile_to_crowns_mapping(crown_gdf, tile_inventory): 
    """ 
    Create an efficient mapping from tiles to the crowns they contain using spatial indexing.
        Args:
        crown_gdf: GeoDataFrame with crown polygons
        tile_inventory: Dictionary of tiles by coordinate
        
    Returns:
        dict: Mapping of tile coordinates to crown indices
    """
    # Initialize the output dictionary
    tile_to_crowns = {}

    # Create spatial index for crowns (this is an R-tree)
    crown_spatial_index = crown_gdf.sindex

    # Process each tile
    total_tiles = len(tile_inventory)
    processed = 0

    for (tile_x, tile_y), tile_data in tile_inventory.items():
        # Create tile bounding box
        tile_box = box(tile_x, tile_y, tile_x + 1000, tile_y + 1000)
        
        # Query spatial index for crowns that may intersect this tile
        # This is an O(log n) operation
        possible_matches_idx = list(crown_spatial_index.intersection(tile_box.bounds))
        
        # Confirm actual intersection (spatial index returns candidates that need verification)
        if possible_matches_idx:
            possible_matches = crown_gdf.iloc[possible_matches_idx]
            actual_matches = possible_matches[possible_matches.intersects(tile_box)]
            
            if len(actual_matches) > 0:
                # Store the crown indices for this tile
                tile_to_crowns[(tile_x, tile_y)] = list(actual_matches.index)
        
        # Progress tracking
        processed += 1
        if processed % 10 == 0 or processed == total_tiles:
            print(f"Processed {processed}/{total_tiles} tiles")

    # Print summary
    tiles_with_crowns = len(tile_to_crowns)
    total_crown_mappings = sum(len(crowns) for crowns in tile_to_crowns.values())

    print(f"\nTile-to-Crown Mapping Summary:")
    print(f"Tiles with at least one crown: {tiles_with_crowns}/{total_tiles}")
    print(f"Total crown mappings: {total_crown_mappings}")

    # Count crowns per tile
    crown_counts = {}
    for crowns in tile_to_crowns.values():
        num_crowns = len(crowns)
        crown_counts[num_crowns] = crown_counts.get(num_crowns, 0) + 1

    print("\nTiles by number of crowns:")
    for num_crowns, count in sorted(crown_counts.items()):
        print(f"  {num_crowns} crown(s): {count} tiles")

    return tile_to_crowns

def crop_crowns_from_tile(tile_coords, crown_indices, crown_gdf, tile_inventory, output_dir, metadata_records, buffer_meters=2):
    """
    Crop all crowns from a specific tile for all available modalities
    
    Args:
        tile_coords: (x, y) coordinates of the tile
        crown_indices: List of indices in crown_gdf for crowns in this tile
        crown_gdf: GeoDataFrame with crown polygons
        tile_inventory: Dictionary of tiles by coordinate pair
        output_dir: Base directory for outputs
        metadata_records: List to collect crown metadata records
        buffer_meters: Optional buffer around crown in meters
        
    Returns:
        dict: Results of cropping operation
    """
    tile_x, tile_y = tile_coords

    if tile_coords not in tile_inventory:
        print(f"❌ Error: Tile {tile_coords} not found in inventory")
        return {'success': False, 'reason': 'tile_not_found'}

    tile_data = tile_inventory[tile_coords]

    # Track results
    results = {
        'tile_coords': tile_coords,
        'crowns_processed': 0,
        'modalities_processed': {},
        'success': True
    }

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Process each modality (in order of increasing memory usage)
    for modality, file_key in [('RGB', 'rgb'), ('LiDAR', 'lidar'), ('HSI', 'hsi')]:
        if file_key not in tile_data:
            print(f"⚠️ Warning: {modality} data not available for tile {tile_coords}")
            continue
        
        # Modality-specific results
        mod_results = {
            'crowns_processed': 0,
            'success': True
        }
        
        try:
            # Open the tile file
            print(f"Processing {modality} tile at {tile_coords} with {len(crown_indices)} crowns...")
            
            with rasterio.open(tile_data[file_key]) as src:
                # Get resolution for appropriate buffering
                res_x = src.transform[0]  # pixel width
                res_y = -src.transform[4]  # pixel height (negative as y increases downward)
                
                print(f"  {modality} resolution: {res_x}x{res_y} m/pixel")
                
                # Buffer in pixels (based on meters and resolution)
                buffer_pixels = max(buffer_meters / res_x, buffer_meters / res_y)
                
                # Process each crown in this tile
                for i, crown_idx in enumerate(crown_indices):
                    crown = crown_gdf.iloc[crown_idx]
                    site = crown['site']
                    plot = crown['plot']
                    year = crown['year']
                    
                    # Create a unique crown ID
                    crown_id = f"{site}_{plot}_{year}_{crown_idx}"
                    
                    # Create output path in the flat directory structure
                    output_path = os.path.join(output_dir, f"{crown_id}_{modality}.tif")
                    
                    # Add buffer to crown geometry
                    buffered_geom = crown.geometry.buffer(buffer_pixels * max(res_x, res_y))
                    
                    try:
                        # Crop raster to crown geometry
                        out_image, out_transform = mask(src, [buffered_geom], crop=True)
                        
                        # Skip if output is empty
                        if out_image.size == 0 or out_image.shape[1] <= 0 or out_image.shape[2] <= 0:
                            print(f"  ⚠️ Empty crop for {crown_id} {modality}")
                            continue
                        
                        # Update metadata
                        out_meta = src.meta.copy()
                        out_meta.update({
                            "driver": "GTiff",
                            "height": out_image.shape[1],
                            "width": out_image.shape[2],
                            "transform": out_transform
                        })
                        
                        # Save cropped image
                        with rasterio.open(output_path, "w", **out_meta) as dest:
                            dest.write(out_image)
                        
                        # If this is the first modality processed for this crown, add metadata record
                        if modality == 'RGB':
                            # Get crown centroid coordinates
                            centroid = crown.geometry.centroid
                            easting, northing = centroid.x, centroid.y
                            
                            # Create metadata record
                            metadata_record = {
                                'crown_id': crown_id,
                                'site': site,
                                'plot': plot,
                                'year': year,
                                'tile_x': tile_x,
                                'tile_y': tile_y,
                                'easting': easting,
                                'northing': northing,
                                'rgb_path': os.path.basename(output_path),
                                'lidar_path': '',  # Will be filled if processed
                                'hsi_path': ''     # Will be filled if processed
                            }
                            metadata_records.append(metadata_record)
                        else:
                            # Update existing record with path to this modality
                            for record in metadata_records:
                                if record['crown_id'] == crown_id:
                                    record[f'{modality.lower()}_path'] = os.path.basename(output_path)
                                    break
                        
                        # Update results
                        mod_results['crowns_processed'] += 1
                        
                        # Progress update
                        if (i + 1) % 10 == 0 or (i + 1) == len(crown_indices):
                            print(f"  Processed {i + 1}/{len(crown_indices)} crowns for {modality}")
                    
                    except Exception as e:
                        print(f"  ❌ Error processing crown {crown_id} for {modality}: {e}")
        
        except Exception as e:
            print(f"❌ Error processing {modality} tile at {tile_coords}: {e}")
            mod_results['success'] = False
        
        # Add modality results to overall results
        results['modalities_processed'][modality] = mod_results
        results['crowns_processed'] += mod_results['crowns_processed']

    return results

def process_all_crown_tiles(tile_to_crowns, crown_gdf, tile_inventory, output_dir, csv_path=None):
    """
    Process all tiles that contain crowns and create metadata CSV
    
    Args:
        tile_to_crowns: Mapping of tile coordinates to crown indices
        crown_gdf: GeoDataFrame with crown data
        tile_inventory: Inventory of tile files
        output_dir: Base directory for outputs
        csv_path: Path to save the metadata CSV (default: output_dir/crown_metadata.csv)
        
    Returns:
        dict: Processing results
    """
    # Set default CSV path if not provided
    if csv_path is None:
        csv_path = os.path.join(output_dir, 'crown_metadata.csv')
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Count total crowns to process
    total_crowns = sum(len(crowns) for crowns in tile_to_crowns.values())
    print(f"Processing {len(tile_to_crowns)} tiles containing {total_crowns} crowns")
    
    # Results tracking
    results = {
        'tiles_processed': 0,
        'tiles_succeeded': 0,
        'crowns_processed': 0,
        'crowns_succeeded': {
            'RGB': 0,
            'LiDAR': 0,
            'HSI': 0
        }
    }
    
    # List to collect metadata for all crowns
    metadata_records = []
    
    # Sort tiles by number of crowns (largest first)
    # This helps tackle the most complex tiles first
    sorted_tiles = sorted(tile_to_crowns.items(), key=lambda x: len(x[1]), reverse=True)
    
    # Process each tile
    for i, (tile_coords, crown_indices) in enumerate(sorted_tiles):
        print(f"\n[{i+1}/{len(sorted_tiles)}] Processing tile {tile_coords} with {len(crown_indices)} crowns")
        
        try:
            # Crop all crowns in this tile
            tile_results = crop_crowns_from_tile(tile_coords, crown_indices, crown_gdf, 
                                              tile_inventory, output_dir, metadata_records)
            
            # Update results
            results['tiles_processed'] += 1
            if tile_results['success']:
                results['tiles_succeeded'] += 1
            
            results['crowns_processed'] += len(crown_indices)
            for modality, mod_results in tile_results['modalities_processed'].items():
                results['crowns_succeeded'][modality] += mod_results['crowns_processed']
                
            print(f"Tile {i+1}/{len(sorted_tiles)} completed")
            print(f"Progress: {results['crowns_processed']}/{total_crowns} crowns processed")
            
        except Exception as e:
            print(f"Error processing tile {tile_coords}: {e}")
    
    # Save metadata to CSV
    print(f"\nSaving metadata for {len(metadata_records)} crowns to {csv_path}")
    metadata_df = pd.DataFrame(metadata_records)
    metadata_df.to_csv(csv_path, index=False)
    
    # Print final summary
    print("\nProcessing Complete!")
    print(f"Tiles: {results['tiles_succeeded']}/{results['tiles_processed']} completed successfully")
    print(f"Crowns processed: {results['crowns_processed']}")
    print("Crowns succeeded by modality:")
    for modality, count in results['crowns_succeeded'].items():
        print(f"  {modality}: {count}/{results['crowns_processed']} ({count/results['crowns_processed']*100:.1f}%)")
    
    return results

if __name__ == "__main__":
    # Configuration for paths
    flat_dir = '/blue/azare/riteshchowdhry/Macrosystems/Data_files/hand_annotated_neon/neon_tiles_FULL/flat_dir'
    hsi_converted_dir = '/blue/azare/riteshchowdhry/Macrosystems/Data_files/hand_annotated_neon/neon_tiles_FULL/flat_dir/HSI_converted'

    # Create tile inventory
    tile_inventory = create_tile_inventory(flat_dir, hsi_converted_dir)

    # Check a few specific coordinates to verify inventory
    if tile_inventory:
        # Print details for the first few tiles
        print("\nSample tiles:")
        for i, ((x, y), data) in enumerate(list(tile_inventory.items())[:3]):
            print(f"Tile {i+1}: ({x}, {y})")
            for modality, path in data.items():
                print(f"  {modality}: {os.path.basename(path)}")
    
    site = 'BART' # Example: Bartlett Experimental Forest 
    year = 2019 # Example: 2019 data
    shapefile_dir = '/blue/azare/riteshchowdhry/Macrosystems/Data_files/hand_annotated_neon/clean_annotations/consolidated_dir' 
    crown_csv = '/blue/azare/riteshchowdhry/Macrosystems/Data_files/hand_annotated_neon/clean_annotations/neon_sites_coordinates_CORRECTED_CLEAN.csv'

    crown_gdf = load_crown_data(crown_csv, shapefile_dir, site=site, year=year)
    
    tile_to_crowns = create_tile_to_crowns_mapping(crown_gdf, tile_inventory)
    output_dir = '/blue/azare/riteshchowdhry/Macrosystems/Data_files/hand_annotated_neon/neon_tiles_FULL/processed_crowns'
    metadata_csv = os.path.join(output_dir, 'crown_metadata.csv')
    results = process_all_crown_tiles(tile_to_crowns, crown_gdf, tile_inventory, output_dir, metadata_csv)
        