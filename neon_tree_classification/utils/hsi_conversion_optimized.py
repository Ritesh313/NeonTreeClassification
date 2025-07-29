#!/usr/bin/env python3
"""
High-Performance HSI Conversion Utilities

Optimized HSI conversion with parallel processing, memory management,
and efficient I/O operations.

Author: Ritesh Chowdhry
"""

import os
import glob
import re
import h5py
import numpy as np
import rasterio
from rasterio.transform import Affine
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from pathlib import Path
import time
import psutil

def get_optimal_chunk_size():
    """Calculate optimal chunk size based on available memory"""
    available_memory = psutil.virtual_memory().available
    # Use 25% of available memory for chunk processing
    target_memory = available_memory * 0.25
    # Estimate memory per chunk (assuming float32, ~400 bands after filtering)
    bytes_per_pixel = 4 * 400  # 4 bytes * 400 bands
    pixels_per_chunk = int(target_memory / bytes_per_pixel)
    # Convert to rows (assuming 1000x1000 tiles)
    chunk_rows = max(50, min(500, int(np.sqrt(pixels_per_chunk))))
    return chunk_rows

def get_filtered_band_indices():
    """Get band indices with water absorption bands removed"""
    band_indices = np.r_[0:425]
    # Remove water absorption bands more efficiently
    bad_bands = np.concatenate([
        np.r_[419:425],  # ~2500nm
        np.r_[283:315],  # ~1900nm  
        np.r_[192:210]   # ~1400nm
    ])
    return np.setdiff1d(band_indices, bad_bands)

def convert_hsi_optimized(h5_file, output_dir, site, year, coords, compress='lzw'):
    """
    Optimized HSI conversion with memory efficiency and compression
    
    Args:
        h5_file: Path to HSI H5 file
        output_dir: Output directory
        site: NEON site code
        year: Year string  
        coords: Coordinate string
        compress: Compression method ('lzw', 'deflate', 'none')
    
    Returns:
        str: Output file path or None if failed
    """
    
    output_file = f"{year}_{site}_{coords}_HSI.tif"
    output_path = os.path.join(output_dir, output_file)
    
    # Skip if already exists
    if os.path.exists(output_path):
        return output_path
    
    # Get optimal parameters
    chunk_size = get_optimal_chunk_size()
    band_indices = get_filtered_band_indices()
    
    try:
        start_time = time.time()
        
        with h5py.File(h5_file, 'r') as hdf:
            # Get sitename and reflectance data
            sitename = list(hdf.keys())[0]
            refl_group = hdf[sitename]['Reflectance']
            refl_dataset = refl_group['Reflectance_Data']
            
            # Extract metadata efficiently
            metadata = refl_group['Metadata']['Coordinate_System']
            epsg = str(metadata['EPSG Code'][()]).split("'")[1]
            map_info = str(metadata['Map_Info'][()]).split(",")
            
            # Geospatial parameters
            pixel_width = float(map_info[5])
            pixel_height = float(map_info[6])
            x_min = float(map_info[3])
            y_max = float(map_info[4])
            transform = Affine.translation(x_min, y_max) * Affine.scale(pixel_width, -pixel_height)
            
            # Data parameters
            scale_factor = float(refl_dataset.attrs['Scale_Factor'])
            no_data_val = float(refl_dataset.attrs['Data_Ignore_Value'])
            height, width, _ = refl_dataset.shape
            
            # Optimized rasterio profile
            profile = {
                'driver': 'GTiff',
                'height': height,
                'width': width,
                'count': len(band_indices),
                'dtype': np.float32,
                'crs': f'EPSG:{epsg}',
                'transform': transform,
                'nodata': no_data_val,
                'compress': compress,
                'tiled': True,         # Enable tiling for better performance
                'blockxsize': 512,     # Optimal block size
                'blockysize': 512,
                'BIGTIFF': 'YES'       # Handle large files
            }
            
            # Process with chunking
            with rasterio.open(output_path, 'w', **profile) as dst:
                for start_row in range(0, height, chunk_size):
                    end_row = min(start_row + chunk_size, height)
                    
                    # Read only selected bands to save memory
                    chunk_data = refl_dataset[start_row:end_row, :, band_indices]
                    
                    # Efficient type conversion and dimension swap
                    chunk_data = chunk_data.astype(np.float32, copy=False)
                    chunk_data = np.transpose(chunk_data, (2, 0, 1))  # Faster than moveaxis
                    
                    # Write all bands at once using windows
                    window = rasterio.windows.Window(0, start_row, width, end_row - start_row)
                    dst.write(chunk_data, window=window)
        
        conversion_time = time.time() - start_time
        file_size_mb = os.path.getsize(output_path) / (1024*1024)
        
        return {
            'path': output_path,
            'time': conversion_time,
            'size_mb': file_size_mb,
            'coords': coords
        }
        
    except Exception as e:
        # Clean up partial file
        if os.path.exists(output_path):
            os.remove(output_path)
        return {'error': str(e), 'coords': coords}

def convert_hsi_parallel(hsi_files, output_dir, site, year, max_workers=None):
    """
    Convert multiple HSI files in parallel
    
    Args:
        hsi_files: List of HSI file paths
        output_dir: Output directory
        site: NEON site code
        year: Year string
        max_workers: Number of parallel workers (None = auto)
    
    Returns:
        dict: Conversion results and statistics
    """
    
    if max_workers is None:
        # Use 75% of available cores, but limit to avoid memory issues
        max_workers = max(1, min(mp.cpu_count() - 1, len(hsi_files) // 2))
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"🚀 PARALLEL HSI CONVERSION")
    print(f"Files: {len(hsi_files)}")
    print(f"Workers: {max_workers}")
    print(f"Target: {output_dir}")
    print("="*50)
    
    # Prepare tasks
    tasks = []
    for hsi_file in hsi_files:
        # Extract coordinates from filename
        basename = os.path.basename(hsi_file)
        coords_match = re.search(r'(\d+)_(\d+)', basename)
        if coords_match:
            coords = f"{coords_match.group(1)}_{coords_match.group(2)}"
            tasks.append((hsi_file, output_dir, site, year, coords))
    
    # Execute in parallel
    results = {'success': [], 'errors': [], 'stats': {}}
    start_time = time.time()
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_task = {
            executor.submit(convert_hsi_optimized, *task): task 
            for task in tasks
        }
        
        # Collect results
        for future in as_completed(future_to_task):
            task = future_to_task[future]
            try:
                result = future.result()
                if 'error' in result:
                    results['errors'].append(result)
                    print(f"❌ {result['coords']}: {result['error']}")
                else:
                    results['success'].append(result)
                    print(f"✅ {result['coords']}: {result['time']:.1f}s, {result['size_mb']:.1f}MB")
            except Exception as e:
                results['errors'].append({'coords': task[4], 'error': str(e)})
                print(f"❌ {task[4]}: {e}")
    
    # Calculate statistics
    total_time = time.time() - start_time
    success_count = len(results['success'])
    error_count = len(results['errors'])
    
    if results['success']:
        avg_time = np.mean([r['time'] for r in results['success']])
        total_size = sum([r['size_mb'] for r in results['success']])
        results['stats'] = {
            'total_time': total_time,
            'avg_time_per_file': avg_time,
            'total_size_mb': total_size,
            'success_count': success_count,
            'error_count': error_count,
            'speedup': len(hsi_files) * avg_time / total_time if total_time > 0 else 1
        }
    
    print(f"\n📊 CONVERSION SUMMARY")
    print(f"✅ Success: {success_count}")
    print(f"❌ Errors: {error_count}")
    print(f"⏱️  Total time: {total_time:.1f}s")
    if results['success']:
        print(f"🚀 Speedup: {results['stats']['speedup']:.1f}x")
        print(f"💾 Total size: {results['stats']['total_size_mb']:.1f}MB")
    
    return results

# Example usage function
def convert_flattened_hsi_tiles(flat_hsi_dir, output_dir, max_workers=None):
    """
    Convert all HSI tiles from flattened directory structure
    
    Args:
        flat_hsi_dir: Path to flattened HSI directory
        output_dir: Output directory for converted TIFFs
        max_workers: Number of parallel workers
    """
    
    # Find all HSI files
    hsi_pattern = os.path.join(flat_hsi_dir, '*_reflectance.h5')
    hsi_files = glob.glob(hsi_pattern)
    
    if not hsi_files:
        print(f"❌ No HSI files found in {flat_hsi_dir}")
        return
    
    # Group by site and year for organized output
    site_year_groups = {}
    for hsi_file in hsi_files:
        basename = os.path.basename(hsi_file)
        # Extract site and year from filename
        parts = basename.split('_')
        if len(parts) >= 6:
            domain = parts[1]  # D01, D03, etc.
            site = parts[2]    # BART, HARV, etc.
            # Year might be in filename (if flattened with year) or need default
            year = '2019'  # Default, could be extracted from directory or filename
            
            key = f"{site}_{year}"
            if key not in site_year_groups:
                site_year_groups[key] = {'site': site, 'year': year, 'files': []}
            site_year_groups[key]['files'].append(hsi_file)
    
    # Convert each group
    all_results = {}
    for group_key, group_data in site_year_groups.items():
        group_output_dir = os.path.join(output_dir, 'HSI_converted', group_key)
        print(f"\n🎯 Processing group: {group_key} ({len(group_data['files'])} files)")
        
        results = convert_hsi_parallel(
            hsi_files=group_data['files'],
            output_dir=group_output_dir,
            site=group_data['site'],
            year=group_data['year'],
            max_workers=max_workers
        )
        all_results[group_key] = results
    
    return all_results

if __name__ == "__main__":
    # Example usage
    flat_hsi_dir = "/path/to/neon_tiles_FLAT/HSI"
    output_dir = "/path/to/converted_hsi"
    
    results = convert_flattened_hsi_tiles(flat_hsi_dir, output_dir, max_workers=4)
