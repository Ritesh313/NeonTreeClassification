#!/usr/bin/env python3
"""
Safe HSI Conversion Utilities

Conservative HSI conversion focused on data integrity with modest performance improvements.
Includes validation and verification to ensure data correctness.

Author: Ritesh Chowdhry
"""

import os
import glob
import re
import h5py
import numpy as np
import rasterio
from rasterio.transform import Affine
import time
import hashlib
from pathlib import Path

def validate_hsi_file(h5_file):
    """
    Validate HSI H5 file integrity before processing
    
    Returns:
        dict: Validation results with file metadata
    """
    try:
        with h5py.File(h5_file, 'r') as hdf:
            sitename = list(hdf.keys())[0]
            refl_group = hdf[sitename]['Reflectance']
            refl_dataset = refl_group['Reflectance_Data']
            
            # Basic validation checks
            shape = refl_dataset.shape
            if len(shape) != 3:
                return {'valid': False, 'error': 'Invalid shape'}
            
            if shape[2] < 400:  # Should have ~426 bands
                return {'valid': False, 'error': 'Insufficient bands'}
            
            # Check for required metadata
            try:
                epsg = refl_group['Metadata']['Coordinate_System']['EPSG Code'][()]
                map_info = refl_group['Metadata']['Coordinate_System']['Map_Info'][()]
                scale_factor = refl_dataset.attrs['Scale_Factor']
                no_data_val = refl_dataset.attrs['Data_Ignore_Value']
            except KeyError as e:
                return {'valid': False, 'error': f'Missing metadata: {e}'}
            
            return {
                'valid': True,
                'shape': shape,
                'size_mb': os.path.getsize(h5_file) / (1024*1024),
                'bands': shape[2]
            }
    except Exception as e:
        return {'valid': False, 'error': str(e)}

def get_filtered_band_indices_safe():
    """
    Get band indices with water absorption bands removed
    Uses conservative approach with explicit band removal
    """
    # Start with all bands 0-424 (425 total)
    all_bands = np.arange(425)
    
    # Water absorption regions to remove (well-documented NEON bands)
    water_bands = np.concatenate([
        np.arange(192, 210),  # ~1400nm water absorption
        np.arange(283, 315),  # ~1900nm water absorption  
        np.arange(419, 425)   # ~2500nm (noisy edge bands)
    ])
    
    # Remove water absorption bands
    good_bands = np.setdiff1d(all_bands, water_bands)
    
    print(f"  Band filtering: {len(all_bands)} → {len(good_bands)} bands")
    print(f"  Removed bands: {len(water_bands)} water absorption bands")
    
    return good_bands

def convert_hsi_safe_with_validation(h5_file, output_dir):
    """
    Safe HSI conversion with data validation and integrity checking
    
    Args:
        h5_file: Path to HSI H5 file
        output_dir: Output directory (flat structure)
        
    Returns:
        dict: Conversion results with validation info
    """
    
    # Create output filename based on input filename
    input_basename = os.path.basename(h5_file)
    # Convert: NEON_D01_BART_DP3_317000_4882000_2019_reflectance.h5
    # To:     NEON_D01_BART_DP3_317000_4882000_2019_reflectance.tif
    output_file = input_basename.replace('_reflectance.h5', '_reflectance.tif')
    output_path = os.path.join(output_dir, output_file)
    
    # Extract coordinates for logging
    coords_match = re.search(r'(\d+)_(\d+)', input_basename)
    coords = f"{coords_match.group(1)}_{coords_match.group(2)}" if coords_match else "unknown"
    
    # Skip if already exists and validate it
    if os.path.exists(output_path):
        try:
            with rasterio.open(output_path) as src:
                if src.count > 300:  # Should have ~390 bands after filtering
                    return {'status': 'exists', 'path': output_path, 'coords': coords}
        except:
            # If existing file is corrupted, remove and reconvert
            os.remove(output_path)
    
    print(f"  Converting: {coords}")
    start_time = time.time()
    
    # Step 1: Validate input file
    validation = validate_hsi_file(h5_file)
    if not validation['valid']:
        return {
            'status': 'error',
            'error': f"Input validation failed: {validation['error']}",
            'coords': coords
        }
    
    try:
        # Step 2: Extract metadata and setup
        with h5py.File(h5_file, 'r') as hdf:
            sitename = list(hdf.keys())[0]
            refl_group = hdf[sitename]['Reflectance']
            refl_dataset = refl_group['Reflectance_Data']
            
            # Extract metadata (same as original, proven stable)
            epsg = str(refl_group['Metadata']['Coordinate_System']['EPSG Code'][()])
            epsg = epsg.split("'")[1] if "'" in epsg else epsg
            
            map_info = str(refl_group['Metadata']['Coordinate_System']['Map_Info'][()]).split(",")
            pixel_width = float(map_info[5])
            pixel_height = float(map_info[6])
            x_min = float(map_info[3])
            y_max = float(map_info[4])
            
            # Data attributes
            scale_factor = float(refl_dataset.attrs['Scale_Factor'])
            no_data_val = float(refl_dataset.attrs['Data_Ignore_Value'])
            
            # Get dimensions
            height, width, total_bands = refl_dataset.shape
            
            # Step 3: Get filtered band indices
            band_indices = get_filtered_band_indices_safe()
            
            # Step 4: Read and filter data (conservative approach)
            print(f"    Reading data: {height}x{width}x{len(band_indices)} bands")
            
            # Read ALL data at once (same as original - safe but memory intensive)
            refl_data = refl_dataset[:, :, band_indices]
            
            # Apply scale factor if needed (conservative)
            if scale_factor != 1.0:
                print(f"    Applying scale factor: {scale_factor}")
                refl_data = refl_data.astype(np.float32) * scale_factor
            else:
                refl_data = refl_data.astype(np.float32)
            
            # Step 5: Data validation before writing
            print(f"    Validating data...")
            if np.any(np.isnan(refl_data)):
                print(f"    Warning: Found NaN values, replacing with no_data_val")
                refl_data[np.isnan(refl_data)] = no_data_val
            
            # Check data range (reflectance should be 0-1 or 0-10000)
            valid_data = refl_data[refl_data != no_data_val]
            if len(valid_data) > 0:
                data_min, data_max = np.min(valid_data), np.max(valid_data)
                print(f"    Data range: {data_min:.3f} to {data_max:.3f}")
                
                # Warn about suspicious values
                if data_max > 2.0 and data_max < 100:  # Likely scaled 0-1 range
                    print(f"    Data appears to be in 0-1 range")
                elif data_max > 100:  # Likely 0-10000 range
                    print(f"    Data appears to be in 0-10000 range")
            
        # Step 6: Create geotransform
        transform = Affine.translation(x_min, y_max) * Affine.scale(pixel_width, -pixel_height)
        
        # Step 7: Rearrange dimensions (bands, rows, cols)
        refl_data = np.transpose(refl_data, (2, 0, 1))
        
        # Step 8: Write to GeoTIFF with conservative settings
        print(f"    Writing GeoTIFF...")
        
        # Conservative profile (minimal compression for safety)
        profile = {
            'driver': 'GTiff',
            'height': height,
            'width': width,
            'count': len(band_indices),
            'dtype': np.float32,
            'crs': f'EPSG:{epsg}',
            'transform': transform,
            'nodata': no_data_val,
            'compress': 'lzw',  # Safe compression
            'predictor': 2,     # Improves compression for float data
            'tiled': False      # Keep simple for compatibility
        }
        
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(refl_data)
        
        # Step 9: Verify output file
        print(f"    Verifying output...")
        try:
            with rasterio.open(output_path) as src:
                # Basic checks
                assert src.count == len(band_indices), "Band count mismatch"
                assert src.height == height, "Height mismatch"
                assert src.width == width, "Width mismatch"
                assert src.crs.to_string() == f'EPSG:{epsg}', "CRS mismatch"
                
                # Read a small sample to verify data integrity
                sample = src.read(1, window=rasterio.windows.Window(0, 0, 10, 10))
                assert not np.all(sample == no_data_val), "Output contains only no-data values"
                
        except Exception as e:
            # If verification fails, clean up and report error
            if os.path.exists(output_path):
                os.remove(output_path)
            return {
                'status': 'error',
                'error': f"Output verification failed: {e}",
                'coords': coords
            }
        
        # Step 10: Calculate final statistics
        conversion_time = time.time() - start_time
        output_size_mb = os.path.getsize(output_path) / (1024*1024)
        
        print(f"    ✅ Success: {conversion_time:.1f}s, {output_size_mb:.1f}MB")
        
        return {
            'status': 'success',
            'path': output_path,
            'coords': coords,
            'time': conversion_time,
            'size_mb': output_size_mb,
            'input_bands': total_bands,
            'output_bands': len(band_indices)
        }
        
    except Exception as e:
        # Clean up partial file
        if os.path.exists(output_path):
            os.remove(output_path)
        
        return {
            'status': 'error',
            'error': str(e),
            'coords': coords
        }

def convert_hsi_batch_safe(hsi_files, output_dir):
    """
    Convert multiple HSI files sequentially with safety checks
    
    Args:
        hsi_files: List of HSI file paths
        output_dir: Output directory (flat structure)
        
    Returns:
        dict: Conversion results and statistics
    """
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"🛡️  SAFE HSI CONVERSION")
    print(f"Files: {len(hsi_files)}")
    print(f"Mode: Sequential (data integrity priority)")
    print(f"Target: {output_dir}")
    print("="*50)
    
    results = {'success': [], 'errors': [], 'existing': []}
    start_time = time.time()
    
    for i, hsi_file in enumerate(hsi_files, 1):
        print(f"\n[{i}/{len(hsi_files)}] Processing: {os.path.basename(hsi_file)}")
        
        # Convert file (simplified - no need to extract coordinates separately)
        result = convert_hsi_safe_with_validation(hsi_file, output_dir)
        
        if result['status'] == 'success':
            results['success'].append(result)
        elif result['status'] == 'exists':
            results['existing'].append(result)
            print(f"  ✓ Already exists: {result['coords']}")
        else:
            results['errors'].append(result)
            print(f"  ❌ Error: {result['error']}")
    
    # Calculate statistics
    total_time = time.time() - start_time
    success_count = len(results['success'])
    existing_count = len(results['existing'])
    error_count = len(results['errors'])
    
    print(f"\n📊 CONVERSION SUMMARY")
    print("="*50)
    print(f"✅ New conversions: {success_count}")
    print(f"📁 Already existed: {existing_count}")
    print(f"❌ Errors: {error_count}")
    print(f"⏱️  Total time: {total_time:.1f}s")
    
    if success_count > 0:
        avg_time = np.mean([r['time'] for r in results['success']])
        total_size = sum([r['size_mb'] for r in results['success']])
        print(f"⚡ Avg time per file: {avg_time:.1f}s")
        print(f"💾 Total output size: {total_size:.1f}MB")
    
    return results

def main_safe_conversion():
    """
    Main function for safe HSI conversion
    Simple flat directory structure - all converted files in one folder
    """
    
    # Configuration
    flat_hsi_dir = "/blue/azare/riteshchowdhry/Macrosystems/Data_files/hand_annotated_neon/neon_tiles_FULL/flat_dir/HSI"
    output_dir = "/blue/azare/riteshchowdhry/Macrosystems/Data_files/hand_annotated_neon/neon_tiles_FULL/flat_dir/HSI_converted"
    
    # Find all HSI files
    hsi_pattern = os.path.join(flat_hsi_dir, '*_reflectance.h5')
    hsi_files = glob.glob(hsi_pattern)
    
    if not hsi_files:
        print(f"❌ No HSI files found in {flat_hsi_dir}")
        return
    
    print(f"Found {len(hsi_files)} HSI files")
    print(f"Converting all files to: {output_dir}")
    print(f"Output filename format: NEON_D01_BART_DP3_317000_4882000_2019_reflectance.tif")
    
    # Convert all files to single flat directory
    results = convert_hsi_batch_safe(hsi_files, output_dir)
    
    return results

def test_hsi_conversion(test_count=3):
    """
    Test HSI conversion on a small number of files first
    
    Args:
        test_count: Number of files to test (default: 3)
    
    Returns:
        dict: Test results
    """
    
    # Configuration
    flat_hsi_dir = "/blue/azare/riteshchowdhry/Macrosystems/Data_files/hand_annotated_neon/neon_tiles_FULL/flat_dir/HSI"
    output_dir = "/blue/azare/riteshchowdhry/Macrosystems/Data_files/hand_annotated_neon/neon_tiles_FULL/flat_dir/HSI_converted_TEST"
    
    print(f"🧪 TESTING HSI CONVERSION")
    print(f"Test files: {test_count}")
    print(f"Source: {flat_hsi_dir}")
    print(f"Test output: {output_dir}")
    print(f"Output format: NEON_D01_BART_DP3_317000_4882000_2019_reflectance.tif")
    print("="*50)
    
    # Find all HSI files
    hsi_pattern = os.path.join(flat_hsi_dir, '*_reflectance.h5')
    all_hsi_files = glob.glob(hsi_pattern)
    
    if not all_hsi_files:
        print(f"❌ No HSI files found in {flat_hsi_dir}")
        return {'error': 'No HSI files found'}
    
    print(f"Found {len(all_hsi_files)} total HSI files")
    
    # Select test files (first few files)
    test_files = all_hsi_files[:test_count]
    
    print(f"\nSelected test files:")
    for i, file in enumerate(test_files, 1):
        basename = os.path.basename(file)
        size_mb = os.path.getsize(file) / (1024*1024)
        print(f"  {i}. {basename} ({size_mb:.1f}MB)")
    
    # Process test files (simple flat structure)
    overall_start = time.time()
    
    print(f"\n🎯 Testing conversion to flat directory")
    results = convert_hsi_batch_safe(test_files, output_dir)
    
    # Calculate overall test statistics
    total_test_time = time.time() - overall_start
    total_success = len(results['success'])
    total_errors = len(results['errors'])
    
    print(f"\n🎯 TEST RESULTS SUMMARY")
    print("="*50)
    print(f"✅ Successful conversions: {total_success}/{test_count}")
    print(f"❌ Failed conversions: {total_errors}/{test_count}")
    print(f"⏱️  Total test time: {total_test_time:.1f}s")
    
    if total_success > 0:
        # Calculate estimated time for all files
        avg_time_per_file = total_test_time / test_count
        estimated_total_time = avg_time_per_file * len(all_hsi_files)
        estimated_hours = estimated_total_time / 3600
        
        print(f"⚡ Avg time per file: {avg_time_per_file:.1f}s")
        print(f"📊 Estimated time for all {len(all_hsi_files)} files: {estimated_hours:.1f} hours")
        
        # Check output files
        test_output_files = [result['path'] for result in results['success']]
        
        if test_output_files:
            print(f"\n✅ Test output files created:")
            for output_file in test_output_files:
                size_mb = os.path.getsize(output_file) / (1024*1024)
                print(f"  📁 {os.path.basename(output_file)} ({size_mb:.1f}MB)")
    
    # Provide recommendation
    if total_errors == 0:
        print(f"\n🎉 TEST PASSED! All files converted successfully.")
        print(f"💡 Ready to run full conversion with main_safe_conversion()")
    else:
        print(f"\n⚠️  TEST HAD ERRORS! Check error messages above.")
        print(f"💡 Fix issues before running full conversion.")
    
    print(f"\n🗑️  To clean up test files:")
    print(f"rm -rf {output_dir}")
    
    return {
        'test_count': test_count,
        'total_files': len(all_hsi_files),
        'success_count': total_success,
        'error_count': total_errors,
        'test_time': total_test_time,
        'estimated_full_time_hours': estimated_hours if total_success > 0 else None,
        'test_output_dir': output_dir,
        'results': results
    }

if __name__ == "__main__":
    # Choose test mode or full conversion
    print("🛡️  SAFE HSI CONVERSION TOOL")
    print("="*50)
    print("Options:")
    print("  1. Test conversion (3 files)")
    print("  2. Full conversion (all files)")
    print()
    
    choice = input("Choose option (1 or 2): ").strip()
    
    if choice == "1":
        print("\n🧪 Running test conversion...")
        test_results = test_hsi_conversion(test_count=4)
    elif choice == "2":
        print("\n🚀 Running full conversion...")
        results = main_safe_conversion()
    else:
        print("❌ Invalid choice. Exiting.")
        print("\nTo run directly:")
        print("  Test: python -c 'from hsi_conversion_safe import test_hsi_conversion; test_hsi_conversion()'")
        print("  Full: python -c 'from hsi_conversion_safe import main_safe_conversion; main_safe_conversion()'")
