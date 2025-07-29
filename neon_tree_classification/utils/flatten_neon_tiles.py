#!/usr/bin/env python3
"""
Flatten NEON Tile Directory Structure

This script moves NEON tiles from the deeply nested download structure
to a simple flat structure for easier processing.

Author: Ritesh Chowdhry
"""

import os
import shutil
import glob
from pathlib import Path

def flatten_neon_tiles(source_dir, target_dir):
    """
    Flatten NEON tiles from nested structure to simple structure
    
    Args:
        source_dir: Path to neon_tiles_FULL directory
        target_dir: Path to new flattened directory
    """
    
    print("🗂️  FLATTENING NEON TILE STRUCTURE")
    print("="*60)
    print(f"Source: {source_dir}")
    print(f"Target: {target_dir}")
    print()
    
    # Create target directories
    modalities = ['RGB', 'HSI', 'LiDAR']
    for modality in modalities:
        target_modality_dir = os.path.join(target_dir, modality)
        os.makedirs(target_modality_dir, exist_ok=True)
    
    # Track statistics
    stats = {
        'RGB': {'found': 0, 'moved': 0, 'skipped': 0},
        'HSI': {'found': 0, 'moved': 0, 'skipped': 0},
        'LiDAR': {'found': 0, 'moved': 0, 'skipped': 0}
    }
    
    # Track special cases
    special_cases = {
        'unknown_dirs': set(),
        'duplicate_dirs': set(),
        'empty_dirs': set()
    }
    
    # Process RGB files
    print("📸 Processing RGB files...")
    rgb_pattern = os.path.join(source_dir, 'RGB', '**', '*_image.tif')
    rgb_files = glob.glob(rgb_pattern, recursive=True)
    stats['RGB']['found'] = len(rgb_files)
    
    for rgb_file in rgb_files:
        filename = os.path.basename(rgb_file)
        target_path = os.path.join(target_dir, 'RGB', filename)
        
        if os.path.exists(target_path):
            print(f"  ⚠️  Skipping {filename} (already exists)")
            stats['RGB']['skipped'] += 1
        else:
            shutil.move(rgb_file, target_path)
            print(f"  ✅ Moved {filename}")
            stats['RGB']['moved'] += 1
    
    # Process HSI files
    print("\n🌈 Processing HSI files...")
    hsi_pattern = os.path.join(source_dir, 'HSI', '**', '*_reflectance.h5')
    hsi_files = glob.glob(hsi_pattern, recursive=True)
    stats['HSI']['found'] = len(hsi_files)
    
    for hsi_file in hsi_files:
        filename = os.path.basename(hsi_file)
        
        # Extract year from directory path
        # Path like: .../HSI/JERC_2019/DP3.../2019/FullSite/...
        # Handle cases like SITE_unknown, SITE_2019 (1), etc.
        path_parts = hsi_file.split(os.sep)
        year = None
        site_folder = None
        
        # First, find the site folder (e.g., "JERC_2019", "HARV_unknown", "BART_2019 (1)")
        for part in path_parts:
            if any(site in part for site in ['BART', 'HARV', 'JERC', 'ABBY', 'BONA', 'CLBJ', 
                                           'DEJU', 'DELA', 'GRSM', 'HEAL', 'LENO', 'MLBS', 
                                           'MOAB', 'NIWO', 'OSBS', 'SCBI', 'SERC', 'SJER', 
                                           'SRER', 'TALL', 'TEAK', 'UNDE', 'WREF']):
                site_folder = part
                break
        
        # Extract year from site folder or path
        if site_folder:
            if 'unknown' in site_folder.lower():
                year = 'unknown'
            elif '(' in site_folder:
                # Handle cases like "BART_2019 (1)"
                base_part = site_folder.split('(')[0].strip()
                year_match = [p for p in base_part.split('_') if p.startswith(('201', '202')) and p.isdigit()]
                if year_match:
                    year = f"{year_match[0]}_dup{site_folder.split('(')[1].split(')')[0]}"
                else:
                    year = 'unknown'
            else:
                # Normal case like "JERC_2019"
                year_match = [p for p in site_folder.split('_') if p.startswith(('201', '202')) and p.isdigit()]
                if year_match:
                    year = year_match[0]
        
        # Fallback: look for 4-digit year anywhere in path
        if not year:
            for part in path_parts:
                if part.startswith(('201', '202')) and part.isdigit() and len(part) == 4:
                    year = part
                    break
        
        # Final fallback
        if not year:
            year = 'unknown'
        
        if year:
            # Insert year into filename: NEON_D03_JERC_DP3_742000_3454000_reflectance.h5
            # becomes: NEON_D03_JERC_DP3_742000_3454000_2019_reflectance.h5
            # or: NEON_D03_JERC_DP3_742000_3454000_unknown_reflectance.h5
            # or: NEON_D03_JERC_DP3_742000_3454000_2019_dup1_reflectance.h5
            name_parts = filename.split('_')
            if len(name_parts) >= 6:
                name_parts.insert(-1, year)  # Insert year before "reflectance.h5"
                new_filename = '_'.join(name_parts)
            else:
                new_filename = f"{filename.rsplit('.', 1)[0]}_{year}.{filename.rsplit('.', 1)[1]}"
        else:
            new_filename = filename
        
        target_path = os.path.join(target_dir, 'HSI', new_filename)
        
        if os.path.exists(target_path):
            print(f"  ⚠️  Skipping {new_filename} (already exists)")
            stats['HSI']['skipped'] += 1
        else:
            shutil.move(hsi_file, target_path)
            print(f"  ✅ Moved {filename} → {new_filename}")
            stats['HSI']['moved'] += 1
    
    # Process LiDAR files
    print("\n🌲 Processing LiDAR files...")
    lidar_pattern = os.path.join(source_dir, 'LiDAR', '**', '*_CHM.tif')
    lidar_files = glob.glob(lidar_pattern, recursive=True)
    stats['LiDAR']['found'] = len(lidar_files)
    
    for lidar_file in lidar_files:
        filename = os.path.basename(lidar_file)
        
        # Extract year from directory path
        # Path like: .../LiDAR/JERC_2019/DP3.../2019/FullSite/...
        # Handle cases like SITE_unknown, SITE_2019 (1), etc.
        path_parts = lidar_file.split(os.sep)
        year = None
        site_folder = None
        
        # First, find the site folder (e.g., "JERC_2019", "HARV_unknown", "BART_2019 (1)")
        for part in path_parts:
            if any(site in part for site in ['BART', 'HARV', 'JERC', 'ABBY', 'BONA', 'CLBJ', 
                                           'DEJU', 'DELA', 'GRSM', 'HEAL', 'LENO', 'MLBS', 
                                           'MOAB', 'NIWO', 'OSBS', 'SCBI', 'SERC', 'SJER', 
                                           'SRER', 'TALL', 'TEAK', 'UNDE', 'WREF']):
                site_folder = part
                break
        
        # Extract year from site folder or path
        if site_folder:
            if 'unknown' in site_folder.lower():
                year = 'unknown'
            elif '(' in site_folder:
                # Handle cases like "BART_2019 (1)"
                base_part = site_folder.split('(')[0].strip()
                year_match = [p for p in base_part.split('_') if p.startswith(('201', '202')) and p.isdigit()]
                if year_match:
                    year = f"{year_match[0]}_dup{site_folder.split('(')[1].split(')')[0]}"
                else:
                    year = 'unknown'
            else:
                # Normal case like "JERC_2019"
                year_match = [p for p in site_folder.split('_') if p.startswith(('201', '202')) and p.isdigit()]
                if year_match:
                    year = year_match[0]
        
        # Fallback: look for 4-digit year anywhere in path
        if not year:
            for part in path_parts:
                if part.startswith(('201', '202')) and part.isdigit() and len(part) == 4:
                    year = part
                    break
        
        # Final fallback
        if not year:
            year = 'unknown'
        
        if year:
            # Insert year into filename: NEON_D03_JERC_DP3_742000_3454000_CHM.tif
            # becomes: NEON_D03_JERC_DP3_742000_3454000_2019_CHM.tif
            # or: NEON_D03_JERC_DP3_742000_3454000_unknown_CHM.tif
            # or: NEON_D03_JERC_DP3_742000_3454000_2019_dup1_CHM.tif
            name_parts = filename.split('_')
            if len(name_parts) >= 6:
                name_parts.insert(-1, year)  # Insert year before "CHM.tif"
                new_filename = '_'.join(name_parts)
            else:
                new_filename = f"{filename.rsplit('.', 1)[0]}_{year}.{filename.rsplit('.', 1)[1]}"
        else:
            new_filename = filename
        
        target_path = os.path.join(target_dir, 'LiDAR', new_filename)
        
        if os.path.exists(target_path):
            print(f"  ⚠️  Skipping {new_filename} (already exists)")
            stats['LiDAR']['skipped'] += 1
        else:
            shutil.move(lidar_file, target_path)
            print(f"  ✅ Moved {filename} → {new_filename}")
            stats['LiDAR']['moved'] += 1
    
    print(f"\n📊 FLATTENING SUMMARY")
    print("="*60)
    for modality in modalities:
        s = stats[modality]
        print(f"{modality:>6}: {s['found']:>4} found, {s['moved']:>4} moved, {s['skipped']:>4} skipped")
    
    total_moved = sum(s['moved'] for s in stats.values())
    total_found = sum(s['found'] for s in stats.values())
    print(f"{'TOTAL':>6}: {total_found:>4} found, {total_moved:>4} moved")
    
    # Report on special cases
    if special_cases['unknown_dirs']:
        print(f"\n⚠️  Found {len(special_cases['unknown_dirs'])} 'unknown' directories:")
        for dir_name in sorted(special_cases['unknown_dirs']):
            print(f"   • {dir_name}")
    
    if special_cases['duplicate_dirs']:
        print(f"\n🔄 Found {len(special_cases['duplicate_dirs'])} duplicate directories:")
        for dir_name in sorted(special_cases['duplicate_dirs']):
            print(f"   • {dir_name}")
    
    if not any(special_cases.values()):
        print(f"\n✅ No problematic directories encountered!")
    
    print(f"\n✅ Flattening completed!")
    print(f"Files now organized in: {target_dir}")
    
    return stats

def cleanup_empty_directories(source_dir):
    """Remove empty directories after moving files"""
    
    print(f"\n🧹 Cleaning up empty directories...")
    
    # Walk through directories bottom-up to remove empty ones
    for root, dirs, files in os.walk(source_dir, topdown=False):
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            try:
                if not os.listdir(dir_path):  # Directory is empty
                    os.rmdir(dir_path)
                    print(f"  🗑️  Removed empty directory: {dir_path}")
            except OSError:
                # Directory not empty or permission issue
                pass
    
    print("✅ Cleanup completed!")

def main():
    """Main function"""
    
    # Default paths
    source_dir = "/blue/azare/riteshchowdhry/Macrosystems/Data_files/hand_annotated_neon/neon_tiles_FULL"
    target_dir = "/blue/azare/riteshchowdhry/Macrosystems/Data_files/hand_annotated_neon/neon_tiles_FULL/flat_dir"
    
    # Check if source exists
    if not os.path.exists(source_dir):
        print(f"❌ Source directory not found: {source_dir}")
        return
    
    # Ask for confirmation
    print("⚠️  This will move files from nested structure to flat structure.")
    print(f"Source: {source_dir}")
    print(f"Target: {target_dir}")
    print()
    
    response = input("Proceed? (yes/no): ").lower().strip()
    if response not in ['yes', 'y']:
        print("❌ Operation cancelled.")
        return
    
    # Create target directory
    os.makedirs(target_dir, exist_ok=True)
    
    # Flatten the structure
    stats = flatten_neon_tiles(source_dir, target_dir)
    
    # Ask about cleanup
    print()
    response = input("Remove empty directories from source? (yes/no): ").lower().strip()
    if response in ['yes', 'y']:
        cleanup_empty_directories(source_dir)
    
    print(f"\n🎉 All done! Your tiles are now in: {target_dir}")

if __name__ == "__main__":
    main()
