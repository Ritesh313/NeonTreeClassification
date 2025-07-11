#!/usr/bin/env python3
"""
Test script for the ShapefileProcessor.

This script tests the ShapefileProcessor class that handles
coordinate system issues properly.
"""

import sys
import os
sys.path.append('/blue/azare/riteshchowdhry/Macrosystems/code/NeonTreeClassification')

from neon_tree_classification.data.shapefile_processor import ShapefileProcessor

def main():
    # Set up paths for clean annotations
    parent_dir = "/blue/azare/riteshchowdhry/Macrosystems/Data_files/hand_annotated_neon/clean_annotations"
    destination_dir = "/blue/azare/riteshchowdhry/Macrosystems/Data_files/hand_annotated_neon/clean_annotations/consolidated_dir"
    
    # Create destination directory if it doesn't exist
    os.makedirs(destination_dir, exist_ok=True)
    
    print("Testing ShapefileProcessor")
    print("=" * 50)
    print(f"Source directory: {parent_dir}")
    print(f"Destination directory: {destination_dir}")
    print()
    
    # Create processor instance
    processor = ShapefileProcessor()
    
    try:
        # First consolidate files (like the original curate_shp_files.py)
        print("Step 1: Consolidating shapefiles...")
        errors = processor.consolidate_files(
            parent_dir=parent_dir,
            destination_dir=destination_dir
        )
        
        print(f"Consolidation complete.")
        print(f"Errors during consolidation: {len(errors)}")
        if errors:
            for file_path, error_msg in errors:
                print(f"  Error with {file_path}: {error_msg}")
        
        # Then process the consolidated shapefiles with coordinate fixes
        print("\nStep 2: Processing coordinates with CRS fixes...")
        df, summary = processor.process_shapefiles(
            consolidated_dir=destination_dir,
            output_filename='neon_sites_coordinates.csv'
        )
        
        print("\n" + "=" * 50)
        print("PROCESSING COMPLETE!")
        print(f"Total files: {summary['total_files']}")
        print(f"Successfully processed: {summary['successful']}")
        print(f"Files reprojected: {summary['reprojected']}")
        print(f"Errors: {summary['errors']}")
        print(f"Empty geometries skipped: {summary['empty_geometries']}")
        
        # Show sample of results
        if len(df) > 0:
            print(f"\nSample results (first 5 rows):")
            print(df.head().to_string())
            
            # Show coordinate ranges to verify they look correct
            print(f"\nCoordinate ranges:")
            print(f"X (Easting): {df['center_easting'].min():.0f} to {df['center_easting'].max():.0f}")
            print(f"Y (Northing): {df['center_northing'].min():.0f} to {df['center_northing'].max():.0f}")
        
    except Exception as e:
        print(f"Error running processor: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
