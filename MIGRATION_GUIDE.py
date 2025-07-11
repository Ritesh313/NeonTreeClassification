#!/usr/bin/env python3
"""
Repository Migration Helper

This script helps users navigate the reorganized NEON Tree Classification repository
and provides guidance on which files to use.
"""

import os
import sys

def print_migration_guide():
    """Print guidance for using the reorganized repository."""
    
    print("🔄 NEON TREE CLASSIFICATION REPOSITORY REORGANIZED")
    print("="*60)
    print()
    
    print("📁 NEW STRUCTURE:")
    print("  neon_tree_classification/        # Main Python package")
    print("  ├── data/shapefile_processor.py  # Shapefile processing")
    print("  ├── models/architectures.py      # ML models")
    print("  └── processing/, utils/          # Supporting modules")
    print()
    print("  scripts/                         # Executable scripts")
    print("  ├── download_neon_all_modalities.py")
    print("  ├── process_tiles_to_crowns.py")
    print("  └── test_shapefile_processor.py")
    print()
    
    print("🚨 DEPRECATED FILES (do not use):")
    print("  src/curate_shp_files.py          → Use: scripts/test_shapefile_processor.py")
    print("  src/models.py                    → Use: neon_tree_classification.models.architectures")
    print("  src/download_neon_all_modalities.py → Use: scripts/download_neon_all_modalities.py")
    print("  scripts/download_neon_data.py    → Use: scripts/download_neon_all_modalities.py")
    print("  neon_tree_classification/models/hsi_models.py → Use: architectures.py")
    print()
    
    print("🎯 QUICK START:")
    print("  # Process shapefiles:")
    print("  python scripts/test_shapefile_processor.py")
    print()
    print("  # Process tiles and crowns:")
    print("  python scripts/process_tiles_to_crowns.py --site BART --year 2019")
    print()
    print("  # Download NEON data:")
    print("  python scripts/download_neon_all_modalities.py --mode test")
    print()
    
    print("📦 PYTHON PACKAGE USAGE:")
    print("  from neon_tree_classification.data.shapefile_processor import ShapefileProcessor")
    print("  from neon_tree_classification.models.architectures import HsiPixelClassifier")
    print()
    
    print("✅ TESTED AND WORKING:")
    print("  • Shapefile processing with coordinate validation")
    print("  • HSI tile conversion and cropping")
    print("  • Crown-tile intersection processing")
    print("  • BART 2019 test dataset processing")
    print()
    
    print("📋 NEXT STEPS:")
    print("  1. Test your workflow with the new script locations")
    print("  2. Update any custom scripts to use the new imports")
    print("  3. Remove or ignore the deprecated src/ files")
    print("  4. Use --help flag on scripts to see all options")

if __name__ == "__main__":
    print_migration_guide()
