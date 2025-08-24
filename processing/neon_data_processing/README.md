# NEON Data Processing Tools

This directory contains tools for processing downloaded NEON airborne data products (RGB, HSI, LiDAR) into machine learning-ready formats.

## Overview

NEON data downloads have complex nested directory structures that are difficult to work with directly. These tools flatten and organize the data with standardized naming conventions.

## Tools

### `curate_tiles.py`
Processes downloaded NEON tiles from their deeply nested directory structure into a flattened, organized format suitable for ML workflows.

**Input Structure:**
```
downloaded_tiles/
├── SITE_YEAR/
│   ├── rgb/DP3.../neon-aop-products/year/FullSite/.../YYYY_SITE_#_EASTING_NORTHING_image.tif
│   ├── hsi/DP3.../neon-aop-products/year/FullSite/.../NEON_D##_SITE_DP3_EASTING_NORTHING_reflectance.h5
│   └── lidar/DP3.../neon-aop-products/year/FullSite/.../NEON_D##_SITE_DP3_EASTING_NORTHING_CHM.tif
```

**Output Structure:**
```
curated_tiles/
├── rgb/SITE_YEAR_EASTING_NORTHING_rgb.tif
├── hsi_tif/SITE_YEAR_EASTING_NORTHING_hsi.tif  # Converted from H5
└── lidar/SITE_YEAR_EASTING_NORTHING_lidar.tif
```

**Key Features:**
- Handles complex NEON directory nesting automatically
- Matches tiles across RGB, HSI, and LiDAR modalities by coordinates
- Creates standardized filenames for easy identification
- Only processes complete tile sets (all 3 modalities present)
- Option for flat structure or modality subdirectories
- Preserves original files by default

### `hsi_convert_h5_to_tif.py`
Converts HSI files from HDF5 (.h5) format to GeoTIFF (.tif) format for easier processing and compatibility with standard geospatial tools.

### `crop_crowns_multimodal.py`
Crops individual tree crowns from NEON multi-modal tiles (RGB, LiDAR, HSI) using crown polygon data. Designed for machine learning workflows requiring individual tree-level data.

### `create_training_csv.py`
Creates training-ready CSV files by combining cropped crown data with species labels from NEON Vegetation Structure and Traits (VST) data. Simple script that merges crop metadata with species labels for machine learning workflows.

**Input Requirements:**
- Curated NEON tiles directory with `rgb/`, `lidar/`, and `hsi_tif/` subdirectories
- Crown polygon data in GeoPackage (.gpkg) or Shapefile format with individual tree locations
- Crown data must include `siteID`, `year`, and individual identification columns

**Output Structure (Flat - Default):**
```
output_dir/
├── SITE_YEAR_INDIVIDUAL_CROWNIDX_rgb.tif
├── SITE_YEAR_INDIVIDUAL_CROWNIDX_lidar.tif
├── SITE_YEAR_INDIVIDUAL_CROWNIDX_hsi.tif
└── crop_metadata.csv
```

**Output Structure (Modality Subdirectories):**
```
output_dir/
├── rgb/SITE_YEAR_INDIVIDUAL_CROWNIDX.tif
├── lidar/SITE_YEAR_INDIVIDUAL_CROWNIDX.tif  
├── hsi/SITE_YEAR_INDIVIDUAL_CROWNIDX.tif
└── crop_metadata.csv
```

**Key Features:**
- Site-specific UTM coordinate transformations for accurate cropping
- Multi-modal alignment ensuring each crown has RGB, LiDAR, and HSI crops
- Configurable buffer around crown polygons
- Spatial indexing for efficient crown-tile matching
- Comprehensive metadata logging with processing timestamps
- Flexible output organization (flat or modality-organized)
- Robust error handling and progress tracking

## Usage

### Basic Usage - Tile Curation
```bash
python curate_tiles.py --input-dir downloaded_neon_tiles/ --output-dir curated_tiles/
```

### Basic Usage - Crown Cropping
```bash
python crop_crowns_multimodal.py \
  --tiles_dir curated_tiles/ \
  --crowns_gpkg crown_polygons.gpkg \
  --output_dir cropped_crowns/
```

### Basic Usage - Training CSV Creation
```bash
python create_training_csv.py \
  --crop_metadata cropped_crowns/crop_metadata.csv \
  --vst_labels neon_vst_data.csv \
  --output training_data.csv
```

### Crown Cropping Options
- `--modality_subdir`: Organize output in modality subdirectories (rgb/, lidar/, hsi/)
- `--buffer FLOAT`: Buffer around crowns in meters (default: 2.0)
- `--site SITE`: Filter crowns by NEON site code
- `--year YEAR`: Filter crowns by year
- `--max_crowns N`: Limit processing to N crowns (for testing)

### Tile Curation Options
- `--delete-originals`: Delete source files after copying (default: preserve originals)
- `--flat-structure`: Create completely flat structure without modality subdirectories
- `--dry-run`: Preview what would be processed without actually copying files

### Example Commands

#### Tile Curation
```bash
# Standard curation with modality subdirectories
python curate_tiles.py \
  --input-dir /path/to/downloaded_neon_tiles_20250819 \
  --output-dir /path/to/curated_tiles_20250819

# Dry run to see what would be processed
python curate_tiles.py \
  --input-dir /path/to/downloaded_neon_tiles_20250819 \
  --output-dir /path/to/curated_tiles_20250819 \
  --dry-run

# Completely flat structure
python curate_tiles.py \
  --input-dir /path/to/downloaded_neon_tiles_20250819 \
  --output-dir /path/to/curated_tiles_flat \
  --flat-structure
```

#### Crown Cropping
```bash
# Basic cropping with flat output structure
python crop_crowns_multimodal.py \
  --tiles_dir curated_tiles_20250819/ \
  --crowns_gpkg crown_polygons.gpkg \
  --output_dir cropped_crowns_flat/

# Organized by modality subdirectories
python crop_crowns_multimodal.py \
  --tiles_dir curated_tiles_20250819/ \
  --crowns_gpkg crown_polygons.gpkg \
  --output_dir cropped_crowns_organized/ \
  --modality_subdir

# Filter by site with custom buffer
python crop_crowns_multimodal.py \
  --tiles_dir curated_tiles_20250819/ \
  --crowns_gpkg crown_polygons.gpkg \
  --output_dir cropped_crowns_harv/ \
  --site HARV \
  --buffer 3.0

# Test run with limited crowns
python crop_crowns_multimodal.py \
  --tiles_dir curated_tiles_20250819/ \
  --crowns_gpkg crown_polygons.gpkg \
  --output_dir cropped_crowns_test/ \
  --max_crowns 10
```

## Important Notes

- **Complete sets only**: Only processes coordinates with all 3 modalities (RGB, HSI, LiDAR)
- **Safe operations**: Original files are preserved by default
- **Skip existing**: Re-running will skip files that already exist in output directory
- **Coordinate matching**: Uses regex patterns to extract coordinates from NEON filenames
- **Large files**: Process can be slow due to large HSI files (several GB each)

## Filename Patterns Recognized

- **RGB**: `YYYY_SITE_#_EASTING_NORTHING_image.tif`
- **HSI**: `NEON_D##_SITE_DP3_EASTING_NORTHING_reflectance.h5`  
- **LiDAR**: `NEON_D##_SITE_DP3_EASTING_NORTHING_CHM.tif`

## Expected Workflow

### Complete Processing Pipeline
1. **Download NEON data** using `../neon_downloader.py` 
2. **Curate tiles** with `curate_tiles.py` to flatten and organize the data
3. **Convert HSI format** with `hsi_convert_h5_to_tif.py` (if needed)
4. **Crop individual crowns** with `crop_crowns_multimodal.py` using crown polygon data
5. **Create training CSV** with `create_training_csv.py` to combine crops with species labels
6. **Use for machine learning** model training and evaluation

### For Tile-Level Analysis
1. Download NEON data using `../neon_downloader.py`
2. Run `curate_tiles.py` to flatten and organize the data  
3. Use curated tiles directly for tile-level machine learning workflows

### For Individual Tree Analysis  
1. Complete steps 1-3 above
4. Obtain crown polygon data (from field surveys, automated detection, etc.)
5. Run `crop_crowns_multimodal.py` to extract individual tree crops
6. Use individual crown crops for tree-level classification, detection, or analysis