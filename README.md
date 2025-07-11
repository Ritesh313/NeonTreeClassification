# NEON Tree Classification

A Python package for processing NEON (National Ecological Observatory Network) tree crown annotation data and building machine learning models for tree species classification.

National Ecological Observatory Network (NEON) offers a variety of data products, including airborne data from different forest sites. Airborne data includes RGB orthophotos, LiDAR (CHM) airborne data, and 426 band hyperspectral data. All products are available on https://data.neonscience.org/data-products/.

## NEON Data Products Used

- **RGB Orthophotos**: `DP3.30010.001` - High-resolution orthorectified camera imagery mosaic
- **Hyperspectral Imagery**: `DP3.30006.002` - Spectrometer orthorectified surface bidirectional reflectance - mosaic  
- **LiDAR CHM**: `DP3.30015.001` - Ecosystem structure (Canopy Height Model)

## Features

- **Shapefile Processing**: Handle coordinate system transformations and validation for NEON tree crown shapefiles
- **HSI Tile Processing**: Convert and process hyperspectral imagery (HSI) tiles from H5 to GeoTIFF format
- **Crown-Tile Intersection**: Match tree crown annotations with corresponding image tiles
- **Data Pipeline**: End-to-end processing from raw NEON data to training-ready datasets
- **Coordinate Validation**: Robust handling of invalid coordinates and CRS issues

## Installation

```bash
# Clone the repository
git clone https://github.com/Ritesh313/NeonTreeClassification.git
cd NeonTreeClassification

# Install in development mode
pip install -e .
```

## Quick Start

### 1. Process Shapefiles
```bash
python scripts/test_shapefile_processor.py
```

### 2. Process Tiles and Crowns
```bash
python scripts/process_tiles_to_crowns.py
```

## Package Structure

```
neon_tree_classification/
├── data/
│   └── shapefile_processor.py    # Shapefile processing and CRS handling
├── models/
│   └── hsi_models.py            # PyTorch models for HSI classification
├── processing/
│   └── __init__.py              # Processing utilities
└── utils/
    └── __init__.py              # General utilities

scripts/
├── download_neon_all_modalities.py  # Data download scripts
├── process_tiles_to_crowns.py       # Main processing pipeline
└── test_shapefile_processor.py      # Test shapefile processing

configs/                         # Configuration files
notebooks/                       # Jupyter notebooks for analysis
SLURM/                          # SLURM job scripts
tests/                          # Unit tests
```

## Workflow

### 1. Download NEON Data
Given the Northing, Easting, Year and Site, download the NEON data using the download scripts. There are functions to download the RGB, HSI, and LiDAR data.

### 2. Process Shapefiles  
Process tree crown annotation shapefiles with coordinate system correction and validation.

### 3. Process Tiles and Crowns
Run the full pipeline to match crown annotations with image tiles and create training datasets.

## Usage Examples

### Processing NEON Shapefiles

```python
from neon_tree_classification.data.shapefile_processor import ShapefileProcessor

processor = ShapefileProcessor()

# Consolidate shapefiles from subdirectories
processor.consolidate_files(parent_dir, destination_dir)

# Process with coordinate validation and CRS correction
sites_df, summary = processor.process_shapefiles(destination_dir)
```

### Running the Tile Processing Pipeline

```python
from scripts.process_tiles_to_crowns import run_full_pipeline

results = run_full_pipeline(
    tiles_base_dir="/path/to/neon_tiles",
    crown_csv_path="/path/to/clean_coordinates.csv", 
    site="BART",
    year="2019",
    output_base_dir="/path/to/output"
)
```

## Key Features

### Coordinate System Handling
- Automatic UTM zone detection by NEON site
- CRS transformation and validation
- Invalid coordinate filtering (infinite values, out-of-range)

### Multi-Modal Processing
- RGB imagery (GeoTIFF)
- Hyperspectral imagery (H5 → GeoTIFF conversion) 
- LiDAR CHM data (GeoTIFF)

### Robust Data Validation
- Geometry validation and cleaning
- Coordinate range checking
- File existence verification
- Error handling and reporting

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## Authors

- Ritesh Chowdhry

## Acknowledgments

- National Ecological Observatory Network (NEON) for providing the data
- University of Florida Macrosystems project

# Citations

## NEON Airborne Data Products
NEON (National Ecological Observatory Network). High-resolution orthorectified camera imagery mosaic (DP3.30010.001), RELEASE-2025. https://doi.org/10.48443/gdgn-3r69. Dataset accessed from https://data.neonscience.org/data-products/DP3.30010.001/RELEASE-2025 on April 3, 2025.

NEON (National Ecological Observatory Network). Spectrometer orthorectified surface bidirectional reflectance - mosaic (DP3.30006.002), provisional data. Dataset accessed from https://data.neonscience.org/data-products/DP3.30006.002 on April 3, 2025. Data archived at [your DOI].

NEON (National Ecological Observatory Network). Ecosystem structure (DP3.30015.001), RELEASE-2025. https://doi.org/10.48443/jqqd-1n30. Dataset accessed from https://data.neonscience.org/data-products/DP3.30015.001/RELEASE-2025 on April 3, 2025.

