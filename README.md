# NEON Tree Classification

A modular Python package for processing NEON airborne data and multi-modal tree species classification using RGB, hyperspectral, and LiDAR data.

## Features

### Data Processing
- **NEON data download**: Automated download of RGB, hyperspectral, and LiDAR tiles
- **Shapefile processing**: Coordinate system transformations and validation for tree crowns
- **Multi-modal tile processing**: Convert and process HSI (H5 → GeoTIFF), RGB, and LiDAR data
- **Crown-tile intersection**: Match tree crown annotations with corresponding image tiles

### Machine Learning
- **Multi-modal models**: Separate architectures for RGB, hyperspectral (426 bands), and LiDAR
- **Modular training**: PyTorch Lightning modules with CometML/TensorBoard logging
- **Flexible data pipeline**: Clean tensor-only batches with configurable splits
- **Modern packaging**: Uses `pyproject.toml` and `uv` for dependency management

## Installation

```bash
git clone https://github.com/Ritesh313/NeonTreeClassification.git
cd NeonTreeClassification

# Install with uv (recommended)
pip install uv
uv sync

# Or with pip
pip install -e .
```

## Quick Start

### Data Processing
```bash
# Process NEON shapefiles
python scripts/test_shapefile_processor.py

# Process tiles and match with crowns
python scripts/process_tiles_to_crowns.py
```

### Model Training
```bash
# Train RGB model
python train.py --modality rgb --csv_path data/crowns.csv --data_dir data/

# Train HSI model with CometML logging
python train.py --modality hsi --logger comet --project_name my-project

# Compare all modalities
python compare_modalities.py --csv_path data/crowns.csv --data_dir data/
```

### Using in code
```python
# Data processing
from neon_tree_classification.data.shapefile_processor import ShapefileProcessor

processor = ShapefileProcessor()
sites_df, summary = processor.process_shapefiles(destination_dir)

# Model training
from neon_tree_classification import NeonCrownDataModule, RGBClassifier

datamodule = NeonCrownDataModule(
    csv_path="data/crowns.csv",
    base_data_dir="data/",
    modalities=["rgb"],
    batch_size=32
)

classifier = RGBClassifier(model_type="resnet", num_classes=10)

import lightning as L
trainer = L.Trainer(max_epochs=50)
trainer.fit(classifier, datamodule)
```

## Architecture

```
neon_tree_classification/
├── data/
│   ├── dataset.py            # Multi-modal dataset
│   ├── datamodule.py         # Lightning DataModule
│   └── shapefile_processor.py # NEON shapefile processing
├── models/
│   ├── rgb_models.py         # RGB architectures
│   ├── hsi_models.py         # Hyperspectral architectures
│   ├── lidar_models.py       # LiDAR architectures
│   └── lightning_modules.py  # Training modules
└── processing/               # NEON data processing utilities

scripts/
├── download_neon_all_modalities.py  # Download NEON data
├── process_tiles_to_crowns.py       # Tile processing pipeline
└── test_shapefile_processor.py      # Test shapefile processing
```

## NEON Data Products

- **RGB**: `DP3.30010.001` - High-resolution orthorectified imagery
- **Hyperspectral**: `DP3.30006.002` - 426-band spectrometer reflectance  
- **LiDAR**: `DP3.30015.001` - Canopy Height Model

## Contributing

1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## Authors

Ritesh Chowdhry

## Acknowledgments

- National Ecological Observatory Network (NEON)
- University of Florida Macrosystems project

