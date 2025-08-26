#!/usr/bin/env python3
"""
Simple script to update README.md with current dataset statistics using only pandas.

Usage:
    python scripts/update_readme.py --csv /path/to/data.csv --readme /path/to/README.md
"""

import pandas as pd
import datetime
import argparse


def get_dataset_stats(csv_path):
    """Get dataset statistics directly from CSV without importing the package."""
    print(f"Loading dataset from {csv_path}...")
    df = pd.read_csv(csv_path)

    stats = {
        "total_individuals": len(df),
        "species_count": df["species"].nunique(),
        "sites_count": df["site"].nunique(),
        "years": sorted(df["year"].unique()),
    }

    # Species distribution
    if "species_name" in df.columns:
        species_name_counts = df["species_name"].value_counts()
        stats["top_species_names"] = species_name_counts.head(10).to_dict()
    else:
        species_counts = df["species"].value_counts()
        stats["top_species"] = species_counts.head(10).to_dict()

    # Site distribution
    site_counts = df["site"].value_counts()
    stats["site_distribution"] = site_counts.to_dict()

    return stats


def update_readme(csv_path, readme_path):
    """Update README.md with current dataset statistics."""
    stats = get_dataset_stats(csv_path)

    print("Generating README content...")

    # Generate the new README content
    readme_content = f"""# NEON Multi-Modal Tree Species Dataset

Hyperspectral, RGB and LiDAR airborne data for **{stats['species_count']} tree species** representing **{stats['total_individuals']:,} individual trees** across **{stats['sites_count']} NEON sites** in North America.

## Dataset Overview

- **{stats['total_individuals']:,}** individual tree crowns
- **{stats['species_count']}** unique species  
- **{stats['sites_count']}** NEON sites across North America
- **{stats['years'][0]}-{stats['years'][-1]}** ({len(stats['years'])} years of data)
- **3 modalities:** RGB, Hyperspectral (426 bands), LiDAR CHM

## Quick Start

```python
# Load and explore the dataset
from neon_tree_classification.core.dataset import NeonCrownDataset

# Simple loading
dataset = NeonCrownDataset.load()
dataset.summary()  # Print dataset overview

# Filter for specific species or sites  
conifers = dataset.filter(species=['PSMEM', 'TSHE'])
west_coast = conifers.filter(sites=['ABBY', 'HARV'])

# Get dataset statistics
stats = dataset.get_dataset_stats()
```

## Visualization Examples

The package includes comprehensive visualization tools for all three modalities:

| RGB Image | HSI Pseudo RGB | HSI PCA Decomposition |
|-----------|----------------|----------------------|
| ![RGB](sample_plots/sample_rgb.png) | ![HSI](sample_plots/sample_hsi.png) | ![HSI PCA](sample_plots/sample_hsi_pca.png) |

| HSI Spectral Signatures | LiDAR Canopy Height Model |
|-------------------------|---------------------------|
| ![Spectra](sample_plots/sample_spectra.png) | ![LiDAR](sample_plots/sample_lidar.png) |

```python
# Visualization functions for tree crown data
from neon_tree_classification.core.visualization import (
    plot_rgb, plot_hsi, plot_hsi_pca, plot_hsi_spectra, plot_lidar
)

# RGB visualization
plot_rgb('path/to/crown_rgb.tif')        # True color RGB image

# Hyperspectral visualization options  
plot_hsi('path/to/crown_hsi.tif')        # Pseudo RGB (bands ~660nm, ~550nm, ~450nm)
plot_hsi_pca('path/to/crown_hsi.tif')    # PCA decomposition to 3 components
plot_hsi_spectra('path/to/crown_hsi.tif') # Spectral signatures of pixels

# LiDAR visualization
plot_lidar('path/to/crown_chm.tif')      # Canopy height model with colorbar
```

### Quick Visualization with Dataset

```python
# Easy visualization with dataset integration
from neon_tree_classification.core.dataset import NeonCrownDataset

dataset = NeonCrownDataset.load()
sample = dataset.data.iloc[0]  # Get first sample

# Visualize all modalities for this tree crown
plot_rgb(sample['rgb_path'])
plot_hsi(sample['hsi_path'])  
plot_lidar(sample['lidar_path'])
```

## Top Species

The dataset includes {stats['species_count']} tree species. Here are the most common:

| Rank | Species | Count | Percentage |
|------|---------|-------|------------|"""

    # Add top species table
    if "top_species_names" in stats:
        for i, (species, count) in enumerate(stats["top_species_names"].items(), 1):
            percentage = count / stats["total_individuals"] * 100
            readme_content += f"\n| {i} | {species} | {count:,} | {percentage:.1f}% |"
    else:
        for i, (species, count) in enumerate(stats["top_species"].items(), 1):
            percentage = count / stats["total_individuals"] * 100
            readme_content += f"\n| {i} | {species} | {count:,} | {percentage:.1f}% |"

    readme_content += f"""

## Geographic Distribution

Data collected from **{stats['sites_count']} NEON sites** across North America:

"""

    # Add top sites
    site_items = list(stats["site_distribution"].items())
    site_items.sort(key=lambda x: x[1], reverse=True)

    for i, (site, count) in enumerate(site_items[:10], 1):
        percentage = count / stats["total_individuals"] * 100
        readme_content += f"**{i}.** {site}: {count:,} samples ({percentage:.1f}%)  \n"

    # Add installation and usage sections
    readme_content += f"""
## Installation

### Basic Installation
```bash
# Clone the repository
git clone https://github.com/Ritesh313/NeonTreeClassification.git
cd NeonTreeClassification

# Install core dependencies
pip install .
```

### Optional Dependencies
```bash
# For development (tests, formatting, notebooks)
pip install .[dev]

# For data processing (geospatial tools)
pip install .[processing]

# For experiment logging
pip install .[logging]

# Install all optional dependencies
pip install .[dev,processing,logging]
```

## Repository Structure

```
NeonTreeClassification/
├── neon_tree_classification/          # Main package
│   ├── core/                         # Core functionality (dataset, visualization)
│   │   ├── dataset.py               # Enhanced dataset with filtering & stats
│   │   ├── datamodule.py            # PyTorch Lightning data module  
│   │   └── visualization.py         # All visualization functions
│   └── models/                      # ML architectures & Lightning modules
├── examples/                         # Training and comparison examples
│   ├── train.py                     # Main training script
│   └── compare_modalities.py        # Multi-modal comparison
├── notebooks/                        # Interactive exploration
│   └── visualization.ipynb          # Visualization demo notebook
├── processing/                       # Advanced data processing tools
├── scripts/                          # Automation utilities
├── sample_plots/                     # Generated sample images
└── training_data_clean.csv          # Main dataset file
```

## Interactive Notebook

Explore the dataset and visualization functions interactively:

```bash
# Start Jupyter and open the visualization notebook
jupyter notebook notebooks/visualization.ipynb
```

The notebook includes examples of:
- Loading and filtering the dataset
- RGB, HSI, and LiDAR visualizations  
- Interactive exploration of tree crown data

## Advanced Usage

### Multi-modal Training

```python
# Train models on different modalities
from neon_tree_classification.core.datamodule import NeonCrownDataModule
from neon_tree_classification.models.lightning_modules import RGBClassifier

# Setup data
datamodule = NeonCrownDataModule(
    csv_path="training_data_clean.csv",
    modalities=["rgb", "hsi", "lidar"],
    batch_size=32
)

# Train RGB model
classifier = RGBClassifier(num_classes={stats['species_count']})

import lightning as L
trainer = L.Trainer(max_epochs=50)
trainer.fit(classifier, datamodule)
```

### Data Processing

The package includes tools for processing NEON data, but most users will work with the pre-processed dataset.

```python
# For advanced users: process raw NEON data
from processing.shapefile_processor import ShapefileProcessor
processor = ShapefileProcessor()
sites_df, summary = processor.process_shapefiles(destination_dir)
```

## Dataset Details

### NEON Data Products
- **RGB**: `DP3.30010.001` - High-resolution orthorectified imagery
- **Hyperspectral**: `DP3.30006.002` - 426-band spectrometer reflectance  
- **LiDAR**: `DP3.30015.001` - Canopy Height Model

### Data Structure
```
training_data_clean.csv - Main dataset file
├── crown_id          - Unique identifier for each tree crown
├── site              - NEON site code
├── year              - Data collection year  
├── species           - Species code
├── species_name      - Full species name
├── height            - Tree height (meters)
├── rgb_path          - Path to RGB image
├── hsi_path          - Path to hyperspectral image
├── lidar_path        - Path to LiDAR CHM
└── [other metadata]  - Additional tree measurements
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## Authors

Ritesh Chowdhry

## Acknowledgments

- National Ecological Observatory Network (NEON)
- This dataset details were generated on {datetime.datetime.now().strftime("%Y-%m-%d")}

"""

    # Write the README to the specified path
    with open(readme_path, "w") as f:
        f.write(readme_content)

    print(f"✅ README.md updated successfully at: {readme_path}")
    print(f"   - {stats['total_individuals']:,} individuals")
    print(f"   - {stats['species_count']} species")
    print(f"   - {stats['sites_count']} sites")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Update README.md with dataset statistics"
    )
    parser.add_argument("--csv", required=True, help="Path to the dataset CSV file")
    parser.add_argument(
        "--readme", required=True, help="Path to the README.md file to update"
    )

    args = parser.parse_args()
    update_readme(args.csv, args.readme)
