"""
Dataset Combination Utilities for NEON Tree Classification

Functions to combine multiple NEON datasets with metadata flags,
handle path conversions, and ensure data integrity.

Author: Ritesh Chowdhry
"""

import pandas as pd
import numpy as np
import os
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
from collections import Counter


def load_and_inspect_csvs(
    csv_paths: List[str], names: Optional[List[str]] = None
) -> Dict[str, pd.DataFrame]:
    """
    Load multiple CSV files and provide basic inspection.

    Args:
        csv_paths: List of paths to CSV files
        names: Optional names for the datasets (defaults to csv filenames)

    Returns:
        Dictionary mapping dataset names to DataFrames
    """
    if names is None:
        names = [Path(p).stem for p in csv_paths]

    datasets = {}

    print("Loading CSV files...")
    for name, path in zip(names, csv_paths):
        print(f"  Loading {name} from {path}")
        df = pd.read_csv(path)
        datasets[name] = df

        print(f"    Samples: {len(df):,}")
        print(f"    Columns: {list(df.columns)}")
        if "species" in df.columns:
            print(f"    Species: {df['species'].nunique()}")
        print()

    return datasets


def check_crown_id_conflicts(datasets: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """
    Check for crown_id conflicts between datasets.

    Args:
        datasets: Dictionary of dataset name -> DataFrame

    Returns:
        Dictionary with conflict analysis results
    """
    print("Checking crown_id conflicts...")

    all_crown_ids = {}
    conflicts = {}

    # Collect all crown_ids with their source datasets
    for name, df in datasets.items():
        if "crown_id" not in df.columns:
            print(f"  ⚠️  {name}: No 'crown_id' column found")
            continue

        crown_ids = set(df["crown_id"].astype(str))
        print(f"  {name}: {len(crown_ids):,} unique crown_ids")

        # Check for conflicts with previously seen crown_ids
        for crown_id in crown_ids:
            if crown_id in all_crown_ids:
                # Conflict found
                if crown_id not in conflicts:
                    conflicts[crown_id] = [all_crown_ids[crown_id]]
                conflicts[crown_id].append(name)
            else:
                all_crown_ids[crown_id] = name

    # Report results
    if conflicts:
        print(f"\n❌ Found {len(conflicts)} conflicting crown_ids:")
        for crown_id, dataset_names in list(conflicts.items())[:10]:  # Show first 10
            print(f"    {crown_id}: appears in {dataset_names}")
        if len(conflicts) > 10:
            print(f"    ... and {len(conflicts) - 10} more conflicts")
    else:
        print("✅ No crown_id conflicts found!")

    return {
        "total_unique_ids": len(all_crown_ids),
        "conflicts": conflicts,
        "has_conflicts": len(conflicts) > 0,
        "conflict_count": len(conflicts),
    }


def convert_absolute_to_relative_paths(
    df: pd.DataFrame,
    base_dirs: Dict[str, str],
    path_columns: List[str] = ["rgb_path", "hsi_path", "lidar_path"],
) -> pd.DataFrame:
    """
    Convert absolute paths to relative paths in DataFrame.

    Args:
        df: DataFrame with path columns
        base_dirs: Dictionary mapping modality to base directory to remove
                   e.g., {'rgb': '/absolute/path/to/rgb/', 'hsi': '/absolute/path/to/hsi/'}
        path_columns: List of column names containing paths

    Returns:
        DataFrame with relative paths
    """
    df = df.copy()

    print("Converting absolute paths to relative...")

    for col in path_columns:
        if col not in df.columns:
            continue

        # Extract modality from column name (e.g., 'rgb_path' -> 'rgb')
        modality = col.replace("_path", "")

        if modality in base_dirs:
            base_dir = base_dirs[modality]
            print(f"  Converting {col} paths (removing {base_dir})")

            # Convert paths
            original_paths = df[col].astype(str)
            relative_paths = original_paths.str.replace(base_dir, "", regex=False)

            # Clean up any leading slashes
            relative_paths = relative_paths.str.lstrip("/")

            df[col] = relative_paths

            # Report conversion
            converted_count = (original_paths != relative_paths).sum()
            print(f"    Converted {converted_count:,} paths")

    return df


def analyze_species_overlap(datasets: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """
    Analyze species overlap between datasets.

    Args:
        datasets: Dictionary of dataset name -> DataFrame

    Returns:
        Dictionary with overlap analysis results
    """
    print("Analyzing species overlap...")

    species_sets = {}
    species_counts = {}

    # Collect species from each dataset
    for name, df in datasets.items():
        if "species" not in df.columns:
            print(f"  ⚠️  {name}: No 'species' column found")
            continue

        species_set = set(df["species"].dropna().unique())
        species_count = df["species"].value_counts()

        species_sets[name] = species_set
        species_counts[name] = species_count

        print(f"  {name}: {len(species_set)} unique species")

    if len(species_sets) < 2:
        return {"error": "Need at least 2 datasets with species columns"}

    # Calculate overlaps (for simplicity, just handle 2 datasets for now)
    dataset_names = list(species_sets.keys())
    set1, set2 = species_sets[dataset_names[0]], species_sets[dataset_names[1]]

    overlap = set1.intersection(set2)
    only_in_first = set1 - set2
    only_in_second = set2 - set1

    # Calculate coverage percentages
    coverage_first_by_second = len(overlap) / len(set1) * 100 if len(set1) > 0 else 0
    coverage_second_by_first = len(overlap) / len(set2) * 100 if len(set2) > 0 else 0

    print(f"\nSpecies overlap analysis:")
    print(f"  {dataset_names[0]}: {len(set1)} species")
    print(f"  {dataset_names[1]}: {len(set2)} species")
    print(f"  Overlap: {len(overlap)} species")
    print(f"  Only in {dataset_names[0]}: {len(only_in_first)} species")
    print(f"  Only in {dataset_names[1]}: {len(only_in_second)} species")
    print(
        f"  Coverage of {dataset_names[0]} by {dataset_names[1]}: {coverage_first_by_second:.1f}%"
    )
    print(
        f"  Coverage of {dataset_names[1]} by {dataset_names[0]}: {coverage_second_by_first:.1f}%"
    )

    return {
        "dataset_names": dataset_names,
        "species_sets": species_sets,
        "species_counts": species_counts,
        "overlap": overlap,
        "only_in_first": only_in_first,
        "only_in_second": only_in_second,
        "coverage_percentages": {
            f"{dataset_names[0]}_by_{dataset_names[1]}": coverage_first_by_second,
            f"{dataset_names[1]}_by_{dataset_names[0]}": coverage_second_by_first,
        },
    }


def add_metadata_columns(df: pd.DataFrame, metadata: Dict[str, Any]) -> pd.DataFrame:
    """
    Add metadata columns to DataFrame.

    Args:
        df: DataFrame to add columns to
        metadata: Dictionary of column_name -> value pairs

    Returns:
        DataFrame with added metadata columns
    """
    df = df.copy()

    print(f"Adding metadata columns: {list(metadata.keys())}")

    for col_name, value in metadata.items():
        df[col_name] = value

    return df


# PHASE 1: Create unified NPY directory structure
def create_unified_npy_structure(
    datasets: Dict[str, pd.DataFrame],
    source_configs: List[Dict[str, str]],
    target_dir: str,
    modalities: List[str] = ["rgb", "hsi", "lidar"],
    dry_run: bool = True,
) -> Dict[str, Any]:
    """
    Create unified NPY directory structure from multiple source datasets.

    Args:
        datasets: Dictionary of dataset_name -> DataFrame
        source_configs: List of configs like:
                       [{"name": "high_quality", "npy_base": "/path/to/npy/"},
                        {"name": "large", "npy_base": "/path/to/other/npy/"}]
        target_dir: Target directory for unified structure
        modalities: List of modalities to copy
        dry_run: If True, just preview what would happen

    Returns:
        Dictionary with copy results and updated datasets
    """
    from pathlib import Path
    import shutil

    print(
        f"{'[DRY RUN] ' if dry_run else ''}Creating unified NPY structure at {target_dir}"
    )

    # Create target directory structure
    target_path = Path(target_dir)
    if not dry_run:
        target_path.mkdir(parents=True, exist_ok=True)
        for modality in modalities:
            (target_path / "npy" / modality).mkdir(parents=True, exist_ok=True)
        (target_path / "metadata").mkdir(parents=True, exist_ok=True)

    copy_plan = []
    updated_datasets = {}

    for config in source_configs:
        dataset_name = config["name"]
        source_npy_base = config["npy_base"]

        if dataset_name not in datasets:
            print(f"  ⚠️  Dataset {dataset_name} not found in datasets")
            continue

        df = datasets[dataset_name].copy()
        print(f"\n  Processing {dataset_name}: {len(df)} samples")

        # Plan file copies for each sample
        for idx, row in df.iterrows():
            crown_id = row["crown_id"]

            for modality in modalities:
                # Source path (from current structure)
                source_path = Path(source_npy_base) / modality / f"{crown_id}.npy"

                # Target path (unified structure)
                target_file_path = target_path / "npy" / modality / f"{crown_id}.npy"

                # Update DataFrame path to new relative structure
                path_col = f"{modality}_path"
                if path_col in df.columns:
                    df.loc[idx, path_col] = f"npy/{modality}/{crown_id}.npy"

                copy_plan.append(
                    {
                        "source": source_path,
                        "target": target_file_path,
                        "crown_id": crown_id,
                        "modality": modality,
                        "dataset": dataset_name,
                    }
                )

        updated_datasets[dataset_name] = df

    print(f"\n  Copy plan: {len(copy_plan)} files to copy")

    if dry_run:
        print("  [DRY RUN] Preview of first 5 copies:")
        for item in copy_plan[:5]:
            print(f"    {item['source']} → {item['target']}")
        return {
            "copy_plan": copy_plan,
            "updated_datasets": updated_datasets,
            "target_dir": target_dir,
            "dry_run": True,
        }

    # Execute file copies with progress
    print(f"  Copying {len(copy_plan)} files...")
    copied = 0
    failed = 0

    for i, item in enumerate(copy_plan):
        try:
            shutil.copy2(item["source"], item["target"])
            copied += 1
        except Exception as e:
            print(f"    ❌ Failed to copy {item['source']}: {e}")
            failed += 1

        # Progress update every 1000 files
        if (i + 1) % 1000 == 0:
            print(f"    Progress: {i + 1:,}/{len(copy_plan):,} files")

    print(f"  ✅ Copy complete: {copied} succeeded, {failed} failed")

    return {
        "copy_plan": copy_plan,
        "updated_datasets": updated_datasets,
        "target_dir": target_dir,
        "copied": copied,
        "failed": failed,
        "dry_run": False,
    }


def create_combined_csvs(
    updated_datasets: Dict[str, pd.DataFrame],
    target_metadata_dir: str,
    dataset_metadata: Dict[str, Dict[str, Any]],
) -> Dict[str, str]:
    """
    Create all required CSV files from updated datasets.

    Args:
        updated_datasets: Dictionary of dataset_name -> DataFrame with updated paths
        target_metadata_dir: Directory to save CSV files
        dataset_metadata: Metadata to add to each dataset like:
                         {"high_quality": {"hand_annotated": True},
                          "large": {"hand_annotated": False}}

    Returns:
        Dictionary mapping CSV type to file path
    """
    from pathlib import Path

    metadata_path = Path(target_metadata_dir)
    metadata_path.mkdir(parents=True, exist_ok=True)

    csv_files = {}

    # Add metadata columns to each dataset
    datasets_with_metadata = {}
    for name, df in updated_datasets.items():
        df_meta = df.copy()
        if name in dataset_metadata:
            for col, value in dataset_metadata[name].items():
                df_meta[col] = value
        datasets_with_metadata[name] = df_meta

    # Create individual dataset CSVs
    for name, df in datasets_with_metadata.items():
        csv_path = metadata_path / f"{name}_dataset.csv"
        df.to_csv(csv_path, index=False)
        csv_files[f"{name}_csv"] = str(csv_path)
        print(f"  Created {name}_dataset.csv: {len(df)} samples")

    # Create combined CSV
    combined_df = pd.concat(datasets_with_metadata.values(), ignore_index=True)
    combined_csv_path = metadata_path / "combined_dataset.csv"
    combined_df.to_csv(combined_csv_path, index=False)
    csv_files["combined_csv"] = str(combined_csv_path)
    print(f"  Created combined_dataset.csv: {len(combined_df)} samples")

    return csv_files


# PHASE 2: Convert NPY directory to HDF5 (Data Only) + Create ZIP Distribution
def convert_npy_to_hdf5_data_only(
    csv_path: str,
    npy_base_dir: str,
    hdf5_output_path: str,
    modalities: List[str] = ["rgb", "hsi", "lidar"],
    compression: str = "gzip",
    chunk_size: int = 1000,
) -> Dict[str, Any]:
    """
    Convert unified NPY directory structure to HDF5 format (data only, no metadata).
    Keeps CSVs separate for user-friendliness.

    Args:
        csv_path: Path to main CSV with crown_ids and paths (combined_dataset.csv)
        npy_base_dir: Base directory containing npy/rgb/, npy/hsi/, etc.
        hdf5_output_path: Output HDF5 file path
        modalities: List of modalities to convert
        compression: HDF5 compression ('gzip', 'lzf', None)
        chunk_size: Progress reporting interval

    Returns:
        Conversion results dictionary
    """
    try:
        import h5py
    except ImportError:
        raise ImportError(
            "h5py is required for HDF5 conversion. Install with: pip install h5py"
        )

    from pathlib import Path

    print(f"Converting NPY directory to HDF5 (data only): {hdf5_output_path}")

    # Load main metadata
    df = pd.read_csv(csv_path)
    print(f"  Found {len(df)} samples to convert")

    # Create HDF5 file
    with h5py.File(hdf5_output_path, "w") as h5f:

        # Create data groups for each modality
        for modality in modalities:
            h5f.create_group(modality)
            print(f"  Created data group: {modality}")

        # Convert each sample's NPY data
        converted = 0
        failed = 0

        for idx, row in df.iterrows():
            crown_id = str(row["crown_id"])

            try:
                # Process all modalities for this crown_id
                sample_success = True
                for modality in modalities:
                    # Load NPY file
                    npy_path = Path(npy_base_dir) / f"npy/{modality}/{crown_id}.npy"

                    if npy_path.exists():
                        data = np.load(npy_path)

                        # Store in HDF5 with compression
                        h5f[modality].create_dataset(
                            crown_id,
                            data=data,
                            compression=compression,
                            shuffle=True,  # Often improves compression
                        )
                    else:
                        print(f"    ⚠️  Missing {modality} file: {npy_path}")
                        sample_success = False

                if sample_success:
                    converted += 1

                # Progress reporting (only once per sample)
                if converted > 0 and converted % chunk_size == 0:
                    print(f"    Progress: {converted:,}/{len(df):,} samples converted")

            except Exception as e:
                print(f"    ❌ Failed to convert {crown_id}: {e}")
                failed += 1

    print(f"  ✅ HDF5 conversion complete: {converted} succeeded, {failed} failed")
    print(f"  📁 HDF5 data file created: {hdf5_output_path}")

    return {
        "hdf5_path": hdf5_output_path,
        "converted": converted,
        "failed": failed,
        "modalities": modalities,
    }


def create_dataset_zip(
    base_dir: str,
    zip_output_path: str,
    hdf5_file: Optional[str] = None,
    include_npy: bool = False,
) -> Dict[str, Any]:
    """
    Create a ZIP distribution of the dataset.
    
    Args:
        base_dir: Base directory containing metadata/ and npy/ or .h5 file
        zip_output_path: Output ZIP file path
        hdf5_file: Optional HDF5 file to include instead of NPY directory
        include_npy: Whether to include NPY files (if no HDF5 file provided)
    
    Returns:
        Dictionary with compression results
    """
    import zipfile
    from pathlib import Path
    
    print(f"Creating dataset ZIP: {zip_output_path}")
    
    base_path = Path(base_dir)
    files_added = 0
    total_size = 0
    
    with zipfile.ZipFile(zip_output_path, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as zipf:
        
        # Add metadata CSVs (always included)
        metadata_dir = base_path / "metadata"
        if metadata_dir.exists():
            for csv_file in metadata_dir.glob("*.csv"):
                arcname = f"metadata/{csv_file.name}"
                zipf.write(csv_file, arcname)
                files_added += 1
                total_size += csv_file.stat().st_size
                print(f"  Added: {arcname}")
        
        # Add HDF5 file if provided
        if hdf5_file and Path(hdf5_file).exists():
            hdf5_path = Path(hdf5_file)
            arcname = hdf5_path.name
            zipf.write(hdf5_path, arcname)
            files_added += 1
            total_size += hdf5_path.stat().st_size
            print(f"  Added: {arcname}")
        
        # Add NPY files if requested and no HDF5
        elif include_npy:
            npy_dir = base_path / "npy"
            if npy_dir.exists():
                for modality_dir in npy_dir.iterdir():
                    if modality_dir.is_dir():
                        for npy_file in modality_dir.glob("*.npy"):
                            arcname = f"npy/{modality_dir.name}/{npy_file.name}"
                            zipf.write(npy_file, arcname)
                            files_added += 1
                            total_size += npy_file.stat().st_size
                        print(f"  Added NPY files from: npy/{modality_dir.name}/ ({len(list(modality_dir.glob('*.npy')))} files)")
        
        # Add README
        readme_content = f"""# NEON Tree Classification Dataset

This dataset contains tree crown data from NEON sites with multiple modalities:
- RGB imagery
- Hyperspectral imagery (HSI)  
- LiDAR data

## Files:

### Metadata (CSV files):
- `metadata/combined_dataset.csv`: All samples with hand_annotated flag
- `metadata/high_quality_dataset.csv`: High-quality hand-annotated subset
- `metadata/large_dataset.csv`: Larger dataset subset

### Data:
{"- `neon_dataset.h5`: HDF5 file containing all NPY data organized by modality" if hdf5_file else "- `npy/[rgb|hsi|lidar]/`: NPY files for each sample by crown_id"}

## Usage:

```python
import pandas as pd
{"import h5py" if hdf5_file else "import numpy as np"}

# Load metadata
df = pd.read_csv("metadata/combined_dataset.csv")

{"# Load data from HDF5" if hdf5_file else "# Load data from NPY files"}
{"with h5py.File('neon_dataset.h5', 'r') as f:" if hdf5_file else "# Example: load RGB data for first sample"}
{"    crown_id = str(df.iloc[0]['crown_id'])" if hdf5_file else "crown_id = df.iloc[0]['crown_id']"}
{"    rgb_data = f['rgb'][crown_id][:]" if hdf5_file else "rgb_data = np.load(f'npy/rgb/{crown_id}.npy')"}
```

Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        # Add README to ZIP
        zipf.writestr("README.md", readme_content)
        files_added += 1
        
        # Also save README locally for reference
        local_readme_path = base_path / "README.md"
        with open(local_readme_path, 'w') as f:
            f.write(readme_content)
        print(f"  📝 README saved locally: {local_readme_path}")
    
    zip_size = Path(zip_output_path).stat().st_size
    compression_ratio = (1 - zip_size / total_size) * 100 if total_size > 0 else 0
    
    print(f"  ✅ ZIP created successfully!")
    print(f"  📁 Files added: {files_added}")
    print(f"  � Original size: {total_size / 1024**3:.2f} GB")
    print(f"  📦 ZIP size: {zip_size / 1024**3:.2f} GB")
    print(f"  � Compression ratio: {compression_ratio:.1f}%")
    
    return {
        "zip_path": zip_output_path,
        "files_added": files_added,
        "original_size_gb": total_size / 1024**3,
        "zip_size_gb": zip_size / 1024**3,
        "compression_ratio": compression_ratio,
    }


# Example usage functions for notebook
def quick_analysis(small_csv: str, large_csv: str) -> Tuple[Dict, Dict, Dict]:
    """
    Quick analysis of the two datasets - perfect for notebook cells.

    Args:
        small_csv: Path to smaller (high-quality) dataset CSV
        large_csv: Path to larger dataset CSV
    """
    # Load datasets
    datasets = load_and_inspect_csvs([small_csv, large_csv], ["high_quality", "large"])

    # Check conflicts
    conflict_info = check_crown_id_conflicts(datasets)

    # Analyze species overlap
    overlap_info = analyze_species_overlap(datasets)

    return datasets, conflict_info, overlap_info


def complete_dataset_creation_pipeline(
    small_csv: str,
    large_csv: str,
    small_npy_base: str,
    large_npy_base: str,
    target_dir: str,
    create_hdf5: bool = True,
) -> Dict[str, Any]:
    """
    Complete pipeline to create unified dataset structure.

    This is the main function you'd run in your notebook after analysis.
    """
    print("=== NEON Dataset Creation Pipeline ===")

    # Step 1: Load and analyze datasets
    print("\n1. Loading datasets...")
    datasets, conflict_info, overlap_info = quick_analysis(small_csv, large_csv)

    if conflict_info["has_conflicts"]:
        raise ValueError("Crown ID conflicts found! Please resolve before proceeding.")

    # Step 2: Create unified NPY structure
    print("\n2. Creating unified NPY structure...")
    source_configs = [
        {"name": "high_quality", "npy_base": small_npy_base},
        {"name": "large", "npy_base": large_npy_base},
    ]

    # First do dry run
    dry_result = create_unified_npy_structure(
        datasets, source_configs, target_dir, dry_run=True
    )
    print(f"   Dry run complete. Ready to copy {len(dry_result['copy_plan'])} files.")

    # Ask for confirmation in notebook
    print("\n   ⚠️  Ready to copy files. Run next cell to proceed...")

    return {
        "datasets": datasets,
        "source_configs": source_configs,
        "target_dir": target_dir,
        "dry_result": dry_result,
        "conflict_info": conflict_info,
        "overlap_info": overlap_info,
    }


if __name__ == "__main__":
    # Example usage with your dataset paths
    small_file = "/blue/azare/riteshchowdhry/Macrosystems/Data_files/hand_annotated_neon/curated_tiles_20250822/cropped_crowns_npy/filtered_training_data.csv"
    large_file = "/blue/azare/riteshchowdhry/Macrosystems/Data_files/hand_annotated_neon/curated_vst_tiles_20250825/cropped_crowns_npy/filtered_training_data_42k.csv"

    datasets, conflict_info, overlap_info = quick_analysis(small_file, large_file)
