"""
NEON Tree Classification DataModule

Lightning DataModule for NEON tree crown data with flexible splitting,
label mapping, and multi-modal support.

Author: Ritesh Chowdhry
"""

import os
import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Any, Callable, Union, Literal
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import DataLoader
import lightning as L

from .dataset import NeonCrownDataset, normalize_rgb, normalize_hsi, normalize_lidar


class NeonCrownDataModule(L.LightningDataModule):
    """
    Lightning DataModule for NEON tree crown classification.

    Handles train/val/test splits, label mapping, and DataLoader creation
    for multi-modal tree crown data.
    """

    def __init__(
        self,
        csv_path: str,
        base_data_dir: str,
        modalities: List[str] = ["rgb"],
        split_method: Literal["random", "site", "year"] = "random",
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        split_seed: int = 42,
        site_filter: Optional[List[str]] = None,
        year_filter: Optional[List[int]] = None,
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = True,
        train_transforms: Optional[Dict[str, Callable]] = None,
        val_transforms: Optional[Dict[str, Callable]] = None,
        test_transforms: Optional[Dict[str, Callable]] = None,
    ):
        """
        Initialize DataModule.

        Args:
            csv_path: Path to CSV with crown metadata
            base_data_dir: Base directory containing data files
            modalities: List of modalities to load ['rgb', 'hsi', 'lidar']
            split_method: How to split data ('random', 'site', 'year')
            train_ratio: Fraction of data for training
            val_ratio: Fraction of data for validation
            test_ratio: Fraction of data for testing
            split_seed: Random seed for reproducible splits
            site_filter: Optional list of sites to include
            year_filter: Optional list of years to include
            batch_size: Batch size for DataLoaders
            num_workers: Number of workers for DataLoaders
            pin_memory: Whether to pin memory in DataLoaders
            train_transforms: Transform functions for training
            val_transforms: Transform functions for validation
            test_transforms: Transform functions for testing
        """
        super().__init__()

        # Store parameters
        self.csv_path = csv_path
        self.base_data_dir = base_data_dir
        self.modalities = modalities
        self.split_method = split_method
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.split_seed = split_seed
        self.site_filter = site_filter
        self.year_filter = year_filter

        # DataLoader parameters
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        # Transforms
        self.train_transforms = train_transforms or self._create_default_transforms()
        self.val_transforms = val_transforms or self._create_default_transforms()
        self.test_transforms = test_transforms or self._create_default_transforms()

        # Validate split ratios
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError("Split ratios must sum to 1.0")

        # Will be set in setup()
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.label_to_idx = None
        self.idx_to_label = None
        self.num_classes = None

    def _create_default_transforms(self) -> Dict[str, Callable]:
        """Create default transform functions."""
        return {
            "rgb": lambda x: normalize_rgb(x, method="0_1"),
            "hsi": lambda x: normalize_hsi(x, method="per_pixel"),
            "lidar": lambda x: normalize_lidar(x, method="height"),
        }

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Setup datasets for each stage.

        Args:
            stage: 'fit', 'validate', 'test', or 'predict'
        """
        if stage is None or stage in ["fit", "validate"]:
            # Create full dataset to analyze splits
            full_dataset = NeonCrownDataset(
                csv_path=self.csv_path,
                base_data_dir=self.base_data_dir,
                modalities=self.modalities,
                has_labels=True,
                site_filter=self.site_filter,
                year_filter=self.year_filter,
                transforms=None,  # We'll add transforms per split
            )

            # Create label mapping
            self._create_label_mapping(full_dataset)

            # Split the data
            train_indices, val_indices, test_indices = self._split_data(full_dataset)

            # Create train dataset
            train_data = full_dataset.data.iloc[train_indices].reset_index(drop=True)
            self.train_dataset = self._create_split_dataset(
                train_data, self.train_transforms
            )

            # Create validation dataset
            val_data = full_dataset.data.iloc[val_indices].reset_index(drop=True)
            self.val_dataset = self._create_split_dataset(val_data, self.val_transforms)

            # Create test dataset (always create for completeness)
            test_data = full_dataset.data.iloc[test_indices].reset_index(drop=True)
            self.test_dataset = self._create_split_dataset(
                test_data, self.test_transforms
            )

            print(f"DataModule setup complete:")
            print(f"  Train samples: {len(self.train_dataset)}")
            print(f"  Val samples: {len(self.val_dataset)}")
            print(f"  Test samples: {len(self.test_dataset)}")
            print(f"  Num classes: {self.num_classes}")

    def _create_label_mapping(self, dataset: NeonCrownDataset) -> None:
        """Create mapping between string labels and integer indices."""
        species_list = dataset.get_species_list()
        self.label_to_idx = {species: idx for idx, species in enumerate(species_list)}
        self.idx_to_label = {idx: species for species, idx in self.label_to_idx.items()}
        self.num_classes = len(species_list)

    def _split_data(self, dataset: NeonCrownDataset) -> tuple:
        """Split data into train/val/test indices."""
        np.random.seed(self.split_seed)

        if self.split_method == "random":
            return self._split_random(dataset)
        elif self.split_method == "site":
            return self._split_by_site(dataset)
        elif self.split_method == "year":
            return self._split_by_year(dataset)
        else:
            raise ValueError(f"Unknown split method: {self.split_method}")

    def _split_random(self, dataset: NeonCrownDataset) -> tuple:
        """Random split of all samples."""
        n_samples = len(dataset)
        indices = np.arange(n_samples)

        # First split: train vs (val + test)
        train_indices, temp_indices = train_test_split(
            indices,
            train_size=self.train_ratio,
            random_state=self.split_seed,
            stratify=(
                dataset.data["species"] if len(dataset.get_species_list()) > 1 else None
            ),
        )

        # Second split: val vs test
        val_size = self.val_ratio / (self.val_ratio + self.test_ratio)
        val_indices, test_indices = train_test_split(
            temp_indices,
            train_size=val_size,
            random_state=self.split_seed + 1,
            stratify=(
                dataset.data.iloc[temp_indices]["species"]
                if len(dataset.get_species_list()) > 1
                else None
            ),
        )

        return train_indices, val_indices, test_indices

    def _split_by_site(self, dataset: NeonCrownDataset) -> tuple:
        """Split by site (all samples from a site go to same split)."""
        sites = dataset.get_sites()
        n_sites = len(sites)

        # Split sites
        n_train_sites = int(n_sites * self.train_ratio)
        n_val_sites = int(n_sites * self.val_ratio)

        # Shuffle sites
        sites_shuffled = np.array(sites)
        np.random.shuffle(sites_shuffled)

        train_sites = sites_shuffled[:n_train_sites]
        val_sites = sites_shuffled[n_train_sites : n_train_sites + n_val_sites]
        test_sites = sites_shuffled[n_train_sites + n_val_sites :]

        # Get indices for each split
        train_indices = dataset.data[
            dataset.data["site"].isin(train_sites)
        ].index.tolist()
        val_indices = dataset.data[dataset.data["site"].isin(val_sites)].index.tolist()
        test_indices = dataset.data[
            dataset.data["site"].isin(test_sites)
        ].index.tolist()

        return train_indices, val_indices, test_indices

    def _split_by_year(self, dataset: NeonCrownDataset) -> tuple:
        """Split by year (all samples from a year go to same split)."""
        years = dataset.get_years()
        n_years = len(years)

        # Split years
        n_train_years = int(n_years * self.train_ratio)
        n_val_years = int(n_years * self.val_ratio)

        # Shuffle years
        years_shuffled = np.array(years)
        np.random.shuffle(years_shuffled)

        train_years = years_shuffled[:n_train_years]
        val_years = years_shuffled[n_train_years : n_train_years + n_val_years]
        test_years = years_shuffled[n_train_years + n_val_years :]

        # Get indices for each split
        train_indices = dataset.data[
            dataset.data["year"].isin(train_years)
        ].index.tolist()
        val_indices = dataset.data[dataset.data["year"].isin(val_years)].index.tolist()
        test_indices = dataset.data[
            dataset.data["year"].isin(test_years)
        ].index.tolist()

        return train_indices, val_indices, test_indices

    def _create_split_dataset(
        self, split_data: pd.DataFrame, transforms: Dict[str, Callable]
    ) -> NeonCrownDataset:
        """Create a dataset for a specific split."""
        # Create a temporary CSV for this split
        temp_csv_path = f"/tmp/temp_split_{id(split_data)}.csv"
        split_data.to_csv(temp_csv_path, index=False)

        # Create dataset with transforms and label mapping
        dataset = NeonCrownDataset(
            csv_path=temp_csv_path,
            base_data_dir=self.base_data_dir,
            modalities=self.modalities,
            has_labels=True,
            transforms=transforms,
            label_to_idx=self.label_to_idx,
        )

        # Clean up temp file
        os.remove(temp_csv_path)

        return dataset

    def train_dataloader(self) -> DataLoader:
        """Create training DataLoader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            # Using default collate - samples now contain only tensors
        )

    def val_dataloader(self) -> DataLoader:
        """Create validation DataLoader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            # Using default collate - samples now contain only tensors
        )

    def test_dataloader(self) -> DataLoader:
        """Create test DataLoader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            # Using default collate - samples now contain only tensors
        )

    def get_class_weights(self) -> torch.Tensor:
        """
        Calculate class weights for imbalanced datasets.

        Returns:
            Tensor of class weights for loss function
        """
        if self.train_dataset is None:
            raise RuntimeError("Must call setup() before getting class weights")

        # Count samples per class in training set
        species_counts = {}
        for i in range(len(self.train_dataset)):
            sample = self.train_dataset[i]
            species = sample["species"]
            species_counts[species] = species_counts.get(species, 0) + 1

        # Calculate weights (inverse frequency)
        total_samples = sum(species_counts.values())
        weights = []
        for species in sorted(species_counts.keys()):
            weight = total_samples / (len(species_counts) * species_counts[species])
            weights.append(weight)

        return torch.tensor(weights, dtype=torch.float32)

    def get_species_mapping(self) -> Dict[str, int]:
        """Get the species name to index mapping."""
        if self.label_to_idx is None:
            raise RuntimeError("Must call setup() before getting species mapping")
        return self.label_to_idx.copy()

    def get_split_info(self) -> Dict[str, Any]:
        """Get information about the current splits."""
        if self.train_dataset is None:
            raise RuntimeError("Must call setup() before getting split info")

        return {
            "split_method": self.split_method,
            "train_samples": len(self.train_dataset),
            "val_samples": len(self.val_dataset),
            "test_samples": len(self.test_dataset),
            "num_classes": self.num_classes,
            "class_names": list(self.label_to_idx.keys()),
            "train_sites": self.train_dataset.get_sites(),
            "val_sites": self.val_dataset.get_sites(),
            "test_sites": self.test_dataset.get_sites(),
        }
