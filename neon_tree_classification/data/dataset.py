"""
NEON Tree Classification Dataset

Complete dataloader with loaders, transforms, and dataset class.
Supports RGB, HSI, and LiDAR modalities with flexible normalization.

Author: Ritesh Chowdhry
"""

import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import rasterio
from typing import List, Optional, Dict, Any, Callable, Union


# ============================================================================
# NORMALIZATION FUNCTIONS
# ============================================================================


def normalize_rgb(rgb_tensor: torch.Tensor, method: str = "0_1") -> torch.Tensor:
    """
    Normalize RGB data.

    Args:
        rgb_tensor: RGB tensor [C, H, W]
        method: '0_1' for 0-1 normalization, 'imagenet' for ImageNet stats

    Returns:
        Normalized RGB tensor
    """
    if method == "0_1":
        # Simple 0-1 normalization
        return rgb_tensor.float() / 255.0
    elif method == "imagenet":
        # ImageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        rgb_normalized = rgb_tensor.float() / 255.0
        return (rgb_normalized - mean) / std
    else:
        raise ValueError(f"Unknown RGB normalization method: {method}")


def normalize_hsi(hsi_tensor: torch.Tensor, method: str = "per_pixel") -> torch.Tensor:
    """
    Normalize hyperspectral data.

    Args:
        hsi_tensor: HSI tensor [Bands, H, W]
        method: 'per_pixel', 'per_band', 'global', or 'none'

    Returns:
        Normalized HSI tensor
    """
    if method == "none":
        return hsi_tensor

    if method == "per_pixel":
        # Normalize each pixel's spectrum to unit norm
        # Reshape to [Bands, H*W]
        bands, height, width = hsi_tensor.shape
        reshaped = hsi_tensor.view(bands, -1)

        # Calculate L2 norm for each pixel
        norms = torch.norm(reshaped, dim=0, keepdim=True)
        norms = torch.clamp(norms, min=1e-8)  # Avoid division by zero

        # Normalize and reshape back
        normalized = reshaped / norms
        return normalized.view(bands, height, width)

    elif method == "per_band":
        # Normalize each band independently (z-score)
        mean = hsi_tensor.mean(dim=(1, 2), keepdim=True)
        std = hsi_tensor.std(dim=(1, 2), keepdim=True)
        std = torch.clamp(std, min=1e-8)
        return (hsi_tensor - mean) / std

    elif method == "global":
        # Global z-score normalization
        mean = hsi_tensor.mean()
        std = hsi_tensor.std()
        std = torch.clamp(std, min=1e-8)
        return (hsi_tensor - mean) / std

    else:
        raise ValueError(f"Unknown HSI normalization method: {method}")


def normalize_lidar(lidar_tensor: torch.Tensor, method: str = "height") -> torch.Tensor:
    """
    Normalize LiDAR data.

    Args:
        lidar_tensor: LiDAR tensor [1, H, W]
        method: 'height', 'zscore', or 'none'

    Returns:
        Normalized LiDAR tensor
    """
    if method == "none":
        return lidar_tensor

    if method == "height":
        # Simple height normalization (0-1 based on max height)
        valid_data = lidar_tensor[lidar_tensor > 0]  # Exclude NoData/ground
        if len(valid_data) > 0:
            max_height = valid_data.max()
            return torch.clamp(lidar_tensor / max_height, 0, 1)
        else:
            return lidar_tensor

    elif method == "zscore":
        # Z-score normalization
        valid_data = lidar_tensor[lidar_tensor > 0]
        if len(valid_data) > 0:
            mean = valid_data.mean()
            std = valid_data.std()
            std = torch.clamp(std, min=1e-8)
            # Only normalize valid data
            normalized = lidar_tensor.clone()
            valid_mask = lidar_tensor > 0
            normalized[valid_mask] = (lidar_tensor[valid_mask] - mean) / std
            return normalized
        else:
            return lidar_tensor

    else:
        raise ValueError(f"Unknown LiDAR normalization method: {method}")


def fill_nodata_lidar(
    lidar_tensor: torch.Tensor, fill_value: float = 0.0
) -> torch.Tensor:
    """
    Fill NoData values in LiDAR data.

    Args:
        lidar_tensor: LiDAR tensor [1, H, W]
        fill_value: Value to fill NoData with

    Returns:
        LiDAR tensor with NoData filled
    """
    # Common NoData values in LiDAR: -9999, NaN, very large negative numbers
    filled = lidar_tensor.clone()

    # Fill NaN values
    filled[torch.isnan(filled)] = fill_value

    # Fill very negative values (likely NoData)
    filled[filled < -1000] = fill_value

    return filled


# ============================================================================
# MODALITY LOADERS
# ============================================================================


class RGBLoader:
    """Load RGB TIF files and convert to torch tensors."""

    def __call__(self, file_path: str) -> torch.Tensor:
        """
        Load RGB file.

        Args:
            file_path: Path to RGB TIF file

        Returns:
            RGB tensor [C, H, W] as uint8
        """
        with rasterio.open(file_path) as src:
            # Read all bands: [C, H, W]
            rgb_data = src.read()  # Shape: [bands, height, width]

            # Convert to torch tensor
            rgb_tensor = torch.from_numpy(rgb_data)

            # Ensure we have 3 channels
            if rgb_tensor.shape[0] != 3:
                raise ValueError(f"Expected 3 RGB channels, got {rgb_tensor.shape[0]}")

            return rgb_tensor


class HSILoader:
    """Load HSI TIF files and convert to torch tensors."""

    def __call__(self, file_path: str) -> torch.Tensor:
        """
        Load HSI file.

        Args:
            file_path: Path to HSI TIF file

        Returns:
            HSI tensor [Bands, H, W] as float32
        """
        with rasterio.open(file_path) as src:
            # Read all bands: [Bands, H, W]
            hsi_data = src.read().astype(np.float32)

            # Convert to torch tensor
            hsi_tensor = torch.from_numpy(hsi_data)

            return hsi_tensor


class LiDARLoader:
    """Load LiDAR TIF files and convert to torch tensors."""

    def __init__(self, fill_nodata: bool = True):
        """
        Initialize LiDAR loader.

        Args:
            fill_nodata: Whether to fill NoData values automatically
        """
        self.fill_nodata = fill_nodata

    def __call__(self, file_path: str) -> torch.Tensor:
        """
        Load LiDAR file.

        Args:
            file_path: Path to LiDAR TIF file

        Returns:
            LiDAR tensor [1, H, W] as float32
        """
        with rasterio.open(file_path) as src:
            # Read data (should be single band CHM)
            lidar_data = src.read().astype(np.float32)

            # Convert to torch tensor
            lidar_tensor = torch.from_numpy(lidar_data)

            # Ensure single channel: [1, H, W]
            if lidar_tensor.shape[0] != 1:
                if lidar_tensor.ndim == 2:
                    lidar_tensor = lidar_tensor.unsqueeze(0)
                else:
                    # Take first band if multiple
                    lidar_tensor = lidar_tensor[0:1]

            # Fill NoData if requested
            if self.fill_nodata:
                lidar_tensor = fill_nodata_lidar(lidar_tensor)

            return lidar_tensor


# ============================================================================
# MAIN DATASET CLASS
# ============================================================================


class NeonCrownDataset(Dataset):
    """
    Dataset for NEON tree crown data supporting multiple modalities.

    Expected CSV format:
    crown_id,site,year,plot,species,easting,northing,rgb_path,hsi_path,lidar_path
    """

    def __init__(
        self,
        csv_path: str,
        base_data_dir: str,
        modalities: List[str] = ["rgb"],
        has_labels: bool = True,
        site_filter: Optional[List[str]] = None,
        year_filter: Optional[List[int]] = None,
        transforms: Optional[Dict[str, Callable]] = None,
        label_to_idx: Optional[Dict[str, int]] = None,
    ):
        """
        Initialize dataset.

        Args:
            csv_path: Path to CSV with crown metadata
            base_data_dir: Base directory containing data files
            modalities: List of modalities to load ['rgb', 'hsi', 'lidar']
            has_labels: Whether dataset includes species labels
            site_filter: Optional list of sites to include
            year_filter: Optional list of years to include
            transforms: Optional dict of transform functions per modality
            label_to_idx: Optional mapping from species names to integer indices
        """
        self.csv_path = csv_path
        self.base_data_dir = base_data_dir
        self.modalities = modalities
        self.has_labels = has_labels
        self.transforms = transforms or {}
        self.label_to_idx = label_to_idx

        # Validate modalities
        valid_modalities = {"rgb", "hsi", "lidar"}
        for mod in modalities:
            if mod not in valid_modalities:
                raise ValueError(f"Unknown modality: {mod}. Valid: {valid_modalities}")

        # Initialize loaders
        self.loaders = {"rgb": RGBLoader(), "hsi": HSILoader(), "lidar": LiDARLoader()}

        # Load and filter data
        self.data = self._load_and_filter_data(site_filter, year_filter)

        print(f"NeonCrownDataset initialized:")
        print(f"  Modalities: {modalities}")
        print(f"  Samples: {len(self.data)}")
        print(f"  Has labels: {has_labels}")

    def _load_and_filter_data(
        self, site_filter: Optional[List[str]], year_filter: Optional[List[int]]
    ) -> pd.DataFrame:
        """Load CSV and apply filters."""
        # Load CSV
        df = pd.read_csv(self.csv_path)

        # Apply site filter
        if site_filter is not None:
            df = df[df["site"].isin(site_filter)]

        # Apply year filter
        if year_filter is not None:
            df = df[df["year"].isin(year_filter)]

        # Filter for complete samples (all requested modalities available)
        df = self._filter_complete_samples(df)

        # Reset index
        df = df.reset_index(drop=True)

        return df

    def _filter_complete_samples(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter samples that have all requested modalities available."""
        modality_columns = {"rgb": "rgb_path", "hsi": "hsi_path", "lidar": "lidar_path"}

        # Check each requested modality
        for modality in self.modalities:
            col = modality_columns[modality]
            # Keep only rows where path is not empty/NaN
            df = df[df[col].notna() & (df[col] != "")]

        return df

    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single sample.

        Args:
            idx: Sample index

        Returns:
            Dictionary containing:
            - modality data: torch.Tensor for each requested modality
            - species_idx: Integer label (if has_labels=True and label_to_idx provided)
        """
        row = self.data.iloc[idx]

        # Prepare output dictionary (no crown_id in batch)
        sample = {}

        # Load each modality
        for modality in self.modalities:
            # Get file path
            path_col = f"{modality}_path"
            rel_path = row[path_col]
            file_path = os.path.join(self.base_data_dir, rel_path)

            # Check file exists
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")

            # Load data
            data = self.loaders[modality](file_path)

            # Apply transform if provided
            if modality in self.transforms:
                data = self.transforms[modality](data)

            sample[modality] = data

        # Add species label if available
        if self.has_labels and "species" in row and pd.notna(row["species"]):
            # Only add integer index if mapping provided (for training)
            if self.label_to_idx is not None:
                sample["species_idx"] = self.label_to_idx[row["species"]]

        return sample

    def get_species_list(self) -> List[str]:
        """Get list of unique species in dataset."""
        if not self.has_labels:
            return []

        species_col = self.data["species"]
        unique_species = species_col.dropna().unique().tolist()
        return sorted(unique_species)

    def get_sites(self) -> List[str]:
        """Get list of sites in dataset."""
        return sorted(self.data["site"].unique().tolist())

    def get_years(self) -> List[int]:
        """Get list of years in dataset."""
        return sorted(self.data["year"].unique().tolist())

    def get_crown_id(self, idx: int) -> str:
        """Get crown_id for a specific dataset index."""
        return self.data.iloc[idx]["crown_id"]

    def get_sample_info(self, idx: int) -> Dict[str, Any]:
        """Get full metadata for a specific dataset index."""
        row = self.data.iloc[idx]
        return {
            "crown_id": row["crown_id"],
            "site": row["site"],
            "year": row["year"],
            "species": (
                row["species"]
                if "species" in row and pd.notna(row["species"])
                else None
            ),
            "coordinates": (row["easting"], row["northing"]),
        }
