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
import torch.nn.functional as F
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

    def __init__(
        self,
        target_size: Optional[tuple] = (128, 128),
        spatial_strategy: str = "resize",
        padding_value: float = 0.0,
        crop_if_larger: bool = True,
    ):
        """
        Initialize RGB loader.

        Args:
            target_size: Target size (H, W) to standardize images to. If None, no standardization.
            spatial_strategy: 'resize' for interpolation or 'pad' for padding to target size
            padding_value: Value to use for padding when spatial_strategy='pad'
            crop_if_larger: Whether to crop images larger than target_size when padding
        """
        self.target_size = target_size
        self.spatial_strategy = spatial_strategy
        self.padding_value = padding_value
        self.crop_if_larger = crop_if_larger

        if spatial_strategy not in ["resize", "pad"]:
            raise ValueError(
                f"spatial_strategy must be 'resize' or 'pad', got '{spatial_strategy}'"
            )

    def __call__(self, file_path: str) -> torch.Tensor:
        """
        Load RGB file.

        Args:
            file_path: Path to RGB TIF file

        Returns:
            RGB tensor [C, H, W] as uint8 (if no resizing) or float32 (if resized/padded)
        """
        with rasterio.open(file_path) as src:
            # Read all bands: [C, H, W]
            rgb_data = src.read()  # Shape: [bands, height, width]

            # Convert to torch tensor
            rgb_tensor = torch.from_numpy(rgb_data)

            # Ensure we have 3 channels
            if rgb_tensor.shape[0] != 3:
                raise ValueError(f"Expected 3 RGB channels, got {rgb_tensor.shape[0]}")

            # Apply spatial standardization if target size specified
            if self.target_size is not None:
                rgb_tensor = self._apply_spatial_transform(rgb_tensor)

            return rgb_tensor

    def _apply_spatial_transform(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply spatial transformation (resize or pad) to tensor."""
        C, H, W = tensor.shape
        target_h, target_w = self.target_size

        if self.spatial_strategy == "resize":
            # Resize using nearest neighbor interpolation
            tensor = F.interpolate(
                tensor.unsqueeze(0).float(),  # Add batch dim and convert to float
                size=self.target_size,
                mode="nearest",
            ).squeeze(
                0
            )  # Remove batch dim

        elif self.spatial_strategy == "pad":
            # Convert to float for consistent output type
            tensor = tensor.float()

            # Determine crop region if image is larger than target
            h_to_place, w_to_place = H, W
            h_start, w_start = 0, 0

            if H > target_h and self.crop_if_larger:
                h_start = (H - target_h) // 2  # Center crop
                h_to_place = target_h
            elif H > target_h:
                h_to_place = target_h

            if W > target_w and self.crop_if_larger:
                w_start = (W - target_w) // 2  # Center crop
                w_to_place = target_w
            elif W > target_w:
                w_to_place = target_w

            # Crop if needed
            tensor_to_place = tensor[
                :, h_start : h_start + h_to_place, w_start : w_start + w_to_place
            ]
            actual_h, actual_w = tensor_to_place.shape[1], tensor_to_place.shape[2]

            # Create padded canvas
            padded_tensor = torch.full(
                (C, target_h, target_w), self.padding_value, dtype=tensor_to_place.dtype
            )

            # Center the image on the canvas
            h_pad = (target_h - actual_h) // 2
            w_pad = (target_w - actual_w) // 2

            padded_tensor[:, h_pad : h_pad + actual_h, w_pad : w_pad + actual_w] = (
                tensor_to_place
            )
            tensor = padded_tensor

        return tensor


class HSILoader:
    """Load HSI TIF files and convert to torch tensors."""

    def __init__(
        self,
        target_size: Optional[tuple] = (12, 12),
        spatial_strategy: str = "pad",
        padding_value: float = 0.0,
        crop_if_larger: bool = True,
    ):
        """
        Initialize HSI loader.

        Args:
            target_size: Target size (H, W) to standardize images to. If None, no standardization.
            spatial_strategy: 'resize' for interpolation or 'pad' for padding to target size
            padding_value: Value to use for padding when spatial_strategy='pad'
            crop_if_larger: Whether to crop images larger than target_size when padding
        """
        self.target_size = target_size
        self.spatial_strategy = spatial_strategy
        self.padding_value = padding_value
        self.crop_if_larger = crop_if_larger

        if spatial_strategy not in ["resize", "pad"]:
            raise ValueError(
                f"spatial_strategy must be 'resize' or 'pad', got '{spatial_strategy}'"
            )

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

            # Apply spatial standardization if target size specified
            if self.target_size is not None:
                hsi_tensor = self._apply_spatial_transform(hsi_tensor)

            return hsi_tensor

    def _apply_spatial_transform(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply spatial transformation (resize or pad) to tensor."""
        Bands, H, W = tensor.shape
        target_h, target_w = self.target_size

        if self.spatial_strategy == "resize":
            # Use bilinear interpolation for HSI to preserve spectral smoothness
            tensor = F.interpolate(
                tensor.unsqueeze(0),  # Add batch dim
                size=self.target_size,
                mode="bilinear",
                align_corners=False,
            ).squeeze(
                0
            )  # Remove batch dim

        elif self.spatial_strategy == "pad":
            # Determine crop region if image is larger than target
            h_to_place, w_to_place = H, W
            h_start, w_start = 0, 0

            if H > target_h and self.crop_if_larger:
                h_start = (H - target_h) // 2  # Center crop
                h_to_place = target_h
            elif H > target_h:
                h_to_place = target_h

            if W > target_w and self.crop_if_larger:
                w_start = (W - target_w) // 2  # Center crop
                w_to_place = target_w
            elif W > target_w:
                w_to_place = target_w

            # Crop if needed
            tensor_to_place = tensor[
                :, h_start : h_start + h_to_place, w_start : w_start + w_to_place
            ]
            actual_h, actual_w = tensor_to_place.shape[1], tensor_to_place.shape[2]

            # Create padded canvas
            padded_tensor = torch.full(
                (Bands, target_h, target_w),
                self.padding_value,
                dtype=tensor_to_place.dtype,
            )

            # Center the image on the canvas
            h_pad = (target_h - actual_h) // 2
            w_pad = (target_w - actual_w) // 2

            padded_tensor[:, h_pad : h_pad + actual_h, w_pad : w_pad + actual_w] = (
                tensor_to_place
            )
            tensor = padded_tensor

        return tensor


class LiDARLoader:
    """Load LiDAR TIF files and convert to torch tensors."""

    def __init__(
        self,
        fill_nodata: bool = True,
        target_size: Optional[tuple] = (12, 12),
        spatial_strategy: str = "pad",
        padding_value: float = 0.0,
        crop_if_larger: bool = True,
    ):
        """
        Initialize LiDAR loader.

        Args:
            fill_nodata: Whether to fill NoData values automatically
            target_size: Target size (H, W) to standardize images to. If None, no standardization.
            spatial_strategy: 'resize' for interpolation or 'pad' for padding to target size
            padding_value: Value to use for padding when spatial_strategy='pad'
            crop_if_larger: Whether to crop images larger than target_size when padding
        """
        self.fill_nodata = fill_nodata
        self.target_size = target_size
        self.spatial_strategy = spatial_strategy
        self.padding_value = padding_value
        self.crop_if_larger = crop_if_larger

        if spatial_strategy not in ["resize", "pad"]:
            raise ValueError(
                f"spatial_strategy must be 'resize' or 'pad', got '{spatial_strategy}'"
            )

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

            # Apply spatial standardization if target size specified
            if self.target_size is not None:
                lidar_tensor = self._apply_spatial_transform(lidar_tensor)

            return lidar_tensor

    def _apply_spatial_transform(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply spatial transformation (resize or pad) to tensor."""
        Channels, H, W = tensor.shape
        target_h, target_w = self.target_size

        if self.spatial_strategy == "resize":
            # Use bilinear interpolation for LiDAR height data smoothness
            tensor = F.interpolate(
                tensor.unsqueeze(0),  # Add batch dim
                size=self.target_size,
                mode="bilinear",
                align_corners=False,
            ).squeeze(
                0
            )  # Remove batch dim

        elif self.spatial_strategy == "pad":
            # Determine crop region if image is larger than target
            h_to_place, w_to_place = H, W
            h_start, w_start = 0, 0

            if H > target_h and self.crop_if_larger:
                h_start = (H - target_h) // 2  # Center crop
                h_to_place = target_h
            elif H > target_h:
                h_to_place = target_h

            if W > target_w and self.crop_if_larger:
                w_start = (W - target_w) // 2  # Center crop
                w_to_place = target_w
            elif W > target_w:
                w_to_place = target_w

            # Crop if needed
            tensor_to_place = tensor[
                :, h_start : h_start + h_to_place, w_start : w_start + w_to_place
            ]
            actual_h, actual_w = tensor_to_place.shape[1], tensor_to_place.shape[2]

            # Create padded canvas
            padded_tensor = torch.full(
                (Channels, target_h, target_w),
                self.padding_value,
                dtype=tensor_to_place.dtype,
            )

            # Center the image on the canvas
            h_pad = (target_h - actual_h) // 2
            w_pad = (target_w - actual_w) // 2

            padded_tensor[:, h_pad : h_pad + actual_h, w_pad : w_pad + actual_w] = (
                tensor_to_place
            )
            tensor = padded_tensor

        return tensor


# ============================================================================
# MAIN DATASET CLASS
# ============================================================================


class NeonCrownDataset(Dataset):
    """
    Dataset for NEON tree crown data supporting multiple modalities.

    Expected CSV format:
    crown_id,site,year,plot,species,easting,northing,rgb_path,hsi_path,lidar_path
    """

    @classmethod
    def load(
        cls,
        csv_path: str = "training_data_clean.csv",
        modalities: List[str] = ["rgb"],
        **kwargs,
    ) -> "NeonCrownDataset":
        """
        Load dataset with sensible defaults for exploration.

        Args:
            csv_path: Path to CSV file with crown metadata
            modalities: List of modalities to load ['rgb', 'hsi', 'lidar']
            **kwargs: Additional arguments passed to NeonCrownDataset constructor

        Returns:
            NeonCrownDataset instance ready for exploration

        Example:
            >>> dataset = NeonCrownDataset.load()  # Load with RGB only
            >>> dataset.summary()  # Print overview
            >>>
            >>> # Load all modalities
            >>> full_dataset = NeonCrownDataset.load(modalities=['rgb', 'hsi', 'lidar'])
        """
        # Set sensible defaults
        defaults = {
            "base_data_dir": "/",
            "has_labels": True,
        }

        # Update with user-provided kwargs
        defaults.update(kwargs)

        return cls(csv_path=csv_path, modalities=modalities, **defaults)

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
        # RGB parameters
        rgb_target_size: Optional[tuple] = (128, 128),
        rgb_spatial_strategy: str = "resize",
        rgb_padding_value: float = 0.0,
        # HSI parameters
        hsi_target_size: Optional[tuple] = (12, 12),
        hsi_spatial_strategy: str = "pad",
        hsi_padding_value: float = 0.0,
        # LiDAR parameters
        lidar_target_size: Optional[tuple] = (12, 12),
        lidar_spatial_strategy: str = "pad",
        lidar_padding_value: float = 0.0,
        lidar_fill_nodata: bool = True,
        # Common parameters
        crop_if_larger: bool = True,
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
            rgb_target_size: Target size (H, W) for RGB images. If None, no standardization.
            rgb_spatial_strategy: 'resize' or 'pad' for RGB spatial handling
            rgb_padding_value: Padding value for RGB when using 'pad' strategy
            hsi_target_size: Target size (H, W) for HSI images. If None, no standardization.
            hsi_spatial_strategy: 'resize' or 'pad' for HSI spatial handling
            hsi_padding_value: Padding value for HSI when using 'pad' strategy
            lidar_target_size: Target size (H, W) for LiDAR images. If None, no standardization.
            lidar_spatial_strategy: 'resize' or 'pad' for LiDAR spatial handling
            lidar_padding_value: Padding value for LiDAR when using 'pad' strategy
            lidar_fill_nodata: Whether to fill NoData values in LiDAR
            crop_if_larger: Whether to crop images larger than target size when padding
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

        # Initialize loaders with their respective parameters
        self.loaders = {
            "rgb": RGBLoader(
                target_size=rgb_target_size,
                spatial_strategy=rgb_spatial_strategy,
                padding_value=rgb_padding_value,
                crop_if_larger=crop_if_larger,
            ),
            "hsi": HSILoader(
                target_size=hsi_target_size,
                spatial_strategy=hsi_spatial_strategy,
                padding_value=hsi_padding_value,
                crop_if_larger=crop_if_larger,
            ),
            "lidar": LiDARLoader(
                fill_nodata=lidar_fill_nodata,
                target_size=lidar_target_size,
                spatial_strategy=lidar_spatial_strategy,
                padding_value=lidar_padding_value,
                crop_if_larger=crop_if_larger,
            ),
        }

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

    def filter(
        self,
        species: Optional[Union[str, List[str]]] = None,
        sites: Optional[Union[str, List[str]]] = None,
        years: Optional[Union[int, List[int]]] = None,
        height_range: Optional[tuple] = None,
        **kwargs,
    ) -> "NeonCrownDataset":
        """
        Create a new filtered dataset instance.

        Args:
            species: Species code(s) to include (e.g., 'PSMEM' or ['PSMEM', 'TSHE'])
            sites: Site code(s) to include (e.g., 'ABBY' or ['ABBY', 'HARV'])
            years: Year(s) to include (e.g., 2018 or [2018, 2019])
            height_range: Tuple of (min_height, max_height) in meters
            **kwargs: Additional column filters as column_name=value pairs

        Returns:
            New NeonCrownDataset instance with filtered data

        Example:
            >>> # Filter for Douglas Fir at ABBY site
            >>> filtered = dataset.filter(species='PSMEM', sites='ABBY')
            >>>
            >>> # Chain filtering
            >>> tall_trees = dataset.filter(height_range=(15, 50))
            >>> recent_tall = tall_trees.filter(years=[2019, 2020])
        """
        # Start with current data
        filtered_df = self.data.copy()

        # Apply species filter
        if species is not None:
            if isinstance(species, str):
                species = [species]
            filtered_df = filtered_df[filtered_df["species"].isin(species)]

        # Apply sites filter
        if sites is not None:
            if isinstance(sites, str):
                sites = [sites]
            filtered_df = filtered_df[filtered_df["site"].isin(sites)]

        # Apply years filter
        if years is not None:
            if isinstance(years, int):
                years = [years]
            filtered_df = filtered_df[filtered_df["year"].isin(years)]

        # Apply height filter
        if height_range is not None:
            min_height, max_height = height_range
            if "height" in filtered_df.columns:
                filtered_df = filtered_df[
                    (filtered_df["height"] >= min_height)
                    & (filtered_df["height"] <= max_height)
                ]

        # Apply any additional keyword filters
        for column, value in kwargs.items():
            if column in filtered_df.columns:
                if isinstance(value, (list, tuple)):
                    filtered_df = filtered_df[filtered_df[column].isin(value)]
                else:
                    filtered_df = filtered_df[filtered_df[column] == value]

        # Create temporary CSV for filtered data
        import tempfile

        temp_csv = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
        filtered_df.to_csv(temp_csv.name, index=False)
        temp_csv.close()

        # Create new dataset instance with same parameters
        filtered_dataset = NeonCrownDataset(
            csv_path=temp_csv.name,
            base_data_dir=self.base_data_dir,
            modalities=self.modalities,
            has_labels=self.has_labels,
            transforms=self.transforms,
            label_to_idx=self.label_to_idx,
            # RGB parameters
            rgb_target_size=self.loaders["rgb"].target_size,
            rgb_spatial_strategy=self.loaders["rgb"].spatial_strategy,
            rgb_padding_value=self.loaders["rgb"].padding_value,
            # HSI parameters
            hsi_target_size=self.loaders["hsi"].target_size,
            hsi_spatial_strategy=self.loaders["hsi"].spatial_strategy,
            hsi_padding_value=self.loaders["hsi"].padding_value,
            # LiDAR parameters
            lidar_target_size=self.loaders["lidar"].target_size,
            lidar_spatial_strategy=self.loaders["lidar"].spatial_strategy,
            lidar_padding_value=self.loaders["lidar"].padding_value,
            lidar_fill_nodata=self.loaders["lidar"].fill_nodata,
            crop_if_larger=self.loaders["rgb"].crop_if_larger,
        )

        # Clean up temp file
        import os

        os.unlink(temp_csv.name)

        return filtered_dataset

    def get_dataset_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive dataset statistics for README generation.

        Returns:
            Dictionary with dataset statistics including counts, distributions, etc.
        """
        stats = {
            "total_individuals": len(self.data),
            "species_count": len(self.get_species_list()),
            "sites_count": len(self.get_sites()),
            "years": self.get_years(),
            "sites": self.get_sites(),
        }

        # Species distribution
        if self.has_labels and "species" in self.data.columns:
            species_counts = self.data["species"].value_counts()
            stats["top_species"] = species_counts.head(10).to_dict()
            stats["species_distribution"] = species_counts.to_dict()

        # Species names if available
        if "species_name" in self.data.columns:
            species_name_counts = self.data["species_name"].value_counts()
            stats["top_species_names"] = species_name_counts.head(10).to_dict()

        # Site distribution
        site_counts = self.data["site"].value_counts()
        stats["site_distribution"] = site_counts.to_dict()

        # Year distribution
        year_counts = self.data["year"].value_counts().sort_index()
        stats["year_distribution"] = year_counts.to_dict()

        # Height statistics if available
        if "height" in self.data.columns:
            height_data = self.data["height"].dropna()
            if len(height_data) > 0:
                stats["height_stats"] = {
                    "mean": float(height_data.mean()),
                    "std": float(height_data.std()),
                    "min": float(height_data.min()),
                    "max": float(height_data.max()),
                    "median": float(height_data.median()),
                }

        # Modalities availability
        modality_columns = {"rgb": "rgb_path", "hsi": "hsi_path", "lidar": "lidar_path"}
        modality_availability = {}
        for modality, col in modality_columns.items():
            if col in self.data.columns:
                available = self.data[col].notna().sum()
                modality_availability[modality] = {
                    "available": int(available),
                    "percentage": float(available / len(self.data) * 100),
                }
        stats["modality_availability"] = modality_availability

        return stats

    def summary(self) -> None:
        """Print a comprehensive dataset summary."""
        stats = self.get_dataset_stats()

        print("=" * 50)
        print("NEON Tree Crown Dataset Summary")
        print("=" * 50)

        # Basic counts
        print(f"Total individuals: {stats['total_individuals']:,}")
        print(f"Species: {stats['species_count']}")
        print(f"Sites: {stats['sites_count']}")
        print(
            f"Years: {stats['years'][0]}-{stats['years'][-1]} ({len(stats['years'])} years)"
        )

        # Modalities
        print("\nModality Coverage:")
        for modality, info in stats.get("modality_availability", {}).items():
            print(
                f"  {modality.upper()}: {info['available']:,} samples ({info['percentage']:.1f}%)"
            )

        # Top species
        if "top_species_names" in stats:
            print("\nTop 10 Species:")
            for i, (species, count) in enumerate(stats["top_species_names"].items(), 1):
                percentage = count / stats["total_individuals"] * 100
                print(f"  {i:2d}. {species}: {count:,} ({percentage:.1f}%)")
        elif "top_species" in stats:
            print("\nTop 10 Species (by code):")
            for i, (species, count) in enumerate(stats["top_species"].items(), 1):
                percentage = count / stats["total_individuals"] * 100
                print(f"  {i:2d}. {species}: {count:,} ({percentage:.1f}%)")

        # Top sites
        print("\nTop 10 Sites:")
        site_items = list(stats["site_distribution"].items())
        site_items.sort(key=lambda x: x[1], reverse=True)
        for i, (site, count) in enumerate(site_items[:10], 1):
            percentage = count / stats["total_individuals"] * 100
            print(f"  {i:2d}. {site}: {count:,} ({percentage:.1f}%)")

        # Height statistics
        if "height_stats" in stats:
            h = stats["height_stats"]
            print(f"\nTree Height Statistics:")
            print(f"  Mean: {h['mean']:.1f}m ± {h['std']:.1f}m")
            print(f"  Range: {h['min']:.1f}m - {h['max']:.1f}m")
            print(f"  Median: {h['median']:.1f}m")

        print("=" * 50)
