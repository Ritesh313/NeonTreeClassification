"""
Simple NEON tree crown visualization functions.

Usage Examples:
    In Jupyter notebook:
        fig = plot_rgb('path/to/crown.tif')  # Displays automatically

    In Python script:
        fig = plot_rgb('path/to/crown.tif')
        plt.show()  # Explicitly show
        # or
        fig.savefig('crown_plot.png')  # Save to file

Author: Ritesh Chowdhry
"""

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from sklearn.decomposition import PCA

# NEON HSI band indices for RGB approximation
# Based on NEON's hyperspectral wavelengths (~380-2510 nm, 426 bands)
HSI_RGB_BANDS = {"red": 190, "green": 90, "blue": 28}  # ~660 nm  # ~550 nm  # ~450 nm


def plot_rgb(rgb_path, figsize=(3, 3)):
    """Plot RGB image with basic contrast stretching.

    Args:
        rgb_path (str): Path to RGB TIFF file
        figsize (tuple): Figure size as (width, height)

    Returns:
        matplotlib.figure.Figure: The figure object

    Example:
        >>> fig = plot_rgb('crown.tif')
        >>> fig.savefig('my_crown.png')
    """
    with rasterio.open(rgb_path) as src:
        rgb = src.read([1, 2, 3]).transpose(1, 2, 0)

    # Simple contrast stretch
    rgb_stretched = np.clip((rgb - rgb.min()) / (rgb.max() - rgb.min()), 0, 1)

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(rgb_stretched, interpolation="nearest")
    ax.axis("off")
    ax.set_title("RGB Crown")
    fig.tight_layout()

    return fig


def plot_hsi(hsi_path, figsize=(3, 3)):
    """Plot HSI as pseudo RGB using specific wavelength bands.

    Args:
        hsi_path (str): Path to HSI TIFF file
        figsize (tuple): Figure size as (width, height)

    Returns:
        matplotlib.figure.Figure: The figure object

    Example:
        >>> fig = plot_hsi('crown_hsi.tif')
        >>> fig.savefig('my_hsi_crown.png')
    """
    with rasterio.open(hsi_path) as src:
        # Read all bands (426 bands for NEON HSI)
        hsi = src.read()  # Shape: (bands, height, width)

    # Extract RGB-like bands
    red_band = hsi[HSI_RGB_BANDS["red"], :, :]
    green_band = hsi[HSI_RGB_BANDS["green"], :, :]
    blue_band = hsi[HSI_RGB_BANDS["blue"], :, :]

    # Stack into RGB format
    pseudo_rgb = np.stack([red_band, green_band, blue_band], axis=2)

    # Simple contrast stretch
    rgb_stretched = np.clip(
        (pseudo_rgb - pseudo_rgb.min()) / (pseudo_rgb.max() - pseudo_rgb.min()), 0, 1
    )

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(rgb_stretched, interpolation="nearest")
    ax.axis("off")
    ax.set_title("HSI Pseudo RGB")
    fig.tight_layout()

    return fig


def plot_hsi_pca(hsi_path, figsize=(3, 3)):
    """Plot HSI using PCA decomposition to 3 components.

    Args:
        hsi_path (str): Path to HSI TIFF file
        figsize (tuple): Figure size as (width, height)

    Returns:
        matplotlib.figure.Figure: The figure object

    Example:
        >>> fig = plot_hsi_pca('crown_hsi.tif')
        >>> fig.savefig('my_hsi_pca.png')
    """
    with rasterio.open(hsi_path) as src:
        hsi = src.read()  # Shape: (bands, height, width)

    # Reshape for PCA: (pixels, bands)
    h, w = hsi.shape[1], hsi.shape[2]
    hsi_reshaped = hsi.transpose(1, 2, 0).reshape(-1, hsi.shape[0])

    # Remove any NaN or infinite values
    valid_mask = np.isfinite(hsi_reshaped).all(axis=1)
    hsi_clean = hsi_reshaped[valid_mask]

    if len(hsi_clean) == 0:
        raise ValueError("No valid pixels found in HSI data")

    # Apply PCA
    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(hsi_clean)

    # Reconstruct to full image shape
    pca_full = np.full((hsi_reshaped.shape[0], 3), np.nan)
    pca_full[valid_mask] = pca_result
    pca_image = pca_full.reshape(h, w, 3)

    # Normalize each component to 0-1
    for i in range(3):
        component = pca_image[:, :, i]
        valid_component = component[np.isfinite(component)]
        if len(valid_component) > 0:
            pmin, pmax = np.percentile(valid_component, [1, 99])
            component_norm = np.clip((component - pmin) / (pmax - pmin), 0, 1)
            pca_image[:, :, i] = component_norm

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(pca_image, interpolation="nearest")
    ax.axis("off")
    ax.set_title("HSI PCA Decomposition")
    fig.tight_layout()

    return fig


def plot_hsi_spectra(hsi_path, num_pixels=5, figsize=(8, 5)):
    """Plot spectral signatures from random pixels in HSI data.

    Args:
        hsi_path (str): Path to HSI TIFF file
        num_pixels (int): Number of random pixels to plot
        figsize (tuple): Figure size as (width, height)

    Returns:
        matplotlib.figure.Figure: The figure object

    Example:
        >>> fig = plot_hsi_spectra('crown_hsi.tif', num_pixels=3)
        >>> fig.savefig('spectra.png')
    """
    with rasterio.open(hsi_path) as src:
        hsi = src.read()  # Shape: (bands, height, width)

    # Create wavelength array based on actual number of bands
    num_bands = hsi.shape[0]
    wavelengths = np.linspace(380, 2510, num_bands)  # Dynamic wavelength array

    h, w = hsi.shape[1], hsi.shape[2]

    # Find valid pixels (no NaN/infinite values AND bright enough)
    valid_pixels = []
    for i in range(h):
        for j in range(w):
            pixel_spectrum = hsi[:, i, j]
            if np.isfinite(pixel_spectrum).all():
                # Filter out dark background pixels - use NIR band for vegetation detection
                nir_band = pixel_spectrum[
                    HSI_RGB_BANDS["red"]
                ]  # Use red band as proxy for bright pixels
                mean_reflectance = np.mean(pixel_spectrum)

                # Only include pixels with sufficient brightness (vegetation)
                if (
                    mean_reflectance > 0.1 and nir_band > 0.05
                ):  # Threshold for vegetation
                    valid_pixels.append((i, j))

    if len(valid_pixels) == 0:
        raise ValueError("No valid pixels found in HSI data")

    # Randomly sample pixels
    np.random.seed(42)  # For reproducibility
    sample_pixels = np.random.choice(
        len(valid_pixels), min(num_pixels, len(valid_pixels)), replace=False
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Left: HSI pseudo RGB for reference
    red_band = hsi[HSI_RGB_BANDS["red"], :, :]
    green_band = hsi[HSI_RGB_BANDS["green"], :, :]
    blue_band = hsi[HSI_RGB_BANDS["blue"], :, :]
    pseudo_rgb = np.stack([red_band, green_band, blue_band], axis=2)
    rgb_stretched = np.clip(
        (pseudo_rgb - pseudo_rgb.min()) / (pseudo_rgb.max() - pseudo_rgb.min()), 0, 1
    )

    ax1.imshow(rgb_stretched, interpolation="nearest")
    ax1.set_title("HSI Reference (Pixel Locations)")
    ax1.axis("off")

    # Right: Spectral signatures
    colors = plt.cm.tab10(np.linspace(0, 1, len(sample_pixels)))

    for idx, pixel_idx in enumerate(sample_pixels):
        i, j = valid_pixels[pixel_idx]
        spectrum = hsi[:, i, j]

        ax2.plot(
            wavelengths,
            spectrum,
            color=colors[idx],
            linewidth=1.5,
            alpha=0.8,
            label=f"Pixel ({i},{j})",
        )

        # Mark pixel location on reference image
        ax1.plot(
            j,
            i,
            "o",
            color=colors[idx],
            markersize=6,
            markeredgecolor="white",
            markeredgewidth=1,
        )

    ax2.set_xlabel("Wavelength (nm)")
    ax2.set_ylabel("Reflectance")
    ax2.set_title("Spectral Signatures")
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=8)

    # Add spectral regions
    ax2.axvspan(380, 700, alpha=0.1, color="blue", label="Visible")
    ax2.axvspan(700, 1000, alpha=0.1, color="red", label="NIR")
    ax2.axvspan(1000, 2500, alpha=0.1, color="orange", label="SWIR")

    fig.tight_layout()
    return fig


def plot_lidar(lidar_path, figsize=(3, 3), colormap="viridis"):
    """Plot LiDAR CHM (Canopy Height Model) data.

    Args:
        lidar_path (str): Path to LiDAR CHM TIFF file
        figsize (tuple): Figure size as (width, height)
        colormap (str): Colormap for height visualization

    Returns:
        matplotlib.figure.Figure: The figure object

    Example:
        >>> fig = plot_lidar('crown_chm.tif')
        >>> fig.savefig('my_lidar_crown.png')
    """
    with rasterio.open(lidar_path) as src:
        chm = src.read(1)  # Read first (and typically only) band

    # Mask out no-data values if present
    if hasattr(src, "nodata") and src.nodata is not None:
        chm = np.ma.masked_equal(chm, src.nodata)

    # Remove any negative heights (ground level artifacts)
    chm = np.ma.masked_less(chm, 0)

    fig, ax = plt.subplots(figsize=figsize)

    # Plot CHM with colormap
    im = ax.imshow(chm, cmap=colormap, interpolation="nearest")

    # Add colorbar with height legend
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Height (m)", rotation=270, labelpad=15)

    ax.axis("off")
    ax.set_title("LiDAR CHM")
    fig.tight_layout()

    return fig
