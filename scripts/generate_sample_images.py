#!/usr/bin/env python3
"""
Generate sample visualization images for README using direct imports.
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Add current directory to path
sys.path.append(".")

# Create sample_plots directory
os.makedirs("sample_plots", exist_ok=True)


def generate_sample_images():
    """Generate sample visualization images."""

    # Load dataset to get sample paths
    df = pd.read_csv("training_data_clean.csv")

    # Find a good sample (one with all three modalities)
    sample_found = False
    for idx, row in df.head(10).iterrows():  # Check first 10 samples
        rgb_path = row["rgb_path"]
        hsi_path = row["hsi_path"]
        lidar_path = row["lidar_path"]

        # Check if all files exist
        if (
            Path(rgb_path).exists()
            and Path(hsi_path).exists()
            and Path(lidar_path).exists()
        ):

            print(f"Using sample {idx}: {row['crown_id']}")
            print(f"Species: {row['species_name']}")
            print(f"Site: {row['site']}")
            sample_found = True
            break

    if not sample_found:
        print("No complete samples found with all three modalities in first 10 samples")
        print("Checking if any files exist...")
        for idx, row in df.head(5).iterrows():
            print(f"Sample {idx}:")
            print(f"  RGB exists: {Path(row['rgb_path']).exists()}")
            print(f"  HSI exists: {Path(row['hsi_path']).exists()}")
            print(f"  LiDAR exists: {Path(row['lidar_path']).exists()}")
        return

    # Import visualization functions directly
    try:
        # Direct import from the file
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "visualization",
            "neon_tree_classification/core/visualization.py",
        )
        viz_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(viz_module)

        plot_rgb = viz_module.plot_rgb
        plot_hsi = viz_module.plot_hsi
        plot_hsi_pca = viz_module.plot_hsi_pca
        plot_hsi_spectra = viz_module.plot_hsi_spectra
        plot_lidar = viz_module.plot_lidar

    except Exception as e:
        print(f"Could not import visualization functions: {e}")
        return

    print("Generating visualization samples...")

    # Generate individual plots
    try:
        # RGB
        print("Generating RGB plot...")
        fig_rgb = plot_rgb(rgb_path)
        fig_rgb.savefig("sample_plots/sample_rgb.png", dpi=150, bbox_inches="tight")
        plt.close(fig_rgb)
        print("✓ RGB image saved")

        # HSI Pseudo RGB
        print("Generating HSI pseudo RGB plot...")
        fig_hsi = plot_hsi(hsi_path)
        fig_hsi.savefig("sample_plots/sample_hsi.png", dpi=150, bbox_inches="tight")
        plt.close(fig_hsi)
        print("✓ HSI pseudo RGB saved")

        # HSI PCA
        print("Generating HSI PCA plot...")
        fig_hsi_pca = plot_hsi_pca(hsi_path)
        fig_hsi_pca.savefig(
            "sample_plots/sample_hsi_pca.png", dpi=150, bbox_inches="tight"
        )
        plt.close(fig_hsi_pca)
        print("✓ HSI PCA saved")

        # HSI Spectra
        print("Generating HSI spectra plot...")
        fig_spectra = plot_hsi_spectra(hsi_path, num_pixels=3)
        fig_spectra.savefig(
            "sample_plots/sample_spectra.png", dpi=150, bbox_inches="tight"
        )
        plt.close(fig_spectra)
        print("✓ HSI spectra saved")

        # LiDAR
        print("Generating LiDAR plot...")
        fig_lidar = plot_lidar(lidar_path)
        fig_lidar.savefig("sample_plots/sample_lidar.png", dpi=150, bbox_inches="tight")
        plt.close(fig_lidar)
        print("✓ LiDAR CHM saved")

        print("\n✅ All sample images generated successfully!")
        print("Generated files:")
        print("  - sample_plots/sample_rgb.png")
        print("  - sample_plots/sample_hsi.png")
        print("  - sample_plots/sample_hsi_pca.png")
        print("  - sample_plots/sample_spectra.png")
        print("  - sample_plots/sample_lidar.png")

    except Exception as e:
        print(f"Error generating plots: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    generate_sample_images()
