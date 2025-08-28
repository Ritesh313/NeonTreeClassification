#!/usr/bin/env python3
"""
Script to analyze label overlap between two datasets for training/testing compatibility.
"""

import pandas as pd
import numpy as np
from collections import Counter


def analyze_datasets():
    # File paths
    small_file = "/blue/azare/riteshchowdhry/Macrosystems/Data_files/hand_annotated_neon/curated_tiles_20250822/cropped_crowns_npy/filtered_training_data.csv"
    big_file = "/blue/azare/riteshchowdhry/Macrosystems/Data_files/hand_annotated_neon/curated_vst_tiles_20250825/cropped_crowns_npy/filtered_training_data_42k.csv"

    print("Loading datasets...")
    # Read datasets
    small_df = pd.read_csv(small_file)
    big_df = pd.read_csv(big_file)

    print(f"Small dataset size: {len(small_df):,} samples")
    print(f"Big dataset size: {len(big_df):,} samples")
    print()

    # Extract unique labels (species codes)
    small_labels = set(small_df["species"].unique())
    big_labels = set(big_df["species"].unique())

    print(f"Small dataset - Unique species: {len(small_labels)}")
    print(f"Big dataset - Unique species: {len(big_labels)}")
    print()

    # Calculate overlap
    overlap = small_labels.intersection(big_labels)
    small_only = small_labels - big_labels
    big_only = big_labels - small_labels

    print(f"Species overlap: {len(overlap)} species")
    print(f"Species only in small dataset: {len(small_only)}")
    print(f"Species only in big dataset: {len(big_only)}")
    print()

    # Calculate coverage percentages
    small_coverage = (
        len(overlap) / len(small_labels) * 100 if len(small_labels) > 0 else 0
    )
    big_coverage = len(overlap) / len(big_labels) * 100 if len(big_labels) > 0 else 0

    print(f"Small dataset coverage by big dataset: {small_coverage:.1f}%")
    print(f"Big dataset coverage by small dataset: {big_coverage:.1f}%")
    print()

    # Show overlapping species
    print("Overlapping species:")
    for species in sorted(overlap):
        small_count = len(small_df[small_df["species"] == species])
        big_count = len(big_df[big_df["species"] == species])
        print(f"  {species}: Small={small_count:,}, Big={big_count:,}")
    print()

    # Show species only in small dataset
    if small_only:
        print("Species only in small dataset (won't be trainable):")
        for species in sorted(small_only):
            count = len(small_df[small_df["species"] == species])
            print(f"  {species}: {count:,} samples")
        print()

    # Show species only in big dataset
    if big_only:
        print("Species only in big dataset (won't be testable):")
        for species in sorted(big_only):
            count = len(big_df[big_df["species"] == species])
            print(f"  {species}: {count:,} samples")
        print()

    # Calculate usable samples for training/testing
    small_usable = small_df[small_df["species"].isin(overlap)]
    big_usable = big_df[big_df["species"].isin(overlap)]

    print(f"Usable samples for training (from big dataset): {len(big_usable):,}")
    print(f"Usable samples for testing (from small dataset): {len(small_usable):,}")
    print()

    # Show class distribution in overlapping data
    print("Class distribution in overlapping data:")
    print("\nSmall dataset (test set):")
    small_counts = small_usable["species"].value_counts()
    for species, count in small_counts.items():
        print(f"  {species}: {count:,}")

    print("\nBig dataset (train set):")
    big_counts = big_usable["species"].value_counts()
    for species, count in big_counts.items():
        print(f"  {species}: {count:,}")
    print()

    # Calculate class imbalance ratios
    print("Training-to-testing sample ratios:")
    for species in sorted(overlap):
        small_count = len(small_df[small_df["species"] == species])
        big_count = len(big_df[big_df["species"] == species])
        if small_count > 0:
            ratio = big_count / small_count
            print(f"  {species}: {ratio:.1f}x more training samples")
    print()

    # Final recommendation
    print("=== RECOMMENDATION ===")
    if small_coverage >= 80:
        print("✅ FEASIBLE: High coverage of small dataset species in big dataset")
        print(
            f"   - {small_coverage:.1f}% of test species are covered in training data"
        )
        print(
            f"   - Can train on {len(big_usable):,} samples and test on {len(small_usable):,} samples"
        )
        if len(small_only) > 0:
            print(
                f"   - Note: {len(small_only)} species in test set won't have training data"
            )
    elif small_coverage >= 60:
        print("⚠️  PARTIALLY FEASIBLE: Moderate coverage")
        print(
            f"   - {small_coverage:.1f}% of test species are covered in training data"
        )
        print(f"   - Consider if the {len(small_only)} missing species are critical")
    else:
        print("❌ NOT RECOMMENDED: Poor coverage")
        print(
            f"   - Only {small_coverage:.1f}% of test species are covered in training data"
        )
        print(f"   - {len(small_only)} species in test set won't have training data")


if __name__ == "__main__":
    analyze_datasets()
