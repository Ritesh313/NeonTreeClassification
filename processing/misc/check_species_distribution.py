#!/usr/bin/env python3
"""
Check species distribution in dataset.

Usage:
    python check_species_distribution.py dataset.csv
"""
import pandas as pd
import argparse


def check_distribution(csv_path):
    """Check and print species distribution."""
    df = pd.read_csv(csv_path)
    species_counts = df["species"].value_counts()

    print(f"📊 Dataset: {len(df):,} samples, {len(species_counts)} species")

    # Distribution breakdown
    single = (species_counts == 1).sum()
    few = ((species_counts >= 2) & (species_counts <= 5)).sum()
    medium = ((species_counts >= 6) & (species_counts <= 10)).sum()
    many = (species_counts >= 11).sum()

    print(f"   1 sample: {single} species")
    print(f"   2-5 samples: {few} species")
    print(f"   6-10 samples: {medium} species")
    print(f"   11+ samples: {many} species")

    print(f"\nTop 10 species:")
    for species, count in species_counts.head(10).items():
        print(f"   {species}: {count:,}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check species distribution")
    parser.add_argument("csv_path", help="Path to dataset CSV file")
    args = parser.parse_args()

    check_distribution(args.csv_path)
