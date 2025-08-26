#!/usr/bin/env python3
"""
Filter rare species from NEON training data.

Remove species that have fewer than minimum number of samples,
which helps with stratified train/test splits.
"""

import argparse
import pandas as pd
from pathlib import Path


def filter_csv_by_species_count(input_csv, output_csv, min_samples=6):
    """
    Filter CSV to remove species with fewer than min_samples.

    Args:
        input_csv: Path to input CSV file
        output_csv: Path to output filtered CSV file
        min_samples: Minimum number of samples required per species

    Returns:
        Path to output CSV file
    """
    print(f"🔍 Loading data from: {input_csv}")
    df = pd.read_csv(input_csv)

    # Count samples per species
    species_counts = df["species"].value_counts()
    valid_species = species_counts[species_counts >= min_samples].index
    rare_species = species_counts[species_counts < min_samples].index

    # Filter data
    filtered_df = df[df["species"].isin(valid_species)]

    print(f"\n📊 Filtering Results:")
    print(f"  Original: {len(df):,} samples, {df['species'].nunique()} species")
    print(
        f"  Filtered: {len(filtered_df):,} samples, {filtered_df['species'].nunique()} species"
    )
    print(
        f"  Removed:  {len(df) - len(filtered_df):,} samples from {len(rare_species)} rare species"
    )

    if len(rare_species) > 0:
        print(f"\n🗑️ Removed species (< {min_samples} samples):")
        for species in rare_species[:10]:  # Show first 10
            count = species_counts[species]
            print(f"    {species}: {count} sample{'s' if count != 1 else ''}")
        if len(rare_species) > 10:
            print(f"    ... and {len(rare_species) - 10} more")

    # Create output directory if needed
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save filtered data
    filtered_df.to_csv(output_csv, index=False)
    print(f"\n💾 Filtered data saved to: {output_csv}")

    return output_csv


def main():
    parser = argparse.ArgumentParser(
        description="Filter rare species from NEON training data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        Example usage:
        python filter_rare_species.py \\
            --input neon_training_data.csv \\
            --output neon_training_data_filtered.csv \\
            --min-samples 6
        """,
        )

    parser.add_argument("--input", required=True, help="Path to input CSV file")
    parser.add_argument(
        "--output", required=True, help="Path to output filtered CSV file"
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=6,
        help="Minimum number of samples required per species (default: 6)",
    )

    args = parser.parse_args()

    # Validate input file exists
    if not Path(args.input).exists():
        print(f"❌ Error: Input file not found: {args.input}")
        return 1

    try:
        filter_csv_by_species_count(args.input, args.output, args.min_samples)
        print("✅ Filtering completed successfully!")
        return 0

    except Exception as e:
        print(f"❌ Error during filtering: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
