#!/usr/bin/env python3
"""Check species distribution in the dataset"""
import pandas as pd

csv_path = "/blue/azare/riteshchowdhry/Macrosystems/Data_files/hand_annotated_neon/curated_tiles_20250822/cropped_crowns_modality_organized/training_data_filtered.csv"
df = pd.read_csv(csv_path)

print("📊 Species Distribution Analysis:")
species_counts = df['species'].value_counts()
print(f"Total species: {len(species_counts)}")
print(f"Total samples: {len(df)}")

# Check how many species have only 1 sample
single_sample_species = species_counts[species_counts == 1]
print(f"\nSpecies with only 1 sample: {len(single_sample_species)}")
if len(single_sample_species) > 0:
    print("These species:", single_sample_species.index.tolist()[:10])  # Show first 10

# Check distribution
print(f"\nSample count distribution:")
print(f"1 sample: {(species_counts == 1).sum()} species")
print(f"2-5 samples: {((species_counts >= 2) & (species_counts <= 5)).sum()} species")
print(f"6-10 samples: {((species_counts >= 6) & (species_counts <= 10)).sum()} species")
print(f"11+ samples: {(species_counts >= 11).sum()} species")

print(f"\nTop 10 most common species:")
print(species_counts.head(10))