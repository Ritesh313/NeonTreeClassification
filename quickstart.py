#!/usr/bin/env python3
"""
NEON Tree Classification Dataset - Quick Start Example

This script demonstrates how to get the dataloaders for the NEON tree classification dataset.
The dataset (590 MB) will be automatically downloaded on first use.

Run this script with:
    uv run python quickstart.py
    # OR (after activating environment):
    python quickstart.py
"""

from scripts.get_dataloaders import get_dataloaders

def main():
    print("🌲 NEON Tree Classification Dataset - Quick Start")
    print("=" * 50)
    
    # Get dataloaders - dataset downloads automatically on first use
    print("📥 Loading dataloaders (downloading dataset if needed)...")
    train_loader, test_loader = get_dataloaders(
        config='large',  # Options: 'combined', 'large', 'high_quality'
        modalities=['rgb', 'hsi', 'lidar'],  # Choose which data types you need
        batch_size=32,
        test_ratio=0.2
    )
    
    print(f"✅ Dataset loaded successfully!")
    print(f"   📊 Training samples: {len(train_loader.dataset)}")
    print(f"   📊 Test samples: {len(test_loader.dataset)}")
    print(f"   📊 Number of species: {len(train_loader.dataset.unique_species)}")
    
    # Show what's in a single batch
    print("\n🔍 Exploring first batch...")
    batch = next(iter(train_loader))
    
    print(f"   🖼️  RGB shape: {batch['rgb'].shape}")        # [batch_size, 3, 128, 128]
    print(f"   🌈 HSI shape: {batch['hsi'].shape}")         # [batch_size, 369, 12, 12]  
    print(f"   🏔️  LiDAR shape: {batch['lidar'].shape}")    # [batch_size, 1, 12, 12]
    print(f"   🏷️  Labels shape: {batch['species_idx'].shape}")  # [batch_size]
    
    # Example of how to use in your own training loop
    print("\n💡 Example usage in your training loop:")
    print("   for batch in train_loader:")
    print("       rgb_data = batch['rgb']")
    print("       hsi_data = batch['hsi'] ")
    print("       lidar_data = batch['lidar']")
    print("       labels = batch['species_idx']")
    print("       # Your training code here...")
    
    print("\n🎉 Ready to use! Copy this code to your own script.")
    print("📖 Check the README for more configuration options.")

if __name__ == "__main__":
    main()
