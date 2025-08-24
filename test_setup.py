#!/usr/bin/env python3
"""
Quick test to verify the training setup works with the new CSV.
"""

import sys

sys.path.append("/blue/azare/riteshchowdhry/Macrosystems/code/NeonTreeClassification")

from neon_tree_classification import NeonCrownDataModule


def main():
    print("🧪 Testing NEON training setup...")

    # Test parameters
    csv_path = "/blue/azare/riteshchowdhry/Macrosystems/code/NeonTreeClassification/training_data_hsi_clean_strict.csv"
    # "/blue/azare/riteshchowdhry/Macrosystems/Data_files/hand_annotated_neon/curated_tiles_20250819/cropped_crowns_modality_organized/neon_training_data_filtered_min6.csv"
    base_data_dir = ""  # Empty string since we have absolute paths

    try:
        # Create data module
        datamodule = NeonCrownDataModule(
            csv_path=csv_path,
            base_data_dir=base_data_dir,
            modalities=["rgb"],  # Test with RGB first
            batch_size=4,  # Small batch for testing
            num_workers=0,  # No multiprocessing for testing
        )

        print("✅ DataModule created successfully")

        # Setup data
        datamodule.setup()
        print("✅ DataModule setup completed")

        # Print dataset info
        print(f"📊 Dataset info:")
        print(f"  Train samples: {len(datamodule.train_dataset)}")
        print(f"  Val samples: {len(datamodule.val_dataset)}")
        print(f"  Test samples: {len(datamodule.test_dataset)}")
        print(f"  Number of classes: {datamodule.num_classes}")
        print(f"  Species: {list(datamodule.get_species_mapping().keys())[:10]}...")

        # Test loading a batch with all modalities
        print("\n✅ Testing all modalities with new spatial handling...")
        all_modalities_datamodule = NeonCrownDataModule(
            csv_path=csv_path,
            base_data_dir=base_data_dir,
            modalities=["rgb", "hsi", "lidar"],  # Test all modalities
            batch_size=4,  # Test with larger batch
            num_workers=0,
            # Use default settings: RGB resize 128x128, HSI/LiDAR pad 12x12
        )
        all_modalities_datamodule.setup()
        train_loader = all_modalities_datamodule.train_dataloader()
        print("✅ Train dataloader created for all modalities")

        # Get one batch
        batch = next(iter(train_loader))
        print("✅ Successfully loaded batch with all modalities")
        print(f"  Batch RGB shape: {batch['rgb'].shape}")
        print(f"  Batch HSI shape: {batch['hsi'].shape}")
        print(f"  Batch LiDAR shape: {batch['lidar'].shape}")
        print(f"  Batch labels shape: {batch['species_idx'].shape}")
        print(f"  Sample labels: {batch['species_idx'].tolist()}")

        # Test loading individual samples to see standardized sizes
        print("\n🔍 Testing individual sample sizes (after spatial handling):")
        for i in range(3):
            sample = all_modalities_datamodule.train_dataset[i]
            print(
                f"  Sample {i}: RGB {sample['rgb'].shape}, HSI {sample['hsi'].shape}, LiDAR {sample['lidar'].shape}, species_idx: {sample['species_idx']}"
            )

        print("\n🎉 All modality data loading tests passed!")
        print("✅ All modalities are now standardized and can be batched!")
        print("🎯 Spatial handling summary:")
        print("   - RGB: Resize to 128x128 (nearest neighbor)")
        print("   - HSI: Pad to 12x12 (preserves spectral data)")
        print("   - LiDAR: Pad to 12x12 (preserves height measurements)")

        # Test the individual RGB functionality we had before
        test_datamodule = NeonCrownDataModule(
            csv_path=csv_path,
            base_data_dir=base_data_dir,
            modalities=["rgb"],
            batch_size=4,  # Test with larger batch now that we have resizing
            num_workers=0,
            rgb_target_size=(128, 128),  # Explicitly set target size
        )
        test_datamodule.setup()
        train_loader = test_datamodule.train_dataloader()
        print("✅ RGB-only dataloader created")

        # Get one batch
        batch = next(iter(train_loader))
        print("✅ Successfully loaded RGB batch with multiple samples")
        print(f"  Batch RGB shape: {batch['rgb'].shape}")
        print(f"  Batch labels shape: {batch['species_idx'].shape}")
        print(f"  Sample labels: {batch['species_idx'].tolist()}")

        # Test loading a few individual samples to see standardized sizes
        print("\n🔍 Testing individual RGB sample sizes (after resizing):")
        for i in range(3):
            sample = test_datamodule.train_dataset[i]
            print(
                f"  Sample {i}: RGB shape {sample['rgb'].shape}, species_idx: {sample['species_idx']}"
            )

        print("\n🎉 Data loading tests passed!")
        print("✅ RGB images are now standardized and can be batched!")
        print("✅ HSI and LiDAR loading finalized with spatial handling!")
        print("✅ All modalities can be batched together!")
        print("🚀 Ready for multi-modal model training!")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
