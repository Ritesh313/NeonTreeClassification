#!/usr/bin/env python3
"""
Training script for NEON tree species classification.

Usage examples:
  # Train RGB classifier
  python train.py --modality rgb --model_type resnet --data_dir /path/to/data --csv_path /path/to/crowns.csv

  # Train HSI classifier with custom params
  python train.py --modality hsi --model_type spectral_cnn --num_bands 426 --lr 5e-4 --batch_size 16

  # Train LiDAR classifier
  python train.py --modality lidar --model_type height_cnn --epochs 30
"""

import argparse
import os
from datetime import datetime
import lightning as L
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
)
from lightning.pytorch.loggers import TensorBoardLogger
import torch

# Optimize CUDA performance for Tensor Cores
if torch.cuda.is_available():
    torch.set_float32_matmul_precision("medium")

try:
    from lightning.pytorch.loggers import CometLogger

    COMET_AVAILABLE = True
except ImportError:
    COMET_AVAILABLE = False

from neon_tree_classification.core.datamodule import NeonCrownDataModule
from neon_tree_classification.models.lightning_modules import (
    RGBClassifier,
    HSIClassifier,
    LiDARClassifier,
)


def main():

    parser = argparse.ArgumentParser(description="Train NEON tree species classifier")

    # Data arguments
    parser.add_argument(
        "--csv_path", type=str, required=True, help="Path to crown CSV file"
    )
    parser.add_argument(
        "--data_dir", type=str, required=True, help="Base data directory"
    )
    parser.add_argument(
        "--modality", type=str, choices=["rgb", "hsi", "lidar"], required=True
    )

    # Model arguments
    parser.add_argument(
        "--model_type", type=str, default="simple", help="Model architecture type"
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=None,
        help="Number of tree species classes (auto-detected if not provided)",
    )
    parser.add_argument(
        "--num_bands", type=int, default=426, help="Number of HSI bands (HSI only)"
    )

    # Training arguments
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument(
        "--optimizer", type=str, default="adamw", choices=["adam", "adamw", "sgd"]
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        default="plateau",
        choices=["plateau", "cosine", "step"],
    )
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")

    # Data split arguments
    parser.add_argument(
        "--split_method", type=str, default="random", choices=["random", "site", "year"]
    )
    parser.add_argument(
        "--train_ratio", type=float, default=0.7, help="Training split ratio"
    )
    parser.add_argument(
        "--val_ratio", type=float, default=0.15, help="Validation split ratio"
    )
    parser.add_argument(
        "--split_seed", type=int, default=42, help="Random seed for splits"
    )

    # Hardware arguments
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs")
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of data loader workers"
    )
    parser.add_argument(
        "--distributed", action="store_true", help="Enable distributed training"
    )

    # Logging arguments
    parser.add_argument(
        "--logger", type=str, default="tensorboard", choices=["tensorboard", "comet"]
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Directory to save logs, checkpoints, and results (auto-generated if not provided)",
    )

    args = parser.parse_args()

    # Set up experiment name (auto-generate)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = (
        f"{args.modality}_{args.model_type}_{args.lr}_{args.batch_size}_{timestamp}"
    )

    # Set up output directory (organize by modality and timestamp)
    if args.output_dir is None:
        args.output_dir = f"./outputs/{args.modality}_{timestamp}"

    print(f"🌲 Training {args.modality.upper()} classifier: {args.model_type}")
    print(f"📁 Data: {args.data_dir}")
    print(f"🧪 Experiment: {experiment_name}")
    print(f"💾 Output directory: {args.output_dir}")

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Create data module
    datamodule = NeonCrownDataModule(
        csv_path=args.csv_path,
        base_data_dir=args.data_dir,
        modalities=[args.modality],
        split_method=args.split_method,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=1 - args.train_ratio - args.val_ratio,
        split_seed=args.split_seed,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Setup data to get number of classes
    datamodule.setup()

    # Auto-detect or validate number of classes
    actual_num_classes = datamodule.num_classes
    if args.num_classes is None:
        args.num_classes = actual_num_classes
        print(f"🔍 Auto-detected {args.num_classes} classes")
    else:
        if args.num_classes != actual_num_classes:
            raise ValueError(
                f"Mismatch: You specified {args.num_classes} classes, but dataset has {actual_num_classes} classes"
            )
        print(f"✅ Verified {args.num_classes} classes match dataset")

    # Create classifier based on modality
    if args.modality == "rgb":
        classifier = RGBClassifier(
            model_type=args.model_type,
            num_classes=args.num_classes,
            learning_rate=args.lr,
            optimizer=args.optimizer,
            scheduler=args.scheduler,
            weight_decay=args.weight_decay,
            log_images=True,  # Enable image logging for RGB
        )
    elif args.modality == "hsi":
        classifier = HSIClassifier(
            model_type=args.model_type,
            num_bands=args.num_bands,
            num_classes=args.num_classes,
            learning_rate=args.lr,
            optimizer=args.optimizer,
            scheduler=args.scheduler,
            weight_decay=args.weight_decay,
        )
    elif args.modality == "lidar":
        classifier = LiDARClassifier(
            model_type=args.model_type,
            num_classes=args.num_classes,
            learning_rate=args.lr,
            optimizer=args.optimizer,
            scheduler=args.scheduler,
            weight_decay=args.weight_decay,
        )

    # Set up logger
    if args.logger == "comet":
        if not COMET_AVAILABLE:
            raise ImportError(
                "CometML not available. Install with: pip install comet-ml"
            )
        logger = CometLogger(
            save_dir=args.output_dir,
        )
    else:
        logger = TensorBoardLogger(save_dir=args.output_dir, name=experiment_name)

    # Set up callbacks
    callbacks = [
        ModelCheckpoint(
            monitor="val/accuracy",
            mode="max",
            save_top_k=3,
            filename=f"{args.modality}-{args.model_type}-"
            + "{epoch:02d}-{val/accuracy:.3f}",
        ),
        EarlyStopping(monitor="val/loss", patience=10, mode="min"),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    # Create trainer
    trainer = L.Trainer(
        max_epochs=args.epochs,
        accelerator="gpu" if torch.cuda.is_available() and args.gpus > 0 else "cpu",
        devices=args.gpus if torch.cuda.is_available() and args.gpus > 0 else 1,
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=10,
        enable_progress_bar=True,
        precision="16-mixed" if torch.cuda.is_available() else "32",
    )

    # Setup data
    # datamodule.setup()  # Already called above for class detection

    # Get class weights for imbalanced datasets
    print("⚖️  Calculating class weights...")
    class_weights = datamodule.get_class_weights()
    if class_weights is not None:
        classifier.class_weights = class_weights
        print(f"📊 Using class weights for {len(class_weights)} classes")

    # Log dataset info
    print(
        f"📈 Dataset: {len(datamodule.train_dataset)} train, {len(datamodule.val_dataset)} val, {len(datamodule.test_dataset)} test"
    )

    # Train
    print("🚀 Starting training...")
    trainer.fit(classifier, datamodule)

    # Test
    print("🧪 Testing best model...")
    trainer.test(classifier, datamodule, ckpt_path="best")

    print(
        f"✅ Training complete! Best model saved to: {trainer.checkpoint_callback.best_model_path}"
    )


if __name__ == "__main__":
    main()
