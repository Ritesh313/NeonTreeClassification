#!/usr/bin/env python3
"""
Compare different modalities on the same dataset splits.

Usage:
  python compare_modalities.py --csv_path /path/to/crowns.csv --data_dir /path/to/data
"""

import argparse
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
import torch
import json
from pathlib import Path

from neon_tree_classification.core.datamodule import NeonCrownDataModule
from neon_tree_classification.models.lightning_modules import (
    RGBClassifier,
    HSIClassifier,
    LiDARClassifier,
)


def train_modality(modality, datamodule, args):
    """Train a single modality and return test results."""
    print(f"\n🌲 Training {modality.upper()} classifier...")

    # Create classifier
    if modality == "rgb":
        classifier = RGBClassifier(
            model_type=args.model_type,
            num_classes=args.num_classes,
            learning_rate=args.lr,
            optimizer=args.optimizer,
            weight_decay=args.weight_decay,
        )
    elif modality == "hsi":
        classifier = HSIClassifier(
            model_type=args.model_type,
            num_bands=args.num_bands,
            num_classes=args.num_classes,
            learning_rate=args.lr,
            optimizer=args.optimizer,
            weight_decay=args.weight_decay,
        )
    elif modality == "lidar":
        classifier = LiDARClassifier(
            model_type=args.model_type,
            num_classes=args.num_classes,
            learning_rate=args.lr,
            optimizer=args.optimizer,
            weight_decay=args.weight_decay,
        )

    # Set class weights
    class_weights = datamodule.get_class_weights()
    if class_weights is not None:
        classifier.class_weights = class_weights

    # Set up callbacks
    callbacks = [
        ModelCheckpoint(
            monitor="val/accuracy",
            mode="max",
            save_top_k=1,
            filename=f"{modality}-best",
        ),
        EarlyStopping(monitor="val/loss", patience=8, mode="min"),
    ]

    # Create trainer
    trainer = L.Trainer(
        max_epochs=args.epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        callbacks=callbacks,
        enable_progress_bar=False,  # Less clutter
        log_every_n_steps=50,
        enable_model_summary=False,
    )

    # Update datamodule for this modality
    datamodule.modalities = [modality]
    datamodule.setup()

    # Train
    trainer.fit(classifier, datamodule)

    # Test
    test_results = trainer.test(classifier, datamodule, ckpt_path="best", verbose=False)

    return {
        "test_accuracy": test_results[0]["test/accuracy"],
        "test_f1": test_results[0]["test/f1"],
        "test_loss": test_results[0]["test/loss"],
        "best_model_path": trainer.checkpoint_callback.best_model_path,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compare modalities for tree classification"
    )

    # Data arguments
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument(
        "--modalities",
        nargs="+",
        default=["rgb", "hsi", "lidar"],
        choices=["rgb", "hsi", "lidar"],
        help="Modalities to compare",
    )

    # Model arguments
    parser.add_argument("--model_type", type=str, default="simple")
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--num_bands", type=int, default=426)

    # Training arguments
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--optimizer", type=str, default="adamw")
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    # Data split arguments
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.15)

    args = parser.parse_args()

    print("🔬 NEON Tree Classification - Modality Comparison")
    print(f"📁 Data: {args.data_dir}")
    print(f"🧪 Comparing: {', '.join(args.modalities)}")
    print(f"🎯 Classes: {args.num_classes}, Epochs: {args.epochs}")

    # Create base datamodule (will be updated for each modality)
    datamodule = NeonCrownDataModule(
        csv_path=args.csv_path,
        base_data_dir=args.data_dir,
        modalities=["rgb"],  # Will be updated
        split_method="random",
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=1 - args.train_ratio - args.val_ratio,
        split_seed=args.split_seed,
        batch_size=args.batch_size,
        num_workers=4,
        pin_memory=True,
    )

    # Train each modality
    results = {}
    for modality in args.modalities:
        try:
            results[modality] = train_modality(modality, datamodule, args)
            print(
                f"✅ {modality.upper()}: {results[modality]['test_accuracy']:.3f} accuracy"
            )
        except Exception as e:
            print(f"❌ {modality.upper()} failed: {e}")
            results[modality] = {"error": str(e)}

    # Print comparison
    print("\n" + "=" * 60)
    print("📊 FINAL RESULTS")
    print("=" * 60)

    for modality in args.modalities:
        if "error" not in results[modality]:
            res = results[modality]
            print(
                f"{modality.upper():>6}: Acc={res['test_accuracy']:.3f} | F1={res['test_f1']:.3f} | Loss={res['test_loss']:.3f}"
            )
        else:
            print(f"{modality.upper():>6}: ERROR - {results[modality]['error']}")

    # Find best modality
    valid_results = {k: v for k, v in results.items() if "error" not in v}
    if valid_results:
        best_modality = max(
            valid_results.keys(), key=lambda k: valid_results[k]["test_accuracy"]
        )
        print(
            f"\n🏆 Best modality: {best_modality.upper()} ({valid_results[best_modality]['test_accuracy']:.3f} accuracy)"
        )

    # Save results
    results_file = Path("modality_comparison_results.json")
    with open(results_file, "w") as f:
        json.dump({"args": vars(args), "results": results}, f, indent=2)
    print(f"💾 Results saved to: {results_file}")


if __name__ == "__main__":
    main()
