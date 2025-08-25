"""
PyTorch Lightning modules for training tree classification models.

Provides base class with common training logic and modality-specific extensions.
"""

import torch
import torch.nn.functional as F
import lightning as L
from torch.optim import Adam, SGD, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, StepLR
from typing import Dict, Any, Optional, Union
import torchmetrics

from .rgb_models import create_rgb_model
from .hsi_models import create_hsi_model
from .lidar_models import create_lidar_model


class BaseTreeClassifier(L.LightningModule):
    """
    Base Lightning module for tree species classification.

    Provides common training logic that works with our clean data format:
    batch = {"modality": tensor, "species_idx": tensor}
    """

    def __init__(
        self,
        model: torch.nn.Module,
        num_classes: int,
        learning_rate: float = 1e-3,
        optimizer: str = "adamw",
        scheduler: str = "plateau",
        weight_decay: float = 1e-4,
        class_weights: Optional[torch.Tensor] = None,
    ):
        """
        Initialize base classifier.

        Args:
            model: PyTorch model (from rgb_models.py, hsi_models.py, etc.)
            num_classes: Number of tree species classes
            learning_rate: Learning rate for optimizer
            optimizer: Optimizer type ('adam', 'adamw', 'sgd')
            scheduler: Scheduler type ('plateau', 'cosine', 'step')
            weight_decay: Weight decay for optimizer
            class_weights: Optional class weights for imbalanced datasets
        """
        super().__init__()
        self.save_hyperparameters(ignore=["model", "class_weights"])

        self.model = model
        self.num_classes = num_classes
        self.class_weights = class_weights

        # Common metrics for all modalities
        self.train_acc = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes
        )
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.test_acc = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes
        )

        self.train_f1 = torchmetrics.F1Score(
            task="multiclass", num_classes=num_classes, average="macro"
        )
        self.val_f1 = torchmetrics.F1Score(
            task="multiclass", num_classes=num_classes, average="macro"
        )
        self.test_f1 = torchmetrics.F1Score(
            task="multiclass", num_classes=num_classes, average="macro"
        )

        # Confusion matrix for validation
        self.val_confmat = torchmetrics.ConfusionMatrix(
            task="multiclass", num_classes=num_classes
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model."""
        return self.model(x)

    def _shared_step(self, batch: Dict[str, torch.Tensor], stage: str):
        """Shared step logic for train/val/test."""
        # Extract labels
        targets = batch["species_idx"]

        # Get modality data (to be implemented by subclasses)
        inputs = self._extract_modality_data(batch)

        # Forward pass
        logits = self.forward(inputs)

        # Compute loss
        if self.class_weights is not None:
            loss = F.cross_entropy(
                logits, targets, weight=self.class_weights.to(self.device)
            )
        else:
            loss = F.cross_entropy(logits, targets)

        # Get predictions
        preds = torch.argmax(logits, dim=1)

        return loss, preds, targets, logits

    def training_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Training step."""
        loss, preds, targets, logits = self._shared_step(batch, "train")

        # Update metrics
        self.train_acc(preds, targets)
        self.train_f1(preds, targets)

        # Log metrics
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/accuracy", self.train_acc, on_epoch=True, prog_bar=True)
        self.log("train/f1", self.train_f1, on_epoch=True)

        return loss

    def validation_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Validation step."""
        loss, preds, targets, logits = self._shared_step(batch, "val")

        # Update metrics
        self.val_acc(preds, targets)
        self.val_f1(preds, targets)
        self.val_confmat(preds, targets)

        # Log metrics
        self.log("val/loss", loss, on_epoch=True, prog_bar=True)
        self.log("val/accuracy", self.val_acc, on_epoch=True, prog_bar=True)
        self.log("val/f1", self.val_f1, on_epoch=True)

        return loss

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Test step."""
        loss, preds, targets, logits = self._shared_step(batch, "test")

        # Update metrics
        self.test_acc(preds, targets)
        self.test_f1(preds, targets)

        # Log metrics
        self.log("test/loss", loss, on_epoch=True)
        self.log("test/accuracy", self.test_acc, on_epoch=True)
        self.log("test/f1", self.test_f1, on_epoch=True)

        return loss

    def configure_optimizers(self):
        """Configure optimizers and schedulers."""
        # Optimizer
        if self.hparams.optimizer == "adam":
            optimizer = Adam(
                self.parameters(),
                lr=self.hparams.learning_rate,
                weight_decay=self.hparams.weight_decay,
            )
        elif self.hparams.optimizer == "adamw":
            optimizer = AdamW(
                self.parameters(),
                lr=self.hparams.learning_rate,
                weight_decay=self.hparams.weight_decay,
            )
        elif self.hparams.optimizer == "sgd":
            optimizer = SGD(
                self.parameters(),
                lr=self.hparams.learning_rate,
                weight_decay=self.hparams.weight_decay,
                momentum=0.9,
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.hparams.optimizer}")

        # Scheduler
        if self.hparams.scheduler == "plateau":
            scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "frequency": 1,
                },
            }
        elif self.hparams.scheduler == "cosine":
            scheduler = CosineAnnealingLR(optimizer, T_max=100)
            return [optimizer], [scheduler]
        elif self.hparams.scheduler == "step":
            scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
            return [optimizer], [scheduler]
        else:
            return optimizer

    def _extract_modality_data(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Extract modality-specific data from batch.

        To be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement _extract_modality_data")


class RGBClassifier(BaseTreeClassifier):
    """
    Lightning module for RGB-based tree species classification.

    Includes RGB-specific evaluation features like image logging.
    """

    def __init__(
        self,
        model_type: str = "simple",
        num_classes: int = 10,
        learning_rate: float = 1e-3,
        optimizer: str = "adamw",
        scheduler: str = "plateau",
        weight_decay: float = 1e-4,
        class_weights: Optional[torch.Tensor] = None,
        log_images: bool = False,
        **model_kwargs,
    ):
        """
        Initialize RGB classifier.

        Args:
            model_type: Type of RGB model ("simple", "resnet")
            num_classes: Number of tree species classes
            learning_rate: Learning rate for optimizer
            optimizer: Optimizer type ('adam', 'adamw', 'sgd')
            scheduler: Scheduler type ('plateau', 'cosine', 'step')
            weight_decay: Weight decay for optimizer
            class_weights: Optional class weights for imbalanced datasets
            log_images: Whether to log sample images during validation
            **model_kwargs: Additional arguments for model creation
        """
        # Create RGB model
        model = create_rgb_model(
            model_type=model_type, num_classes=num_classes, **model_kwargs
        )

        super().__init__(
            model=model,
            num_classes=num_classes,
            learning_rate=learning_rate,
            optimizer=optimizer,
            scheduler=scheduler,
            weight_decay=weight_decay,
            class_weights=class_weights,
        )

        self.log_images = log_images
        self.logged_images_this_epoch = False

    def _extract_modality_data(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Extract RGB data from batch."""
        return batch["rgb"]

    def validation_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Validation step with optional image logging."""
        loss = super().validation_step(batch, batch_idx)

        # Log sample images periodically
        if (
            self.log_images
            and not self.logged_images_this_epoch
            and batch_idx == 0
            and self.trainer.current_epoch % 10 == 0
        ):
            self._log_sample_images(batch)
            self.logged_images_this_epoch = True

        return loss

    def on_validation_epoch_start(self):
        """Reset image logging flag at start of validation epoch."""
        self.logged_images_this_epoch = False

    def _log_sample_images(self, batch: Dict[str, torch.Tensor]):
        """Log sample RGB images with predictions."""
        if not hasattr(self.logger, "experiment"):
            return

        # Get a few samples
        rgb_images = batch["rgb"][:4]  # First 4 images
        targets = batch["species_idx"][:4]

        # Get predictions
        with torch.no_grad():
            logits = self.forward(rgb_images)
            preds = torch.argmax(logits, dim=1)

        # Normalize images for display (assuming they're in [0, 1] range)
        images_np = rgb_images.cpu().numpy()

        # Log to comet/tensorboard (implementation depends on logger)
        try:
            from lightning.pytorch.loggers import CometLogger

            if isinstance(self.logger, CometLogger):
                for i in range(len(images_np)):
                    img = images_np[i].transpose(1, 2, 0)  # CHW -> HWC
                    caption = f"True: {targets[i].item()}, Pred: {preds[i].item()}"
                    self.logger.experiment.log_image(
                        img,
                        name=f"val_image_{i}_epoch_{self.trainer.current_epoch}",
                        step=self.trainer.current_epoch,
                    )
        except ImportError:
            pass  # comet not available


class HSIClassifier(BaseTreeClassifier):
    """
    Lightning module for hyperspectral-based tree species classification.

    Includes HSI-specific evaluation features like spectral analysis.
    """

    def __init__(
        self,
        model_type: str = "simple",
        num_bands: int = 426,
        num_classes: int = 10,
        learning_rate: float = 1e-3,
        optimizer: str = "adamw",
        scheduler: str = "plateau",
        weight_decay: float = 1e-4,
        class_weights: Optional[torch.Tensor] = None,
        **model_kwargs,
    ):
        """
        Initialize HSI classifier.

        Args:
            model_type: Type of HSI model ("simple", "spectral_cnn", "hypernet")
            num_bands: Number of hyperspectral bands
            num_classes: Number of tree species classes
            learning_rate: Learning rate for optimizer
            optimizer: Optimizer type ('adam', 'adamw', 'sgd')
            scheduler: Scheduler type ('plateau', 'cosine', 'step')
            weight_decay: Weight decay for optimizer
            class_weights: Optional class weights for imbalanced datasets
            **model_kwargs: Additional arguments for model creation
        """
        # Create HSI model
        model = create_hsi_model(
            model_type=model_type,
            num_bands=num_bands,
            num_classes=num_classes,
            **model_kwargs,
        )

        super().__init__(
            model=model,
            num_classes=num_classes,
            learning_rate=learning_rate,
            optimizer=optimizer,
            scheduler=scheduler,
            weight_decay=weight_decay,
            class_weights=class_weights,
        )

        self.num_bands = num_bands

    def _extract_modality_data(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Extract HSI data from batch."""
        return batch["hsi"]

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Test step with HSI-specific analysis."""
        loss = super().test_step(batch, batch_idx)

        # Add HSI-specific metrics if needed
        # TODO: Add spectral band importance analysis
        # TODO: Add spectral confusion analysis

        return loss


class LiDARClassifier(BaseTreeClassifier):
    """
    Lightning module for LiDAR-based tree species classification.

    Includes LiDAR-specific evaluation features like height analysis.
    """

    def __init__(
        self,
        model_type: str = "simple",
        num_classes: int = 10,
        learning_rate: float = 1e-3,
        optimizer: str = "adamw",
        scheduler: str = "plateau",
        weight_decay: float = 1e-4,
        class_weights: Optional[torch.Tensor] = None,
        **model_kwargs,
    ):
        """
        Initialize LiDAR classifier.

        Args:
            model_type: Type of LiDAR model ("simple", "height_cnn", "structural")
            num_classes: Number of tree species classes
            learning_rate: Learning rate for optimizer
            optimizer: Optimizer type ('adam', 'adamw', 'sgd')
            scheduler: Scheduler type ('plateau', 'cosine', 'step')
            weight_decay: Weight decay for optimizer
            class_weights: Optional class weights for imbalanced datasets
            **model_kwargs: Additional arguments for model creation
        """
        # Create LiDAR model
        model = create_lidar_model(
            model_type=model_type, num_classes=num_classes, **model_kwargs
        )

        super().__init__(
            model=model,
            num_classes=num_classes,
            learning_rate=learning_rate,
            optimizer=optimizer,
            scheduler=scheduler,
            weight_decay=weight_decay,
            class_weights=class_weights,
        )

    def _extract_modality_data(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Extract LiDAR data from batch."""
        return batch["lidar"]

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Test step with LiDAR-specific analysis."""
        loss = super().test_step(batch, batch_idx)

        # Add LiDAR-specific metrics if needed
        # TODO: Add height distribution analysis
        # TODO: Add structural pattern analysis

        return loss
