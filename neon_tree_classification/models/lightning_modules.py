"""
PyTorch Lightning modules for training tree classification models.
"""

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from typing import Dict, Any, Optional, Tuple
import torchmetrics
from .architectures import HsiPixelClassifier, MultiModalClassifier, CNNClassifier


class HsiClassificationModule(pl.LightningModule):
    """
    Lightning module for hyperspectral image classification.
    """
    
    def __init__(self,
                 input_dim: int,
                 num_classes: int,
                 encoding_dim: int = 32,
                 learning_rate: float = 1e-3,
                 reconstruction_weight: float = 1.0,
                 classification_weight: float = 1.0,
                 optimizer: str = 'adam',
                 scheduler: str = 'plateau'):
        """
        Initialize HSI classification module.
        
        Args:
            input_dim: Number of hyperspectral bands
            num_classes: Number of tree species classes
            encoding_dim: Dimension of encoded representation
            learning_rate: Learning rate for optimization
            reconstruction_weight: Weight for reconstruction loss
            classification_weight: Weight for classification loss
            optimizer: Optimizer type ('adam' or 'sgd')
            scheduler: Scheduler type ('plateau' or 'cosine')
        """
        super().__init__()
        self.save_hyperparameters()
        
        self.model = HsiPixelClassifier(input_dim, num_classes, encoding_dim)
        
        # Metrics
        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.test_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        
        self.train_f1 = torchmetrics.F1Score(task="multiclass", num_classes=num_classes, average='weighted')
        self.val_f1 = torchmetrics.F1Score(task="multiclass", num_classes=num_classes, average='weighted')
        self.test_f1 = torchmetrics.F1Score(task="multiclass", num_classes=num_classes, average='weighted')
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the model."""
        return self.model(x)
    
    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step."""
        x, y = batch
        reconstructed, predictions = self.forward(x)
        
        # Calculate losses
        reconstruction_loss = F.mse_loss(reconstructed, x)
        classification_loss = F.cross_entropy(predictions, y)
        
        # Combined loss
        total_loss = (self.hparams.reconstruction_weight * reconstruction_loss + 
                     self.hparams.classification_weight * classification_loss)
        
        # Update metrics
        self.train_acc(predictions, y)
        self.train_f1(predictions, y)
        
        # Log metrics
        self.log('train/loss', total_loss, on_step=True, on_epoch=True)
        self.log('train/reconstruction_loss', reconstruction_loss, on_epoch=True)
        self.log('train/classification_loss', classification_loss, on_epoch=True)
        self.log('train/accuracy', self.train_acc, on_epoch=True)
        self.log('train/f1', self.train_f1, on_epoch=True)
        
        return total_loss
    
    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Validation step."""
        x, y = batch
        reconstructed, predictions = self.forward(x)
        
        # Calculate losses
        reconstruction_loss = F.mse_loss(reconstructed, x)
        classification_loss = F.cross_entropy(predictions, y)
        total_loss = (self.hparams.reconstruction_weight * reconstruction_loss + 
                     self.hparams.classification_weight * classification_loss)
        
        # Update metrics
        self.val_acc(predictions, y)
        self.val_f1(predictions, y)
        
        # Log metrics
        self.log('val/loss', total_loss, on_epoch=True)
        self.log('val/reconstruction_loss', reconstruction_loss, on_epoch=True)
        self.log('val/classification_loss', classification_loss, on_epoch=True)
        self.log('val/accuracy', self.val_acc, on_epoch=True)
        self.log('val/f1', self.val_f1, on_epoch=True)
        
        return total_loss
    
    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Test step."""
        x, y = batch
        reconstructed, predictions = self.forward(x)
        
        # Calculate losses
        reconstruction_loss = F.mse_loss(reconstructed, x)
        classification_loss = F.cross_entropy(predictions, y)
        total_loss = (self.hparams.reconstruction_weight * reconstruction_loss + 
                     self.hparams.classification_weight * classification_loss)
        
        # Update metrics
        self.test_acc(predictions, y)
        self.test_f1(predictions, y)
        
        # Log metrics
        self.log('test/loss', total_loss)
        self.log('test/reconstruction_loss', reconstruction_loss)
        self.log('test/classification_loss', classification_loss)
        self.log('test/accuracy', self.test_acc)
        self.log('test/f1', self.test_f1)
        
        return total_loss
    
    def configure_optimizers(self) -> Dict[str, Any]:
        """Configure optimizers and schedulers."""
        if self.hparams.optimizer == 'adam':
            optimizer = Adam(self.parameters(), lr=self.hparams.learning_rate)
        elif self.hparams.optimizer == 'sgd':
            optimizer = SGD(self.parameters(), lr=self.hparams.learning_rate, momentum=0.9)
        else:
            raise ValueError(f"Unknown optimizer: {self.hparams.optimizer}")
        
        if self.hparams.scheduler == 'plateau':
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'monitor': 'val/loss',
                    'frequency': 1
                }
            }
        elif self.hparams.scheduler == 'cosine':
            scheduler = CosineAnnealingLR(optimizer, T_max=100)
            return [optimizer], [scheduler]
        else:
            return optimizer


class MultiModalClassificationModule(pl.LightningModule):
    """
    Lightning module for multi-modal tree classification.
    """
    
    def __init__(self,
                 hsi_dim: int,
                 rgb_channels: int = 3,
                 lidar_channels: int = 1,
                 num_classes: int = 10,
                 fusion_method: str = 'concatenate',
                 learning_rate: float = 1e-3,
                 optimizer: str = 'adam'):
        """
        Initialize multi-modal classification module.
        
        Args:
            hsi_dim: Number of hyperspectral bands
            rgb_channels: Number of RGB channels
            lidar_channels: Number of LiDAR channels
            num_classes: Number of tree species classes
            fusion_method: How to combine modalities
            learning_rate: Learning rate for optimization
            optimizer: Optimizer type
        """
        super().__init__()
        self.save_hyperparameters()
        
        self.model = MultiModalClassifier(
            hsi_dim=hsi_dim,
            rgb_channels=rgb_channels,
            lidar_channels=lidar_channels,
            num_classes=num_classes,
            fusion_method=fusion_method
        )
        
        # Metrics
        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.test_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        
        self.train_f1 = torchmetrics.F1Score(task="multiclass", num_classes=num_classes, average='weighted')
        self.val_f1 = torchmetrics.F1Score(task="multiclass", num_classes=num_classes, average='weighted')
        self.test_f1 = torchmetrics.F1Score(task="multiclass", num_classes=num_classes, average='weighted')
    
    def forward(self, hsi: torch.Tensor, rgb: torch.Tensor, lidar: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model."""
        return self.model(hsi, rgb, lidar)
    
    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], 
                     batch_idx: int) -> torch.Tensor:
        """Training step."""
        hsi, rgb, lidar, y = batch
        predictions = self.forward(hsi, rgb, lidar)
        
        loss = F.cross_entropy(predictions, y)
        
        # Update metrics
        self.train_acc(predictions, y)
        self.train_f1(predictions, y)
        
        # Log metrics
        self.log('train/loss', loss, on_step=True, on_epoch=True)
        self.log('train/accuracy', self.train_acc, on_epoch=True)
        self.log('train/f1', self.train_f1, on_epoch=True)
        
        return loss
    
    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], 
                       batch_idx: int) -> torch.Tensor:
        """Validation step."""
        hsi, rgb, lidar, y = batch
        predictions = self.forward(hsi, rgb, lidar)
        
        loss = F.cross_entropy(predictions, y)
        
        # Update metrics
        self.val_acc(predictions, y)
        self.val_f1(predictions, y)
        
        # Log metrics
        self.log('val/loss', loss, on_epoch=True)
        self.log('val/accuracy', self.val_acc, on_epoch=True)
        self.log('val/f1', self.val_f1, on_epoch=True)
        
        return loss
    
    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], 
                 batch_idx: int) -> torch.Tensor:
        """Test step."""
        hsi, rgb, lidar, y = batch
        predictions = self.forward(hsi, rgb, lidar)
        
        loss = F.cross_entropy(predictions, y)
        
        # Update metrics
        self.test_acc(predictions, y)
        self.test_f1(predictions, y)
        
        # Log metrics
        self.log('test/loss', loss)
        self.log('test/accuracy', self.test_acc)
        self.log('test/f1', self.test_f1)
        
        return loss
    
    def configure_optimizers(self):
        """Configure optimizers."""
        if self.hparams.optimizer == 'adam':
            return Adam(self.parameters(), lr=self.hparams.learning_rate)
        elif self.hparams.optimizer == 'sgd':
            return SGD(self.parameters(), lr=self.hparams.learning_rate, momentum=0.9)
        else:
            raise ValueError(f"Unknown optimizer: {self.hparams.optimizer}")


class CNNClassificationModule(pl.LightningModule):
    """
    Lightning module for CNN-based RGB classification.
    """
    
    def __init__(self,
                 input_channels: int = 3,
                 num_classes: int = 10,
                 dropout_rate: float = 0.3,
                 learning_rate: float = 1e-3,
                 optimizer: str = 'adam'):
        """
        Initialize CNN classification module.
        
        Args:
            input_channels: Number of input channels
            num_classes: Number of tree species classes
            dropout_rate: Dropout rate for regularization
            learning_rate: Learning rate for optimization
            optimizer: Optimizer type
        """
        super().__init__()
        self.save_hyperparameters()
        
        self.model = CNNClassifier(
            input_channels=input_channels,
            num_classes=num_classes,
            dropout_rate=dropout_rate
        )
        
        # Metrics
        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.test_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        
        self.train_f1 = torchmetrics.F1Score(task="multiclass", num_classes=num_classes, average='weighted')
        self.val_f1 = torchmetrics.F1Score(task="multiclass", num_classes=num_classes, average='weighted')
        self.test_f1 = torchmetrics.F1Score(task="multiclass", num_classes=num_classes, average='weighted')
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model."""
        return self.model(x)
    
    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step."""
        x, y = batch
        predictions = self.forward(x)
        
        loss = F.cross_entropy(predictions, y)
        
        # Update metrics
        self.train_acc(predictions, y)
        self.train_f1(predictions, y)
        
        # Log metrics
        self.log('train/loss', loss, on_step=True, on_epoch=True)
        self.log('train/accuracy', self.train_acc, on_epoch=True)
        self.log('train/f1', self.train_f1, on_epoch=True)
        
        return loss
    
    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Validation step."""
        x, y = batch
        predictions = self.forward(x)
        
        loss = F.cross_entropy(predictions, y)
        
        # Update metrics
        self.val_acc(predictions, y)
        self.val_f1(predictions, y)
        
        # Log metrics
        self.log('val/loss', loss, on_epoch=True)
        self.log('val/accuracy', self.val_acc, on_epoch=True)
        self.log('val/f1', self.val_f1, on_epoch=True)
        
        return loss
    
    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Test step."""
        x, y = batch
        predictions = self.forward(x)
        
        loss = F.cross_entropy(predictions, y)
        
        # Update metrics
        self.test_acc(predictions, y)
        self.test_f1(predictions, y)
        
        # Log metrics
        self.log('test/loss', loss)
        self.log('test/accuracy', self.test_acc)
        self.log('test/f1', self.test_f1)
        
        return loss
    
    def configure_optimizers(self):
        """Configure optimizers."""
        if self.hparams.optimizer == 'adam':
            return Adam(self.parameters(), lr=self.hparams.learning_rate)
        elif self.hparams.optimizer == 'sgd':
            return SGD(self.parameters(), lr=self.hparams.learning_rate, momentum=0.9)
        else:
            raise ValueError(f"Unknown optimizer: {self.hparams.optimizer}")
