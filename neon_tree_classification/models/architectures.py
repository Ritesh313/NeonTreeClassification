"""
Neural network architectures for tree species classification.
"""

import torch
from torch import nn
from typing import Optional, List, Tuple


class HsiPixelClassifier(nn.Module):
    """
    Original hyperspectral pixel classifier with autoencoder architecture.

    This is the existing model from src/models.py, preserved for compatibility.
    """

    def __init__(self, input_dim: int, num_classes: int):
        """
        Initialize the HSI pixel classifier.

        Args:
            input_dim: Number of hyperspectral bands
            num_classes: Number of tree species classes
        """
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes

        # Encoder: compresses hyperspectral data to 32-dimensional representation
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
        )

        self.encoder_activation = nn.ReLU()

        # Decoder: reconstructs original hyperspectral data
        self.decoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim),
        )

        # Classifier: predicts tree species from encoded representation
        self.classifier = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network.

        Args:
            x: Input hyperspectral data [batch_size, input_dim]

        Returns:
            Tuple of (reconstructed_data, class_predictions)
        """
        # Encode
        encoded = self.encoder(x)
        encoded = self.encoder_activation(encoded)

        # Decode (reconstruction)
        reconstructed = self.decoder(encoded)

        # Classify
        predictions = self.classifier(encoded)

        return reconstructed, predictions


class MultiModalClassifier(nn.Module):
    """
    Multi-modal classifier that combines RGB, HSI, and LiDAR data.
    """

    def __init__(
        self,
        hsi_dim: int,
        rgb_channels: int = 3,
        lidar_channels: int = 1,
        num_classes: int = 10,
        fusion_method: str = "concatenate",
    ):
        """
        Initialize multi-modal classifier.

        Args:
            hsi_dim: Number of hyperspectral bands
            rgb_channels: Number of RGB channels (typically 3)
            lidar_channels: Number of LiDAR channels (typically 1 for CHM)
            num_classes: Number of tree species classes
            fusion_method: How to combine modalities ('concatenate', 'attention')
        """
        super().__init__()
        self.fusion_method = fusion_method

        # HSI processing branch
        self.hsi_processor = nn.Sequential(
            nn.Linear(hsi_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
        )

        # RGB processing branch (assuming flattened RGB patches)
        rgb_input_dim = rgb_channels * 32 * 32  # Assuming 32x32 patches
        self.rgb_processor = nn.Sequential(
            nn.Linear(rgb_input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
        )

        # LiDAR processing branch
        lidar_input_dim = lidar_channels * 32 * 32  # Assuming 32x32 patches
        self.lidar_processor = nn.Sequential(
            nn.Linear(lidar_input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
        )

        # Fusion and classification
        if fusion_method == "concatenate":
            fusion_dim = 64 + 64 + 32  # HSI + RGB + LiDAR
            self.classifier = nn.Sequential(
                nn.Linear(fusion_dim, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(64, num_classes),
            )
        elif fusion_method == "attention":
            # TODO: Implement attention-based fusion
            raise NotImplementedError("Attention fusion not yet implemented")

    def forward(
        self, hsi: torch.Tensor, rgb: torch.Tensor, lidar: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through multi-modal network.

        Args:
            hsi: Hyperspectral data [batch_size, hsi_dim]
            rgb: RGB data [batch_size, rgb_channels, height, width]
            lidar: LiDAR data [batch_size, lidar_channels, height, width]

        Returns:
            Class predictions [batch_size, num_classes]
        """
        # Process each modality
        hsi_features = self.hsi_processor(hsi)

        # Flatten spatial modalities
        rgb_flat = rgb.view(rgb.size(0), -1)
        rgb_features = self.rgb_processor(rgb_flat)

        lidar_flat = lidar.view(lidar.size(0), -1)
        lidar_features = self.lidar_processor(lidar_flat)

        # Fuse features
        if self.fusion_method == "concatenate":
            fused_features = torch.cat(
                [hsi_features, rgb_features, lidar_features], dim=1
            )

        # Classify
        predictions = self.classifier(fused_features)
        return predictions


class CNNClassifier(nn.Module):
    """
    Convolutional Neural Network for RGB image classification.
    """

    def __init__(
        self, input_channels: int = 3, num_classes: int = 10, dropout_rate: float = 0.3
    ):
        """
        Initialize CNN classifier.

        Args:
            input_channels: Number of input channels (3 for RGB)
            num_classes: Number of tree species classes
            dropout_rate: Dropout rate for regularization
        """
        super().__init__()

        self.features = nn.Sequential(
            # First conv block
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Second conv block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Third conv block
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Calculate the size of flattened features
        # For 32x32 input: 32 -> 16 -> 8 -> 4, so 256 * 4 * 4 = 4096
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through CNN.

        Args:
            x: Input RGB data [batch_size, channels, height, width]

        Returns:
            Class predictions [batch_size, num_classes]
        """
        features = self.features(x)
        predictions = self.classifier(features)
        return predictions
