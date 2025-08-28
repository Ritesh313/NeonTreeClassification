"""
Hyperspectral imaging (HSI) neural network architectures for tree species classification.

All models expect HSI input as [batch_size, bands, height, width] tensors.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class SimpleHSINet(nn.Module):
    """
    Simple neural network for hyperspectral tree crown classification.

    Treats HSI data as multi-channel images and uses 2D convolutions.
    """

    def __init__(
        self, num_bands: int = 369, num_classes: int = 10, input_size: int = 224
    ):
        """
        Initialize SimpleHSINet.

        Args:
            num_bands: Number of hyperspectral bands
            num_classes: Number of tree species classes
            input_size: Expected input image size (assumed square)
        """
        super().__init__()
        self.num_bands = num_bands
        self.num_classes = num_classes
        self.input_size = input_size

        # Spectral dimension reduction first
        self.spectral_conv = nn.Sequential(
            nn.Conv2d(num_bands, 64, kernel_size=1),  # 1x1 conv for spectral mixing
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        # Spatial feature extraction
        self.spatial_features = nn.Sequential(
            # First block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Second block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Third block
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: HSI tensor [batch_size, bands, height, width]

        Returns:
            Class logits [batch_size, num_classes]
        """
        # Reduce spectral dimensionality
        x = self.spectral_conv(x)

        # Extract spatial features
        features = self.spatial_features(x)

        # Classify
        logits = self.classifier(features)
        return logits

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features for fusion/ensemble models.

        Args:
            x: HSI tensor [batch_size, bands, height, width]

        Returns:
            Feature vector [batch_size, 256]
        """
        x = self.spectral_conv(x)
        features = self.spatial_features(x)
        return features.flatten(1)


class SpectralCNN(nn.Module):
    """
    1D CNN operating along the spectral dimension for pixel-wise classification.

    Processes each pixel's spectrum independently, then aggregates spatially.
    """

    def __init__(
        self, num_bands: int = 369, num_classes: int = 10, input_size: int = 144
    ):
        """
        Initialize SpectralCNN.

        Args:
            num_bands: Number of hyperspectral bands
            num_classes: Number of tree species classes
            input_size: Expected input image size (assumed square)
        """
        super().__init__()
        self.num_bands = num_bands
        self.num_classes = num_classes
        self.input_size = input_size

        # 1D CNN for spectral features (applied per pixel)
        self.spectral_cnn = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )

        # Spatial aggregation
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: HSI tensor [batch_size, bands, height, width]

        Returns:
            Class logits [batch_size, num_classes]
        """
        batch_size, bands, height, width = x.shape

        # Reshape for 1D convolution along spectral dimension
        # [batch_size, bands, height, width] -> [batch_size * height * width, 1, bands]
        x_reshaped = x.permute(0, 2, 3, 1).contiguous()  # [B, H, W, bands]
        x_reshaped = x_reshaped.view(-1, 1, bands)  # [B*H*W, 1, bands]

        # Apply 1D CNN along spectral dimension
        spectral_features = self.spectral_cnn(x_reshaped)  # [B*H*W, 128, 1]
        spectral_features = spectral_features.squeeze(-1)  # [B*H*W, 128]

        # Reshape back to spatial format
        spectral_features = spectral_features.view(batch_size, height, width, 128)
        spectral_features = spectral_features.permute(0, 3, 1, 2)  # [B, 128, H, W]

        # Spatial aggregation
        spatial_features = self.spatial_conv(spectral_features)  # [B, 256, 1, 1]

        # Classification
        logits = self.classifier(spatial_features)
        return logits

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features for fusion/ensemble models.

        Args:
            x: HSI tensor [batch_size, bands, height, width]

        Returns:
            Feature vector [batch_size, 256]
        """
        batch_size, bands, height, width = x.shape

        # Process spectral features
        x_reshaped = x.permute(0, 2, 3, 1).contiguous().view(-1, 1, bands)
        spectral_features = self.spectral_cnn(x_reshaped).squeeze(-1)
        spectral_features = spectral_features.view(batch_size, height, width, 128)
        spectral_features = spectral_features.permute(0, 3, 1, 2)

        # Spatial aggregation
        features = self.spatial_conv(spectral_features)
        return features.flatten(1)


class HyperNet(nn.Module):
    """
    Advanced hyperspectral network with attention mechanisms.

    Uses both spectral and spatial attention for better feature learning.
    """

    def __init__(
        self, num_bands: int = 369, num_classes: int = 10, input_size: int = 224
    ):
        """
        Initialize HyperNet.

        Args:
            num_bands: Number of hyperspectral bands
            num_classes: Number of tree species classes
            input_size: Expected input image size (assumed square)
        """
        super().__init__()
        self.num_bands = num_bands
        self.num_classes = num_classes

        # Spectral attention module
        self.spectral_attention = SpectralAttention(num_bands)

        # Feature extraction after attention
        self.features = nn.Sequential(
            nn.Conv2d(num_bands, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: HSI tensor [batch_size, bands, height, width]

        Returns:
            Class logits [batch_size, num_classes]
        """
        # Apply spectral attention
        x = self.spectral_attention(x)

        # Extract features
        features = self.features(x)

        # Classify
        logits = self.classifier(features)
        return logits

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features for fusion/ensemble models.

        Args:
            x: HSI tensor [batch_size, bands, height, width]

        Returns:
            Feature vector [batch_size, 512]
        """
        x = self.spectral_attention(x)
        features = self.features(x)
        return features.flatten(1)


class SpectralAttention(nn.Module):
    """Spectral attention mechanism for hyperspectral data."""

    def __init__(self, num_bands: int):
        super().__init__()
        self.num_bands = num_bands

        # Global average pooling across spatial dimensions
        self.gap = nn.AdaptiveAvgPool2d(1)

        # Attention network
        self.attention = nn.Sequential(
            nn.Linear(num_bands, num_bands // 4),
            nn.ReLU(inplace=True),
            nn.Linear(num_bands // 4, num_bands),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply spectral attention.

        Args:
            x: HSI tensor [batch_size, bands, height, width]

        Returns:
            Attention-weighted HSI tensor [batch_size, bands, height, width]
        """
        batch_size, bands, height, width = x.shape

        # Global spatial pooling: [B, bands, H, W] -> [B, bands, 1, 1]
        pooled = self.gap(x)

        # Flatten for attention: [B, bands, 1, 1] -> [B, bands]
        pooled = pooled.view(batch_size, bands)

        # Compute attention weights: [B, bands] -> [B, bands]
        attention_weights = self.attention(pooled)

        # Reshape for broadcasting: [B, bands] -> [B, bands, 1, 1]
        attention_weights = attention_weights.view(batch_size, bands, 1, 1)

        # Apply attention weights
        return x * attention_weights


# Factory function for easy model creation
def create_hsi_model(
    model_type: str = "simple", num_bands: int = 369, num_classes: int = 10, **kwargs
) -> nn.Module:
    """
    Factory function to create HSI models.

    Args:
        model_type: Type of model ("simple", "spectral_cnn", "hypernet")
        num_bands: Number of hyperspectral bands
        num_classes: Number of output classes
        **kwargs: Additional model-specific arguments

    Returns:
        HSI classification model
    """
    if model_type == "simple":
        return SimpleHSINet(num_bands=num_bands, num_classes=num_classes, **kwargs)
    elif model_type == "spectral_cnn":
        return SpectralCNN(num_bands=num_bands, num_classes=num_classes, **kwargs)
    elif model_type == "hypernet":
        return HyperNet(num_bands=num_bands, num_classes=num_classes, **kwargs)
    else:
        raise ValueError(f"Unknown HSI model type: {model_type}")
