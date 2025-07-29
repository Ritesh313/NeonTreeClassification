"""
LiDAR-based neural network architectures for tree species classification.

All models expect LiDAR input as [batch_size, 1, height, width] tensors representing height maps.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class SimpleLiDARNet(nn.Module):
    """
    Simple CNN for LiDAR height map tree crown classification.

    Treats LiDAR data as single-channel height images.
    """

    def __init__(self, num_classes: int = 10, input_size: int = 224):
        """
        Initialize SimpleLiDARNet.

        Args:
            num_classes: Number of tree species classes
            input_size: Expected input image size (assumed square)
        """
        super().__init__()
        self.num_classes = num_classes
        self.input_size = input_size

        # Feature extraction layers
        self.features = nn.Sequential(
            # First block
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Second block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Third block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Fourth block
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
            x: LiDAR tensor [batch_size, 1, height, width]

        Returns:
            Class logits [batch_size, num_classes]
        """
        features = self.features(x)
        logits = self.classifier(features)
        return logits

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features for fusion/ensemble models.

        Args:
            x: LiDAR tensor [batch_size, 1, height, width]

        Returns:
            Feature vector [batch_size, 256]
        """
        features = self.features(x)
        return features.flatten(1)


class HeightCNN(nn.Module):
    """
    CNN specialized for height-based features in LiDAR data.

    Includes height-specific processing and multi-scale analysis.
    """

    def __init__(self, num_classes: int = 10, input_size: int = 224):
        """
        Initialize HeightCNN.

        Args:
            num_classes: Number of tree species classes
            input_size: Expected input image size (assumed square)
        """
        super().__init__()
        self.num_classes = num_classes
        self.input_size = input_size

        # Height preprocessing
        self.height_preprocess = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=1),  # Point-wise height transformation
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )

        # Multi-scale feature extraction
        self.scale1 = self._make_scale_branch(16, 64, kernel_size=3)  # Fine details
        self.scale2 = self._make_scale_branch(16, 64, kernel_size=5)  # Medium features
        self.scale3 = self._make_scale_branch(16, 64, kernel_size=7)  # Coarse features

        # Feature fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(192, 256, kernel_size=3, padding=1),  # 64*3 = 192 channels
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

    def _make_scale_branch(self, in_channels: int, out_channels: int, kernel_size: int):
        """Create a scale-specific processing branch."""
        padding = kernel_size // 2
        return nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels // 2, kernel_size=kernel_size, padding=padding
            ),
            nn.BatchNorm2d(out_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels // 2,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: LiDAR tensor [batch_size, 1, height, width]

        Returns:
            Class logits [batch_size, num_classes]
        """
        # Preprocess height data
        x = self.height_preprocess(x)

        # Multi-scale processing
        scale1_features = self.scale1(x)
        scale2_features = self.scale2(x)
        scale3_features = self.scale3(x)

        # Concatenate multi-scale features
        combined = torch.cat([scale1_features, scale2_features, scale3_features], dim=1)

        # Fusion and classification
        features = self.fusion(combined)
        logits = self.classifier(features)
        return logits

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features for fusion/ensemble models.

        Args:
            x: LiDAR tensor [batch_size, 1, height, width]

        Returns:
            Feature vector [batch_size, 256]
        """
        x = self.height_preprocess(x)

        # Multi-scale processing
        scale1_features = self.scale1(x)
        scale2_features = self.scale2(x)
        scale3_features = self.scale3(x)

        # Combine and fuse
        combined = torch.cat([scale1_features, scale2_features, scale3_features], dim=1)
        features = self.fusion(combined)
        return features.flatten(1)


class StructuralCNN(nn.Module):
    """
    CNN that extracts structural features from LiDAR height data.

    Focuses on tree crown shape, height gradients, and structural patterns.
    """

    def __init__(self, num_classes: int = 10, input_size: int = 224):
        """
        Initialize StructuralCNN.

        Args:
            num_classes: Number of tree species classes
            input_size: Expected input image size (assumed square)
        """
        super().__init__()
        self.num_classes = num_classes

        # Height gradient computation
        self.gradient_conv = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
        )

        # Structural feature extraction
        self.structural_features = nn.Sequential(
            # Combined height + gradient processing
            nn.Conv2d(
                9, 32, kernel_size=3, padding=1
            ),  # 1 (height) + 8 (gradients) = 9
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
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
            x: LiDAR tensor [batch_size, 1, height, width]

        Returns:
            Class logits [batch_size, num_classes]
        """
        # Compute height gradients
        gradients = self.gradient_conv(x)

        # Combine original height with gradients
        combined = torch.cat([x, gradients], dim=1)

        # Extract structural features
        features = self.structural_features(combined)

        # Classify
        logits = self.classifier(features)
        return logits

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features for fusion/ensemble models.

        Args:
            x: LiDAR tensor [batch_size, 1, height, width]

        Returns:
            Feature vector [batch_size, 256]
        """
        gradients = self.gradient_conv(x)
        combined = torch.cat([x, gradients], dim=1)
        features = self.structural_features(combined)
        return features.flatten(1)


# Factory function for easy model creation
def create_lidar_model(
    model_type: str = "simple", num_classes: int = 10, **kwargs
) -> nn.Module:
    """
    Factory function to create LiDAR models.

    Args:
        model_type: Type of model ("simple", "height_cnn", "structural")
        num_classes: Number of output classes
        **kwargs: Additional model-specific arguments

    Returns:
        LiDAR classification model
    """
    if model_type == "simple":
        return SimpleLiDARNet(num_classes=num_classes, **kwargs)
    elif model_type == "height_cnn":
        return HeightCNN(num_classes=num_classes, **kwargs)
    elif model_type == "structural":
        return StructuralCNN(num_classes=num_classes, **kwargs)
    else:
        raise ValueError(f"Unknown LiDAR model type: {model_type}")
