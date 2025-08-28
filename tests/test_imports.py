"""
Test basic imports and functionality.
"""

import pytest
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_main_package_import():
    """Test that the main package can be imported."""
    import neon_tree_classification

    assert neon_tree_classification.__version__ == "0.1.0"


def test_dataset_import():
    """Test that the dataset class can be imported."""
    from neon_tree_classification.core.dataset import NeonCrownDataset

    assert NeonCrownDataset is not None


def test_datamodule_import():
    """Test that the datamodule class can be imported."""
    from neon_tree_classification.core.datamodule import NeonCrownDataModule

    assert NeonCrownDataModule is not None


def test_model_imports():
    """Test that model architectures can be imported."""
    from neon_tree_classification.models.rgb_models import create_rgb_model
    from neon_tree_classification.models.hsi_models import create_hsi_model
    from neon_tree_classification.models.lidar_models import create_lidar_model

    assert create_rgb_model is not None
    assert create_hsi_model is not None
    assert create_lidar_model is not None


def test_lightning_modules_import():
    """Test that Lightning modules can be imported."""
    from neon_tree_classification.models.lightning_modules import (
        BaseTreeClassifier,
        RGBClassifier,
        HSIClassifier,
        LiDARClassifier,
    )

    assert BaseTreeClassifier is not None
    assert RGBClassifier is not None
    assert HSIClassifier is not None
    assert LiDARClassifier is not None


def test_visualization_import():
    """Test that visualization functions can be imported."""
    from neon_tree_classification.core.visualization import (
        plot_rgb,
        plot_hsi,
        plot_hsi_pca,
        plot_hsi_spectra,
        plot_lidar,
    )

    assert plot_rgb is not None
    assert plot_hsi is not None
    assert plot_hsi_pca is not None
    assert plot_hsi_spectra is not None
    assert plot_lidar is not None


def test_dataloader_script_import():
    """Test that the dataloader utility script can be imported."""
    from scripts.get_dataloaders import get_dataloaders

    assert get_dataloaders is not None


def test_package_all_exports():
    """Test that __all__ exports work correctly."""
    import neon_tree_classification

    # Test that major components are accessible
    assert hasattr(neon_tree_classification, "NeonCrownDataset")
    assert hasattr(neon_tree_classification, "NeonCrownDataModule")
    assert hasattr(neon_tree_classification, "RGBClassifier")
    assert hasattr(neon_tree_classification, "HSIClassifier")
    assert hasattr(neon_tree_classification, "LiDARClassifier")
    assert hasattr(neon_tree_classification, "create_rgb_model")
    assert hasattr(neon_tree_classification, "create_hsi_model")
    assert hasattr(neon_tree_classification, "create_lidar_model")


def test_torch_compatibility():
    """Test that torch imports work with the versions specified."""
    import torch
    import lightning

    # Check minimum version requirements
    torch_version = torch.__version__.split(".")
    lightning_version = lightning.__version__.split(".")

    assert int(torch_version[0]) >= 2, f"PyTorch version {torch.__version__} is too old"
    assert (
        int(lightning_version[0]) >= 2
    ), f"Lightning version {lightning.__version__} is too old"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
