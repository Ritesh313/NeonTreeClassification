"""
Setup script for the NEON Tree Classification package.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="neon-tree-classification",
    version="0.1.0",
    author="Ritesh Chowdhry",
    description="A package for tree species classification using NEON remote sensing data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: GIS",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.9.0", 
        "pytorch-lightning>=1.5.0",
        "torchmetrics>=0.6.0",
        "geopandas>=0.10.0",
        "pandas>=1.3.0",
        "numpy>=1.21.0",
        "rasterio>=1.2.0",
        "shapely>=1.8.0",
        "pyproj>=3.2.0",
        "h5py>=3.1.0",
        "opencv-python>=4.5.0",
        "rpy2>=3.4.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=21.0",
            "flake8>=3.9",
            "jupyter>=1.0.0",
        ],
    },
)
