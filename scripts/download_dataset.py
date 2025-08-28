"""
Download and extract NEON Tree Classification dataset from Dropbox.

Usage:
    python download_dataset.py
"""

import zipfile
from pathlib import Path
from urllib.request import urlretrieve

# Your Dropbox dataset URL (with dl=1)
DATASET_URL = "https://www.dropbox.com/scl/fi/v49xi6d7wtetctqphebx0/neon_tree_classification_dataset.zip?rlkey=fb7bz6kd0ckip4u0qd5xdor58&st=dvjyd5ry&dl=1"


def main():
    # Create dataset directory
    dataset_dir = Path("_neon_tree_classification_dataset_files")
    dataset_dir.mkdir(exist_ok=True)

    zip_path = dataset_dir / "dataset.zip"

    print(f"Downloading dataset...")
    print(f"URL: {DATASET_URL}")

    try:
        # Download with progress
        def progress(block_num, block_size, total_size):
            if total_size > 0:
                percent = min(100, (block_num * block_size / total_size) * 100)
                print(f"\rProgress: {percent:.1f}%", end="", flush=True)

        urlretrieve(DATASET_URL, zip_path, progress)
        print(f"\nDownloaded {zip_path.stat().st_size / 1024 / 1024:.1f} MB")

        # Extract
        print("Extracting...")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(dataset_dir)

        # Clean up zip
        zip_path.unlink()

        print(f"Dataset ready in: {dataset_dir.absolute()}")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
