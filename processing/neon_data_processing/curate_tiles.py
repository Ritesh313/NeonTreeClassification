"""
NEON Tile Curation Tool

This module processes downloaded NEON tiles from their nested directory structure
and creates a flattened, organized structure suitable for machine learning workflows.

Key Features:
- Handles complex NEON directory nesting (DP3.../neon-aop-products/year/...)
- Matches tiles across RGB, HSI, and LiDAR modalities by coordinates
- Creates hybrid filenames: SITE_YEAR_<original_neon_name>.ext (preserves NEON metadata)
- Option to organize by modality subdirectories or completely flat structure
- Only processes complete tile sets (all 3 modalities present)

Input Structure:
  downloaded_tiles/
  ├── SITE_YEAR/
  │   ├── rgb/DP3.../.../*.tif
  │   ├── hsi/DP3.../.../*.h5
  │   └── lidar/DP3.../.../*.tif

Output Structure (organized):
  curated_tiles/
  ├── rgb/SITE_YEAR_2019_SITE_5_EASTING_NORTHING_image.tif
  ├── hsi/SITE_YEAR_NEON_D01_SITE_DP3_EASTING_NORTHING_reflectance.h5
  └── lidar/SITE_YEAR_NEON_D01_SITE_DP3_EASTING_NORTHING_CHM.tif

Usage:
  python curate_tiles.py --input-dir downloaded_tiles/ --output-dir curated_tiles/
"""

import shutil
import os
import re


def flatten_tiles_inventory(
    all_tile_inventory, output_dir, delete_originals=False, organize_by_modality=True
):
    """
    Copy or move all complete modality tiles into a flat directory with hybrid filenames.
    Only includes coordinate pairs with all 3 modalities.

    Args:
        all_tile_inventory: Dictionary with site_year as keys and tile inventories as values
        output_dir: Output directory for flattened tiles
        delete_originals: If True, original files are deleted after moving
        organize_by_modality: If True, create subdirectories for each modality
    """
    os.makedirs(output_dir, exist_ok=True)

    # Create modality subdirectories if requested
    if organize_by_modality:
        for modality in ["rgb", "hsi", "lidar"]:
            os.makedirs(os.path.join(output_dir, modality), exist_ok=True)

    total_complete_sets = 0
    total_files_processed = 0

    print(f"\\n📁 FLATTENING TILES TO: {output_dir}")
    print("=" * 60)

    for site_year, tile_inventory in all_tile_inventory.items():
        print(f"\\n Processing {site_year}...")
        site, year = site_year.split("_", 1)  # Split on first underscore only

        complete_sets = 0
        for (x, y), mods in tile_inventory.items():
            # Only process tiles with all 3 modalities
            if all(mod in mods for mod in ["rgb", "hsi", "lidar"]):
                complete_sets += 1

                for mod, source_path in mods.items():
                    # Get original file extension
                    ext = os.path.splitext(source_path)[1]

                    # new_filename = f"{site}_{year}_{x}_{y}_{mod}{ext}" # old file naming method
                    # Create hybrid filename: SITE_YEAR_<original_neon_name>
                    original_filename = os.path.basename(source_path)
                    original_basename = os.path.splitext(original_filename)[0]
                    new_filename = f"{site}_{year}_{original_basename}{ext}"

                    # Determine output path (with or without modality subdirectory)
                    if organize_by_modality:
                        output_path = os.path.join(output_dir, mod, new_filename)
                    else:
                        output_path = os.path.join(output_dir, new_filename)

                    # Copy or move the file
                    if not os.path.exists(output_path):
                        try:
                            if delete_originals:
                                shutil.move(source_path, output_path)
                                action = "moved"
                            else:
                                shutil.copy2(source_path, output_path)
                                action = "copied"
                            total_files_processed += 1

                        except Exception as e:
                            print(f"❌ Error processing {source_path}: {e}")
                    else:
                        print(f"⚠️  Skipping {new_filename} - already exists")

        print(f"  ✅ {site_year}: {complete_sets} complete tile sets processed")
        total_complete_sets += complete_sets

    print(f"\\n🎉 FLATTENING COMPLETE!")
    print(f"  📊 Total complete tile sets: {total_complete_sets}")
    print(f"  📁 Total files processed: {total_files_processed}")
    print(f"  📂 Output directory: {output_dir}")

    if organize_by_modality:
        print(f"  🗂️  Files organized in subdirectories: rgb/, hsi/, lidar/")

    return total_complete_sets, total_files_processed


def extract_coordinates(filename):
    """
    Extract coordinates from NEON tile filenames.
    Handles the actual NEON file naming conventions:
    - RGB: 2019_BART_5_318000_4879000_image.tif
    - HSI: NEON_D01_BART_DP3_318000_4880000_reflectance.h5
    - LiDAR: NEON_D01_BART_DP3_319000_4881000_CHM.tif
    """
    basename = os.path.basename(filename)

    # RGB pattern: YYYY_SITE_#_EASTING_NORTHING_image.tif
    rgb_match = re.search(r"(\d{4})_([A-Z]{4})_\d+_(\d+)_(\d+)_image\.tif$", basename)
    if rgb_match:
        return int(rgb_match.group(3)), int(rgb_match.group(4))  # easting, northing

    # HSI pattern: NEON_D##_SITE_DP3_EASTING_NORTHING_(bidirectional_)reflectance.h5
    # Handles both pre-2022 (reflectance.h5) and post-2022 (bidirectional_reflectance.h5) naming
    hsi_match = re.search(
        r"NEON_D\d+_[A-Z]{4}_DP3_(\d+)_(\d+)_(?:bidirectional_)?reflectance\.h5$",
        basename,
    )
    if hsi_match:
        return int(hsi_match.group(1)), int(hsi_match.group(2))  # easting, northing

    # LiDAR pattern: NEON_D##_SITE_DP3_EASTING_NORTHING_CHM.tif
    lidar_match = re.search(r"NEON_D\d+_[A-Z]{4}_DP3_(\d+)_(\d+)_CHM\.tif$", basename)
    if lidar_match:
        return int(lidar_match.group(1)), int(lidar_match.group(2))  # easting, northing

    return None


def organize_downloaded_tiles(tiles_base_dir):
    """
    Organize tiles by coordinate pairs across all modalities.
    Handles the deep NEON directory structure:
    site_year/modality/DP3.../neon-aop-products/year/FullSite/...

    Args:
        tiles_base_dir: Base directory containing modality subdirectories (e.g., BART_2019/)

    Returns:
        dict: Nested dictionary with coordinate pairs as keys
    """
    print(f"ORGANIZING DOWNLOADED TILES for {os.path.basename(tiles_base_dir)}")
    print("=" * 50)

    tile_inventory = {}

    # Define modality directories and their file extensions
    modality_config = {
        "rgb": {"subdir": "rgb", "extension": ".tif", "pattern": "image.tif"},
        "hsi": {"subdir": "hsi", "extension": ".h5", "pattern": "reflectance.h5"},
        "lidar": {"subdir": "lidar", "extension": ".tif", "pattern": "CHM.tif"},
    }

    modality_files = {}

    for modality, config in modality_config.items():
        modality_dir = os.path.join(tiles_base_dir, config["subdir"])
        files = []

        if os.path.exists(modality_dir):
            # Walk through the deep directory structure
            for root, dirs, filenames in os.walk(modality_dir):
                for filename in filenames:
                    if filename.endswith(config["pattern"]):
                        files.append(os.path.join(root, filename))

        modality_files[modality] = files
        print(f"Found {len(files)} {modality.upper()} files")

    # Debug: print sample filenames
    for modality, files in modality_files.items():
        if files:
            print(f"Sample {modality.upper()} filename: {os.path.basename(files[0])}")

    # Process each modality and extract coordinates
    for modality, files in modality_files.items():
        coords_found = set()

        for file_path in files:
            coords = extract_coordinates(file_path)
            if coords:
                coords_found.add(coords)
                x, y = coords
                if (x, y) not in tile_inventory:
                    tile_inventory[(x, y)] = {}
                tile_inventory[(x, y)][modality] = file_path
            else:
                print(
                    f"⚠️  Could not extract coordinates from: {os.path.basename(file_path)}"
                )

        print(f"  {modality.upper()}: {len(coords_found)} unique coordinate pairs")

    # Print summary
    print(f"\nTILE INVENTORY SUMMARY:")
    print(f"Total unique coordinate pairs: {len(tile_inventory)}")

    # Count by modality
    modality_counts = {}
    for modality in modality_config.keys():
        count = sum(1 for tile_data in tile_inventory.values() if modality in tile_data)
        modality_counts[modality] = count
        print(f"  {modality.upper()} tiles: {count}")

    # Count complete tiles (all 3 modalities)
    complete_tiles = sum(
        1
        for tile_data in tile_inventory.values()
        if all(mod in tile_data for mod in ["rgb", "hsi", "lidar"])
    )
    print(f"  Complete tiles (all 3 modalities): {complete_tiles}")

    # Show coordinate overlaps
    rgb_coords = {coords for coords, data in tile_inventory.items() if "rgb" in data}
    hsi_coords = {coords for coords, data in tile_inventory.items() if "hsi" in data}
    lidar_coords = {
        coords for coords, data in tile_inventory.items() if "lidar" in data
    }

    overlap = rgb_coords & hsi_coords & lidar_coords
    print(f"  Coordinates with all modalities: {len(overlap)}")

    if overlap:
        print(f"  Sample complete coordinates: {list(overlap)[:3]}")

    return tile_inventory


def organize_all_sites_tiles(parent_dir):
    """
    Organize tiles for all site_year folders in a parent directory.
    Returns a consolidated tile inventory for all sites/years.

    Args:
        parent_dir: Directory containing site_year subdirectories

    Returns:
        dict: Dictionary with site_year keys and tile inventories as values
    """
    print(f"🔍 Scanning for site_year folders in {parent_dir}")

    if not os.path.exists(parent_dir):
        print(f"❌ Parent directory does not exist: {parent_dir}")
        return {}

    all_tile_inventory = {}
    site_year_folders = []

    # Find all site_year directories
    for entry in os.listdir(parent_dir):
        site_year_path = os.path.join(parent_dir, entry)
        if (
            os.path.isdir(site_year_path) and "_" in entry
        ):  # Basic check for SITE_YEAR format
            site_year_folders.append((entry, site_year_path))

    print(f"Found {len(site_year_folders)} potential site_year folders")

    for entry, site_year_path in site_year_folders:
        print(f"\\n📂 Processing {entry}...")
        try:
            tile_inventory = organize_downloaded_tiles(site_year_path)
            if tile_inventory:  # Only add if we found tiles
                all_tile_inventory[entry] = tile_inventory
            else:
                print(f"⚠️  No tiles found in {entry}")
        except Exception as e:
            print(f"❌ Error processing {entry}: {e}")

    print(f"\\n✅ Successfully processed {len(all_tile_inventory)} site_year folders")
    return all_tile_inventory


# --- Main block for CLI usage ---
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Curate NEON tiles into a flat directory structure. "
        "Processes deeply nested NEON downloads and creates hybrid filenames that preserve NEON metadata."
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Parent directory containing site_year folders (e.g., downloaded_neon_tiles_0818/)",
    )
    parser.add_argument(
        "--output-dir", required=True, help="Directory to store flattened curated tiles"
    )
    parser.add_argument(
        "--delete-originals",
        action="store_true",
        help="Delete original files after moving (default: copy files)",
    )
    parser.add_argument(
        "--flat-structure",
        action="store_true",
        help="Create completely flat structure without modality subdirectories",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be processed without actually moving/copying files",
    )

    args = parser.parse_args()

    print(f"🔍 NEON Tile Curation Tool")
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Action: {'MOVE' if args.delete_originals else 'COPY'}")
    print(f"Structure: {'FLAT' if args.flat_structure else 'ORGANIZED BY MODALITY'}")
    print(f"Mode: {'DRY RUN' if args.dry_run else 'EXECUTE'}")
    print("=" * 60)

    if not os.path.exists(args.input_dir):
        print(f"❌ Input directory does not exist: {args.input_dir}")
        exit(1)

    # Scan and organize all tiles
    print(f"🔍 Scanning for site_year folders in {args.input_dir}")
    all_tile_inventory = organize_all_sites_tiles(args.input_dir)

    if not all_tile_inventory:
        print("❌ No tile inventories found. Check your input directory structure.")
        exit(1)

    # Print summary before processing
    total_sites = len(all_tile_inventory)
    total_complete = sum(
        sum(
            1
            for tile_data in inventory.values()
            if all(mod in tile_data for mod in ["rgb", "hsi", "lidar"])
        )
        for inventory in all_tile_inventory.values()
    )

    print(f"\\n📊 PROCESSING SUMMARY:")
    print(f"  Sites/years found: {total_sites}")
    print(f"  Complete tile sets: {total_complete}")

    if args.dry_run:
        print("\\n🧪 DRY RUN - No files will be moved/copied")
        for site_year, inventory in all_tile_inventory.items():
            complete = sum(
                1
                for tile_data in inventory.values()
                if all(mod in tile_data for mod in ["rgb", "hsi", "lidar"])
            )
            print(f"  {site_year}: {complete} complete tile sets")
    else:
        # Process the tiles
        complete_sets, files_processed = flatten_tiles_inventory(
            all_tile_inventory,
            args.output_dir,
            delete_originals=args.delete_originals,
            organize_by_modality=not args.flat_structure,
        )

        print(f"\\n✅ Processing completed successfully!")
        print(f"Processed {complete_sets} complete tile sets ({files_processed} files)")

        if not args.delete_originals:
            print(f"\\n💡 Original files preserved. To delete them later, run:")
            print(
                f"   python {__file__} --input-dir {args.input_dir} --output-dir {args.output_dir} --delete-originals"
            )
