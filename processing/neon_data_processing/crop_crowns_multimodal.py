#!/usr/bin/env python3
"""
Multi-modal Crown Cropping Script for NEON Data

Crops tree crowns from NEON tiles across RGB, LiDAR, and HSI modalities.
Designed for the curated NEON tiles directory structure with rgb/, lidar/, and hsi_tif/ subdirectories.
Compatible with hybrid filenames that preserve original NEON metadata: SITE_YEAR_<original_neon_name>.ext

Features:
- Site-specific UTM coordinate transformations
- Multi-modal cropping with proper CRS handling
- Flexible output organization (flat or modality subdirectories)
- Comprehensive metadata logging
- Robust error handling and progress tracking
- Support for hybrid NEON filenames with preserved metadata

Usage:
    python crop_crowns_multimodal.py --tiles_dir /path/to/curated_tiles --crowns_gpkg /path/to/crowns.gpkg --output_dir /path/to/output --modality_subdir

Author: NEON Tree Classification Team
Date: August 2025
"""

import os
import sys
import argparse
import glob
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import warnings
from datetime import datetime

import pandas as pd
import geopandas as gpd
import numpy as np
import rasterio
from rasterio.mask import mask
from shapely.geometry import box
from tqdm import tqdm

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)


def extract_coordinates(filename):
    """
    Extract coordinates from NEON tile filenames.
    Handles the actual NEON file naming conventions:
    - RGB: 2019_BART_5_318000_4879000_image.tif
    - HSI: NEON_D01_BART_DP3_318000_4880000_reflectance.(h5|tif)
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
        r"NEON_D\d+_[A-Z]{4}_DP3_(\d+)_(\d+)_(?:bidirectional_)?reflectance\.(h5|tif)$",
        basename,
    )
    if hsi_match:
        return int(hsi_match.group(1)), int(hsi_match.group(2))  # easting, northing

    # LiDAR pattern: NEON_D##_SITE_DP3_EASTING_NORTHING_CHM.tif
    lidar_match = re.search(r"NEON_D\d+_[A-Z]{4}_DP3_(\d+)_(\d+)_CHM\.tif$", basename)
    if lidar_match:
        return int(lidar_match.group(1)), int(lidar_match.group(2))  # easting, northing

    return None


def extract_tile_info_from_filename(
    filename: str,
) -> Tuple[Optional[str], Optional[int], Optional[int], Optional[int]]:
    """
    Extract site, year, x, y coordinates from hybrid NEON tile filename.

    Expected format: SITE_YEAR_<original_neon_name>.tif

    Args:
        filename: Tile filename

    Returns:
        Tuple of (site, year, x, y) or (None, None, None, None) if parsing fails
    """
    basename = os.path.splitext(os.path.basename(filename))[0]

    # Step 1: Extract site_year prefix from hybrid format
    parts = basename.split("_")
    if len(parts) < 3:  # Must have at least SITE_YEAR_...
        return None, None, None, None

    try:
        site = parts[0]  # "BART"
        year = int(parts[1])  # 2019

        # Step 2: Reconstruct original NEON filename for coordinate extraction
        original_part = "_".join(parts[2:])  # "2019_BART_5_318000_4879000_image"
        original_filename = (
            original_part + os.path.splitext(filename)[1]
        )  # Add back extension

        # Step 3: Use local coordinate extraction function
        coords = extract_coordinates(original_filename)
        if coords:
            x, y = coords
            return site, year, x, y

    except (ValueError, IndexError):
        return None, None, None, None

    return None, None, None, None


def create_tile_inventory(
    tiles_dir: str,
) -> Dict[Tuple[str, int, int, int], Dict[str, str]]:
    """
    Create an inventory of all available tiles across modalities.

    Args:
        tiles_dir: Base directory containing rgb/, lidar/, hsi_tif/ subdirectories

    Returns:
        Dict mapping (site, year, x, y) to modality file paths
    """
    print("🔍 Creating tile inventory...")

    inventory = {}
    # Updated modality patterns for hybrid naming with NEON conventions
    modality_patterns = {
        "rgb": {
            "subdir": "rgb",
            "pattern": "*_image.tif",
        },  # NEON RGB files end with _image.tif
        "lidar": {
            "subdir": "lidar",
            "pattern": "*_CHM.tif",
        },  # NEON LiDAR files end with _CHM.tif
        "hsi": {
            "subdir": "hsi_tif",
            "pattern": "*reflectance.tif",
        },  # Converted HSI files end with reflectance.tif (includes bidirectional_reflectance.tif)
    }

    total_files = 0

    for modality, config in modality_patterns.items():
        modality_dir = os.path.join(tiles_dir, config["subdir"])

        if not os.path.exists(modality_dir):
            print(f"⚠️ Warning: {modality} directory not found: {modality_dir}")
            continue

        # Find files using NEON naming patterns
        pattern = os.path.join(modality_dir, config["pattern"])
        files = glob.glob(pattern)

        print(f"  Found {len(files)} {modality.upper()} files")
        total_files += len(files)

        for file_path in files:
            site, year, x, y = extract_tile_info_from_filename(file_path)

            if site is not None:
                key = (site, year, x, y)
                if key not in inventory:
                    inventory[key] = {}
                inventory[key][modality] = file_path

    # Summary statistics
    print(f"\n📊 Tile Inventory Summary:")
    print(f"  Total files found: {total_files}")
    print(f"  Unique tile locations: {len(inventory)}")

    # Count by modality completeness
    complete_tiles = 0
    modality_counts = {mod: 0 for mod in modality_patterns.keys()}

    for tile_data in inventory.values():
        available_mods = set(tile_data.keys())
        for mod in modality_counts.keys():
            if mod in available_mods:
                modality_counts[mod] += 1

        if len(available_mods) == len(modality_patterns):
            complete_tiles += 1

    print(f"  Complete tiles (all 3 modalities): {complete_tiles}")
    for mod, count in modality_counts.items():
        print(f"  {mod.upper()} tiles: {count}")

    return inventory


def get_site_utm_crs(site: str) -> str:
    """
    Get the appropriate UTM CRS for a NEON site.

    Args:
        site: NEON site code

    Returns:
        UTM CRS code as string (e.g., 'EPSG:32610')
    """
    SITE_UTM_ZONES = {
        "ABBY": "EPSG:32610",
        "BART": "EPSG:32619",
        "BONA": "EPSG:32606",
        "CLBJ": "EPSG:32614",
        "DEJU": "EPSG:32606",
        "DELA": "EPSG:32616",
        "GRSM": "EPSG:32617",
        "GUAN": "EPSG:32619",
        "HARV": "EPSG:32618",
        "HEAL": "EPSG:32606",
        "JERC": "EPSG:32616",
        "KONZ": "EPSG:32614",
        "LENO": "EPSG:32616",
        "MLBS": "EPSG:32617",
        "MOAB": "EPSG:32612",
        "NIWO": "EPSG:32613",
        "ONAQ": "EPSG:32612",
        "OSBS": "EPSG:32617",
        "PUUM": "EPSG:32605",
        "RMNP": "EPSG:32613",
        "SCBI": "EPSG:32617",
        "SERC": "EPSG:32618",
        "SJER": "EPSG:32611",
        "SOAP": "EPSG:32611",
        "SRER": "EPSG:32612",
        "TALL": "EPSG:32616",
        "TEAK": "EPSG:32611",
        "UKFS": "EPSG:32615",
        "UNDE": "EPSG:32616",
        "WREF": "EPSG:32610",
    }
    return SITE_UTM_ZONES.get(site, "EPSG:32610")  # Default to zone 10N


def create_crown_tile_mapping(
    crown_gdf: gpd.GeoDataFrame, tile_inventory: Dict
) -> Dict[Tuple[str, int, int, int], List[int]]:
    """
    Create mapping from tiles to crown indices using spatial indexing.
    Handles coordinate system transformations using correct UTM zones per site.

    Args:
        crown_gdf: GeoDataFrame with crown polygons
        tile_inventory: Dictionary of tile information

    Returns:
        Dict mapping tile keys to lists of crown indices
    """
    print("🗺️ Creating crown-tile spatial mapping...")
    print(f"  Original crown CRS: {crown_gdf.crs}")

    tile_to_crowns = {}
    tiles_with_crowns = 0
    total_crown_mappings = 0

    # Group tiles by site for efficient processing
    tiles_by_site = {}
    for tile_key, tile_data in tile_inventory.items():
        site, year, x, y = tile_key
        if site not in tiles_by_site:
            tiles_by_site[site] = []
        tiles_by_site[site].append((tile_key, tile_data))

    # Assume 1km x 1km tiles (standard NEON tile size)
    tile_size = 1000

    # Process each site separately
    for site, site_tiles in tiles_by_site.items():
        print(f"  Processing site {site} with {len(site_tiles)} tiles...")

        # Get site-specific UTM CRS
        site_utm_crs = get_site_utm_crs(site)

        # Filter crowns for this site and transform to correct UTM
        site_crowns = crown_gdf[crown_gdf["siteID"] == site].copy()

        if len(site_crowns) == 0:
            continue

        # Transform site crowns to the correct UTM zone
        if str(site_crowns.crs) != site_utm_crs:
            site_crowns_transformed = site_crowns.to_crs(site_utm_crs)
        else:
            site_crowns_transformed = site_crowns.copy()

        # Create spatial index for this site's crowns
        crown_sindex = site_crowns_transformed.sindex

        # Process tiles for this site
        for tile_key, tile_data in tqdm(site_tiles, desc=f"Processing {site} tiles"):
            site, year, x, y = tile_key

            # Create tile bounding box in site UTM CRS
            tile_bbox = box(x, y, x + tile_size, y + tile_size)

            # Query spatial index for potential crown intersections
            possible_matches = list(crown_sindex.intersection(tile_bbox.bounds))

            if possible_matches:
                # Filter crowns by year and actual intersection
                actual_matches = []

                for local_crown_idx in possible_matches:
                    crown = site_crowns_transformed.iloc[local_crown_idx]
                    crown_year = crown.get("year", 0)

                    # Check year match and actual geometric intersection
                    # Handle both string and int year comparisons
                    crown_year_int = (
                        int(crown_year) if isinstance(crown_year, str) else crown_year
                    )
                    if crown_year_int == year and crown.geometry.intersects(tile_bbox):
                        # Get original crown index from the full GDF
                        original_crown_idx = (
                            crown.name
                        )  # This preserves the original index
                        actual_matches.append(original_crown_idx)

                if actual_matches:
                    tile_to_crowns[tile_key] = actual_matches
                    tiles_with_crowns += 1
                    total_crown_mappings += len(actual_matches)

    print(f"\n📍 Crown-Tile Mapping Summary:")
    print(f"  Tiles with crowns: {tiles_with_crowns}/{len(tile_inventory)}")
    print(f"  Total crown mappings: {total_crown_mappings}")

    # Distribution of crowns per tile
    if tile_to_crowns:
        crown_counts = [len(crowns) for crowns in tile_to_crowns.values()]
        print(
            f"  Crowns per tile - Min: {min(crown_counts)}, Max: {max(crown_counts)}, Avg: {np.mean(crown_counts):.1f}"
        )

    return tile_to_crowns


def crop_crown_from_raster(
    raster_path: str,
    crown_geometry,
    crown_id: str,
    output_dir: str,
    modality_subdir: bool = False,
    buffer_meters: float = 2.0,
    min_size_pixels: int = 5,
) -> Optional[Dict[str, Any]]:
    """
    Crop a single crown from a raster file.

    Args:
        raster_path: Path to the raster file
        crown_geometry: Crown polygon geometry
        crown_id: Unique identifier for the crown (includes site_year prefix)
        output_dir: Base output directory
        modality_subdir: Whether to organize by modality subdirectories
        buffer_meters: Buffer around crown in meters
        min_size_pixels: Minimum size in pixels for valid crop

    Returns:
        Dict with crop metadata or None if failed
    """
    try:
        with rasterio.open(raster_path) as src:
            # Get the modality from filename using NEON naming conventions
            modality = None
            if "_image.tif" in raster_path:  # NEON RGB files
                modality = "rgb"
            elif "_CHM.tif" in raster_path:  # NEON LiDAR files
                modality = "lidar"
            elif "_reflectance.tif" in raster_path:  # Converted HSI files
                modality = "hsi"
            else:
                modality = "unknown"

            # Calculate buffer in pixels
            pixel_size_x = src.transform[0]
            pixel_size_y = abs(src.transform[4])
            buffer_pixels_x = buffer_meters / pixel_size_x
            buffer_pixels_y = buffer_meters / pixel_size_y

            # Buffer the crown geometry
            buffered_geom = crown_geometry.buffer(
                max(buffer_pixels_x * pixel_size_x, buffer_pixels_y * pixel_size_y)
            )

            # Crop the raster
            try:
                out_image, out_transform = mask(
                    src, [buffered_geom], crop=True, nodata=src.nodata
                )

                # Check if crop is large enough
                if (
                    out_image.shape[1] < min_size_pixels
                    or out_image.shape[2] < min_size_pixels
                ):
                    return None

                # Determine output path - either in modality subdir or flat
                if modality_subdir:
                    modality_dir = os.path.join(output_dir, modality)
                    os.makedirs(modality_dir, exist_ok=True)
                    output_filename = f"{crown_id}.tif"
                    output_path = os.path.join(modality_dir, output_filename)
                else:
                    os.makedirs(output_dir, exist_ok=True)
                    output_filename = f"{crown_id}_{modality}.tif"
                    output_path = os.path.join(output_dir, output_filename)

                # Write the cropped image
                out_meta = src.meta.copy()
                out_meta.update(
                    {
                        "driver": "GTiff",
                        "height": out_image.shape[1],
                        "width": out_image.shape[2],
                        "transform": out_transform,
                        "compress": "lzw",
                    }
                )

                with rasterio.open(output_path, "w", **out_meta) as dst:
                    dst.write(out_image)

                return {
                    "crown_id": crown_id,
                    "modality": modality,
                    "output_path": output_path,
                    "width": out_image.shape[2],
                    "height": out_image.shape[1],
                    "bands": out_image.shape[0],
                    "success": True,
                    "processing_time": datetime.now().isoformat(),
                }

            except ValueError:
                # This can happen if crown doesn't intersect the raster
                return None

    except Exception as e:
        print(f"❌ Error cropping {crown_id} from {raster_path}: {e}")
        return None


def crop_crown_from_open_raster(
    src: rasterio.DatasetReader,
    crown_geometry,
    crown_id: str,
    modality: str,
    output_dir: str,
    modality_subdir: bool = False,
    buffer_meters: float = 2.0,
    min_size_pixels: int = 5,
) -> Optional[Dict[str, Any]]:
    """
    Crop a single crown from an already-open raster file.
    OPTIMIZED version that works with an open rasterio dataset.

    Args:
        src: Open rasterio dataset reader
        crown_geometry: Crown polygon geometry
        crown_id: Unique identifier for the crown (includes site_year prefix)
        modality: Modality name (rgb, lidar, hsi)
        output_dir: Base output directory
        modality_subdir: Whether to organize by modality subdirectories
        buffer_meters: Buffer around crown in meters
        min_size_pixels: Minimum size in pixels for valid crop

    Returns:
        Dict with crop metadata or None if failed
    """
    try:
        # Calculate buffer in pixels
        pixel_size_x = src.transform[0]
        pixel_size_y = abs(src.transform[4])
        buffer_pixels_x = buffer_meters / pixel_size_x
        buffer_pixels_y = buffer_meters / pixel_size_y

        # Buffer the crown geometry
        buffered_geom = crown_geometry.buffer(
            max(buffer_pixels_x * pixel_size_x, buffer_pixels_y * pixel_size_y)
        )

        # Crop the raster
        try:
            out_image, out_transform = mask(
                src, [buffered_geom], crop=True, nodata=src.nodata
            )

            # Check if crop is large enough
            if (
                out_image.shape[1] < min_size_pixels
                or out_image.shape[2] < min_size_pixels
            ):
                return None

            # Determine output path - either in modality subdir or flat
            if modality_subdir:
                modality_dir = os.path.join(output_dir, modality)
                os.makedirs(modality_dir, exist_ok=True)
                output_filename = f"{crown_id}.tif"
                output_path = os.path.join(modality_dir, output_filename)
            else:
                os.makedirs(output_dir, exist_ok=True)
                output_filename = f"{crown_id}_{modality}.tif"
                output_path = os.path.join(output_dir, output_filename)

            # Write the cropped image
            out_meta = src.meta.copy()
            out_meta.update(
                {
                    "driver": "GTiff",
                    "height": out_image.shape[1],
                    "width": out_image.shape[2],
                    "transform": out_transform,
                    "compress": "lzw",
                }
            )

            with rasterio.open(output_path, "w", **out_meta) as dst:
                dst.write(out_image)

            return {
                "crown_id": crown_id,
                "modality": modality,
                "output_path": output_path,
                "width": out_image.shape[2],
                "height": out_image.shape[1],
                "bands": out_image.shape[0],
                "success": True,
                "processing_time": datetime.now().isoformat(),
            }

        except ValueError:
            # This can happen if crown doesn't intersect the raster
            return None

    except Exception as e:
        print(f"❌ Error cropping {crown_id} from open raster: {e}")
        return None


def process_crowns_in_tile(
    tile_key: Tuple[str, int, int, int],
    crown_indices: List[int],
    crown_gdf: gpd.GeoDataFrame,
    tile_inventory: Dict,
    output_base_dir: str,
    modality_subdir: bool = False,
    buffer_meters: float = 2.0,
) -> Dict[str, Any]:
    """
    Process all crowns in a single tile across all available modalities.
    OPTIMIZED: Opens each raster file only once per tile, not once per crown.

    Args:
        tile_key: Tuple of (site, year, x, y)
        crown_indices: List of crown indices to process
        crown_gdf: GeoDataFrame with crown data (will be transformed to correct UTM)
        tile_inventory: Dictionary of tile paths
        output_base_dir: Base output directory
        modality_subdir: Whether to organize outputs by modality subdirectories
        buffer_meters: Buffer around crowns in meters

    Returns:
        Dictionary with processing results
    """
    site, year, x, y = tile_key
    tile_data = tile_inventory[tile_key]

    results = {
        "tile_key": tile_key,
        "crowns_processed": 0,
        "successful_crops": [],
        "failed_crops": [],
        "modalities_processed": set(),
    }

    # Get the correct UTM CRS for this site
    target_crs = get_site_utm_crs(site)

    # Transform crowns to correct site UTM CRS if needed
    if str(crown_gdf.crs) != target_crs:
        crown_gdf_transformed = crown_gdf.to_crs(target_crs)
    else:
        crown_gdf_transformed = crown_gdf.copy()

    # OPTIMIZATION: Process by modality to open each raster file only once
    for modality, raster_path in tile_data.items():
        try:
            with rasterio.open(raster_path) as src:
                # Get the modality from filename using NEON naming conventions
                if "_image.tif" in raster_path:  # NEON RGB files
                    modality_name = "rgb"
                elif "_CHM.tif" in raster_path:  # NEON LiDAR files
                    modality_name = "lidar"
                elif "_reflectance.tif" in raster_path:  # Converted HSI files
                    modality_name = "hsi"
                else:
                    modality_name = "unknown"

                results["modalities_processed"].add(modality_name)

                # Process ALL crowns for this modality with the open raster
                for crown_idx in crown_indices:
                    crown = crown_gdf_transformed.loc[crown_idx]

                    # Create unique crown ID with site_year prefix
                    crown_individual = crown.get(
                        "individual", crown.get("individualID", f"crown_{crown_idx}")
                    )
                    # Format: SITE_YEAR_individualID_crownidx (e.g., HARV_2019_NEON.PLA.D01.HARV.12345_1234)
                    crown_id = f"{site}_{year}_{crown_individual}_{crown_idx}"

                    # Crop from the already-open raster
                    crop_result = crop_crown_from_open_raster(
                        src=src,
                        crown_geometry=crown.geometry,
                        crown_id=crown_id,
                        modality=modality_name,
                        output_dir=output_base_dir,
                        modality_subdir=modality_subdir,
                        buffer_meters=buffer_meters,
                    )

                    if crop_result:
                        # Add crown metadata to the result
                        crop_result.update(
                            {
                                "site": site,
                                "year": year,
                                "tile_x": x,
                                "tile_y": y,
                                "crown_idx": crown_idx,
                                "individual": crown_individual,
                            }
                        )
                        results["successful_crops"].append(crop_result)
                    else:
                        results["failed_crops"].append(
                            {
                                "crown_id": crown_id,
                                "modality": modality_name,
                                "reason": "crop_failed",
                            }
                        )

        except Exception as e:
            print(f"❌ Error opening raster {raster_path}: {e}")
            # Mark all crowns as failed for this modality
            for crown_idx in crown_indices:
                crown = crown_gdf_transformed.loc[crown_idx]
                crown_individual = crown.get(
                    "individual", crown.get("individualID", f"crown_{crown_idx}")
                )
                crown_id = f"{site}_{year}_{crown_individual}_{crown_idx}"
                results["failed_crops"].append(
                    {
                        "crown_id": crown_id,
                        "modality": modality,
                        "reason": f"raster_open_failed: {str(e)}",
                    }
                )

    results["crowns_processed"] = len(crown_indices)
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Crop tree crowns from multi-modal NEON tiles",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage (flat structure with site_year_individual_modality.tif naming)
    python crop_crowns_multimodal.py --tiles_dir /path/to/curated_tiles --crowns_gpkg /path/to/crowns.gpkg --output_dir /path/to/output

    # With modality subdirectories (rgb/, lidar/, hsi/)
    python crop_crowns_multimodal.py --tiles_dir /path/to/curated_tiles --crowns_gpkg /path/to/crowns.gpkg --output_dir /path/to/output --modality_subdir

    # With custom buffer
    python crop_crowns_multimodal.py --tiles_dir /path/to/curated_tiles --crowns_gpkg /path/to/crowns.gpkg --output_dir /path/to/output --buffer 3.0

    # Filter by site with flat structure
    python crop_crowns_multimodal.py --tiles_dir /path/to/curated_tiles --crowns_gpkg /path/to/crowns.gpkg --output_dir /path/to/output --site HARV
        """,
    )

    parser.add_argument(
        "--tiles_dir",
        required=True,
        help="Directory containing rgb/, lidar/, hsi_tif/ subdirectories",
    )
    parser.add_argument(
        "--crowns_gpkg",
        required=True,
        help="Path to crown GeoDataFrame (GeoPackage or shapefile)",
    )
    parser.add_argument(
        "--output_dir", required=True, help="Output directory for cropped crowns"
    )
    parser.add_argument(
        "--buffer",
        type=float,
        default=2.0,
        help="Buffer around crowns in meters (default: 2.0)",
    )
    parser.add_argument(
        "--site", type=str, default=None, help="Filter crowns by site (optional)"
    )
    parser.add_argument(
        "--year", type=int, default=None, help="Filter crowns by year (optional)"
    )
    parser.add_argument(
        "--max_crowns",
        type=int,
        default=None,
        help="Maximum number of crowns to process (for testing)",
    )
    parser.add_argument(
        "--modality_subdir",
        action="store_true",
        default=False,
        help="Organize output in modality subdirectories (rgb/, lidar/, hsi/) instead of flat structure",
    )

    args = parser.parse_args()

    # Validate inputs
    if not os.path.exists(args.tiles_dir):
        print(f"❌ Error: Tiles directory does not exist: {args.tiles_dir}")
        sys.exit(1)

    if not os.path.exists(args.crowns_gpkg):
        print(f"❌ Error: Crown GeoDataFrame file does not exist: {args.crowns_gpkg}")
        sys.exit(1)

    print("🌲 Multi-Modal Crown Cropping Pipeline")
    print("=" * 50)
    print(f"Tiles directory: {args.tiles_dir}")
    print(f"Crowns file: {args.crowns_gpkg}")
    print(f"Output directory: {args.output_dir}")
    print(f"Buffer: {args.buffer} meters")
    if args.site:
        print(f"Site filter: {args.site}")
    if args.year:
        print(f"Year filter: {args.year}")
    if args.max_crowns:
        print(f"Max crowns: {args.max_crowns}")

    # Output organization
    if args.modality_subdir:
        print("Output structure: Modality subdirectories (rgb/, lidar/, hsi/)")
    else:
        print("Output structure: Flat with site_year_individual_modality.tif naming")
    print("")

    # Load crown data
    print("🌲 Loading crown data...")
    crown_gdf = gpd.read_file(args.crowns_gpkg)
    print(f"  Loaded {len(crown_gdf)} crowns")

    # Apply filters
    if args.site:
        site_col = "siteID" if "siteID" in crown_gdf.columns else "site"
        crown_gdf = crown_gdf[crown_gdf[site_col] == args.site]
        print(f"  Filtered to {len(crown_gdf)} crowns for site {args.site}")

    if args.year:
        crown_gdf = crown_gdf[crown_gdf["year"] == args.year]
        print(f"  Filtered to {len(crown_gdf)} crowns for year {args.year}")

    if args.max_crowns and len(crown_gdf) > args.max_crowns:
        crown_gdf = crown_gdf.head(args.max_crowns)
        print(f"  Limited to first {args.max_crowns} crowns for testing")

    if len(crown_gdf) == 0:
        print("❌ No crowns remaining after filtering!")
        sys.exit(1)

    # Create tile inventory
    tile_inventory = create_tile_inventory(args.tiles_dir)

    if not tile_inventory:
        print("❌ No tiles found in inventory!")
        sys.exit(1)

    # Create crown-tile mapping
    crown_tile_mapping = create_crown_tile_mapping(crown_gdf, tile_inventory)

    if not crown_tile_mapping:
        print("❌ No crown-tile intersections found!")
        sys.exit(1)

    # Process all tiles with crowns
    print(f"\n🚀 Processing {len(crown_tile_mapping)} tiles with crowns...")

    all_results = []
    total_crowns_processed = 0
    total_successful_crops = 0

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    for tile_key, crown_indices in tqdm(
        crown_tile_mapping.items(), desc="Processing tiles"
    ):
        tile_results = process_crowns_in_tile(
            tile_key=tile_key,
            crown_indices=crown_indices,
            crown_gdf=crown_gdf,
            tile_inventory=tile_inventory,
            output_base_dir=args.output_dir,
            modality_subdir=args.modality_subdir,
            buffer_meters=args.buffer,
        )

        all_results.append(tile_results)
        total_crowns_processed += tile_results["crowns_processed"]
        total_successful_crops += len(tile_results["successful_crops"])

    # Save metadata
    print("\n💾 Saving crop metadata...")

    # Flatten successful crops into DataFrame
    crop_metadata = []
    for result in all_results:
        crop_metadata.extend(result["successful_crops"])

    if crop_metadata:
        metadata_df = pd.DataFrame(crop_metadata)
        metadata_path = os.path.join(args.output_dir, "crop_metadata.csv")
        metadata_df.to_csv(metadata_path, index=False)
        print(f"  Saved metadata to: {metadata_path}")

    # Print final summary
    print("\n" + "=" * 50)
    print("🎉 PROCESSING COMPLETED!")
    print(f"  Tiles processed: {len(crown_tile_mapping)}")
    print(f"  Crowns processed: {total_crowns_processed}")
    print(f"  Successful crops: {total_successful_crops}")

    if crop_metadata:
        # Summary by modality
        modality_counts = metadata_df["modality"].value_counts()
        print("\n📊 Crops by modality:")
        for modality, count in modality_counts.items():
            print(f"  {modality.upper()}: {count}")

    print(f"\n📁 Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()
