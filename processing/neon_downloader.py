"""
NEON data downloader with support for RGB, HSI, and LiDAR data products.

This module works with crown metadata produced by validate_and_extract_crown_metadata.py.
It expects a GeoPackage file with columns: individual, siteID, plotID, year,
center_easting, center_northing, source_file, geometry.

Usage:
    # After running validate_and_extract_crown_metadata.py to create crown_metadata.gpkg
    python neon_downloader.py --coords-file crown_metadata.gpkg --output-dir ./neon_tiles

    # Download specific site/year
    python neon_downloader.py --coords-file crown_metadata.gpkg --site BART --year 2019
"""

import os
import pandas as pd
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri, r
import warnings
import geopandas as gpd
import json
import glob
import re
from datetime import datetime

# Suppress R warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)


class NEONDownloader:
    """
    Downloads NEON airborne data products (RGB, HSI, LiDAR) for specified coordinates.
    """

    # NEON data product codes
    PRODUCTS = {
        "rgb": "DP3.30010.001",  # RGB orthophotos
        "hsi_pre2022": "DP3.30006.001",  # HSI without BRDF correction (pre-2022)
        "hsi_post2022": "DP3.30006.002",  # HSI with BRDF correction (2022+)
        "lidar": "DP3.30015.001",  # LiDAR CHM
    }

    def __init__(self, base_output_dir: str = "/tmp/neon_downloads"):
        """
        Initialize the NEON downloader.

        Args:
            base_output_dir: Base directory for downloaded data
        """
        self.base_output_dir = base_output_dir
        self.download_results = []  # Simple list to track all downloads
        self._setup_r_environment()

    def _setup_r_environment(self):
        """Setup R environment and load required packages."""
        try:
            # Activate pandas2ri for automatic conversion
            pandas2ri.activate()

            # Load required R packages
            r("library(neonUtilities)")
            print("✅ R neonUtilities package loaded successfully")

        except Exception as e:
            print(f"❌ Error setting up R environment: {e}")
            print("Please ensure R and neonUtilities package are installed:")
            print("  R: install.packages('neonUtilities')")
            raise

    def _get_hsi_product_code(self, year: int) -> str:
        """
        Get the appropriate HSI product code based on year.
        NEON changed HSI processing in 2022.
        """
        year = int(year)
        if year >= 2022:
            return self.PRODUCTS["hsi_post2022"]
        else:
            return self.PRODUCTS["hsi_pre2022"]

    def _clean_coordinates(
        self, coordinates_df: pd.DataFrame, site: str
    ) -> pd.DataFrame:
        """
        Basic coordinate validation for downloader input.

        Args:
            coordinates_df: DataFrame with easting/northing coordinates
            site: NEON site code for logging

        Returns:
            Cleaned DataFrame with valid coordinates
        """
        print(f"🧹 Validating coordinates for {site}")
        print(f"Input coordinates: {len(coordinates_df)} entries")

        # Basic sanity check - remove any potential NaN values
        initial_count = len(coordinates_df)
        coordinates_df = coordinates_df.dropna(
            subset=["center_easting", "center_northing"]
        )

        if len(coordinates_df) < initial_count:
            print(
                f"⚠️  Removed {initial_count - len(coordinates_df)} entries with NaN coordinates"
            )

        # Basic validation - ensure coordinates are reasonable UTM values
        valid_mask = (
            (coordinates_df["center_easting"] > 0)
            & (coordinates_df["center_easting"] < 1000000)  # Max UTM easting
            & (coordinates_df["center_northing"] > 0)
            & (coordinates_df["center_northing"] < 10000000)  # Max UTM northing
        )

        coordinates_df = coordinates_df[valid_mask]
        print(f"✅ Valid coordinates: {len(coordinates_df)} entries")

        return coordinates_df

    def _convert_to_tile_coordinates(
        self, coordinates_df: pd.DataFrame
    ) -> List[Tuple[int, int]]:
        """
        Convert center coordinates to NEON tile coordinates (1000m grid).

        NEON tiles are 1km x 1km squares where coordinates represent the
        bottom-left (southwest) corner. Uses floor division to ensure crowns
        are mapped to the correct containing tile.

        Args:
            coordinates_df: DataFrame with center coordinates

        Returns:
            List of unique (easting, northing) tile coordinate pairs
        """
        print(f"📍 Converting to tile coordinates")

        # FIXED: Use floor to map crowns to correct tiles (not round)
        # NEON tiles are 1km x 1km starting at bottom-left corner
        tile_eastings = (
            np.floor(coordinates_df["center_easting"] / 1000).astype(int) * 1000
        )
        tile_northings = (
            np.floor(coordinates_df["center_northing"] / 1000).astype(int) * 1000
        )

        # Get unique coordinate pairs
        tile_coords = list(set(zip(tile_eastings, tile_northings)))

        print(f"Generated {len(tile_coords)} unique tile coordinates")
        if tile_coords:
            print(
                f"Easting range: {min(e for e, n in tile_coords)} to {max(e for e, n in tile_coords)}"
            )
            print(
                f"Northing range: {min(n for e, n in tile_coords)} to {max(n for e, n in tile_coords)}"
            )

        return tile_coords

    def _validate_downloaded_files(
        self,
        output_dir: str,
        modality: str,
        site: str,
        year: int,
        tile_coords: List[Tuple[int, int]],
    ) -> int:
        """
        Simple validation - count how many tiles were actually downloaded.

        Args:
            output_dir: Output directory for this modality
            modality: Data modality
            site: NEON site code
            year: Year
            tile_coords: Expected tile coordinates

        Returns:
            Number of tiles found
        """
        # Define expected file patterns for each modality
        file_patterns = {
            "rgb": f"{year}_{site}_*_*_image.tif",
            "hsi": f"NEON_*_{site}_*_*_*_reflectance.h5",
            "lidar": f"NEON_*_{site}_*_*_*_CHM.tif",
        }

        pattern = file_patterns.get(modality, f"{year}_{site}_*_*_{modality}.*")

        # Search for downloaded files
        search_pattern = os.path.join(output_dir, "**", pattern)
        downloaded_files = glob.glob(search_pattern, recursive=True)

        found_tiles = len(downloaded_files)
        print(f"    Found {found_tiles}/{len(tile_coords)} {modality} tiles")

        return found_tiles

    def _download_modality(
        self,
        product_code: str,
        site: str,
        year: int,
        tile_coords: List[Tuple[int, int]],
        output_dir: str,
        modality: str,
        check_size: bool = False,
    ) -> Dict[str, Any]:
        """
        Download a specific data modality using R neonUtilities.

        Args:
            product_code: NEON product code
            site: NEON site code
            year: Year of data
            tile_coords: List of (easting, northing) coordinates
            output_dir: Output directory
            modality: Data modality name
            check_size: Whether to check file size

        Returns:
            Dictionary with download results
        """
        try:
            print(f"  Downloading {modality.upper()} data...")

            # Prepare coordinates for R
            eastings = [coord[0] for coord in tile_coords]
            northings = [coord[1] for coord in tile_coords]

            # Convert to R vectors
            r_eastings = robjects.FloatVector(eastings)
            r_northings = robjects.FloatVector(northings)

            # Create modality-specific output directory
            modality_output = os.path.join(output_dir, modality)
            os.makedirs(modality_output, exist_ok=True)

            # Assign variables to R environment
            robjects.globalenv["eastings"] = r_eastings
            robjects.globalenv["northings"] = r_northings
            robjects.globalenv["site_code"] = site
            robjects.globalenv["year_val"] = year
            robjects.globalenv["product_code"] = product_code
            robjects.globalenv["output_path"] = modality_output
            robjects.globalenv["check_size"] = check_size

            # R download command
            r_command = """
            tryCatch({
                cat("Downloading with parameters:\n")
                cat("Product:", product_code, "\n")
                cat("Site:", site_code, "\n")
                cat("Year:", year_val, "\n")
                cat("Coordinates:", length(eastings), "tile pairs\n")
                cat("Output:", output_path, "\n")
                cat("Check size:", check_size, "\n")

                byTileAOP(dpID = product_code,
                         site = site_code,
                         year = year_val,
                         easting = eastings,
                         northing = northings,
                         savepath = output_path,
                         include.provisional = TRUE,
                         check.size = check_size)

                cat("Download completed successfully\n")
                return(TRUE)
            }, error = function(e) {
                cat("Error in download:", e$message, "\n")
                return(FALSE)
            })
            """

            print(f"    Executing R download for {len(tile_coords)} tiles...")
            result = r(r_command)
            r_success = bool(result[0])

            # Validate what was actually downloaded
            found_tiles = self._validate_downloaded_files(
                modality_output, modality, site, year, tile_coords
            )

            if r_success:
                print(f"    ✅ R command succeeded")
            else:
                print(f"    ❌ R command failed")

            return {
                "site": site,
                "year": year,
                "modality": modality,
                "product_code": product_code,
                "tiles_requested": len(tile_coords),
                "r_success": r_success,
                "tiles_found": found_tiles,
                "success_rate": (
                    found_tiles / len(tile_coords) if len(tile_coords) > 0 else 0
                ),
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            print(f"    ❌ Exception during {modality} download: {str(e)}")
            return {
                "site": site,
                "year": year,
                "modality": modality,
                "product_code": product_code,
                "tiles_requested": len(tile_coords),
                "r_success": False,
                "tiles_found": 0,
                "success_rate": 0.0,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }

    def download_neon_data(
        self,
        coordinates_df: pd.DataFrame,
        site: str,
        year: int,
        modalities: List[str] = ["rgb", "hsi", "lidar"],
        check_size: bool = False,
    ) -> Dict[str, Any]:
        """
        Download NEON data for specified coordinates and modalities.

        Args:
            coordinates_df: DataFrame with coordinate information
            site: NEON site code
            year: Year of data to download
            modalities: List of data types to download ('rgb', 'hsi', 'lidar')
            check_size: Whether to check file sizes before downloading

        Returns:
            Dictionary with download results and summary
        """
        # Ensure year is always an integer
        try:
            year = int(year)
        except (ValueError, TypeError):
            raise ValueError(
                f"Year must be convertible to int, got: {year} (type: {type(year)})"
            )

        print(f"\n🚀 STARTING DOWNLOAD: {site} {year}")
        print(f"Modalities: {modalities}")

        # Clean coordinates
        cleaned_coords = self._clean_coordinates(coordinates_df, site)
        if len(cleaned_coords) == 0:
            raise ValueError(f"No valid coordinates found for site {site}")

        # Convert to tile coordinates
        tile_coords = self._convert_to_tile_coordinates(cleaned_coords)

        # Setup output directory
        output_dir = os.path.join(self.base_output_dir, f"{site}_{year}")
        os.makedirs(output_dir, exist_ok=True)

        # Download each modality
        for modality in modalities:
            if modality == "hsi":
                product_code = self._get_hsi_product_code(year)
            else:
                product_code = self.PRODUCTS[modality]

            result = self._download_modality(
                product_code=product_code,
                site=site,
                year=year,
                tile_coords=tile_coords,
                output_dir=output_dir,
                modality=modality,
                check_size=check_size,
            )

            # Store result for final logging
            self.download_results.append(result)

        print(f"✅ Completed: {site} {year}")

        return {
            "site": site,
            "year": year,
            "total_tiles": len(tile_coords),
            "output_dir": output_dir,
        }

    def save_final_log(self):
        """Save one comprehensive log file at the very end"""
        if not self.download_results:
            print("No download results to save")
            return

        # Create master log directory
        log_dir = os.path.join(self.base_output_dir, "download_logs")
        os.makedirs(log_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save detailed CSV
        df = pd.DataFrame(self.download_results)
        csv_path = os.path.join(log_dir, f"neon_download_log_{timestamp}.csv")
        df.to_csv(csv_path, index=False)

        # Save detailed JSON
        json_path = os.path.join(log_dir, f"neon_download_log_{timestamp}.json")
        with open(json_path, "w") as f:
            json.dump(self.download_results, f, indent=2, default=str)

        # Create summary report
        report_path = os.path.join(log_dir, f"neon_download_summary_{timestamp}.txt")
        with open(report_path, "w") as f:
            f.write("NEON DOWNLOAD SUMMARY REPORT\n")
            f.write("=" * 50 + "\n\n")

            # Overall statistics
            total_downloads = len(df)
            r_successes = df["r_success"].sum()
            total_tiles_requested = df["tiles_requested"].sum()
            total_tiles_found = df["tiles_found"].sum()

            f.write(f"OVERALL STATISTICS:\n")
            f.write(f"  Total downloads: {total_downloads}\n")
            f.write(
                f"  R command successes: {r_successes}/{total_downloads} ({r_successes/total_downloads:.1%})\n"
            )
            f.write(
                f"  Total tiles found: {total_tiles_found}/{total_tiles_requested} ({total_tiles_found/total_tiles_requested:.1%})\n\n"
            )

            # By modality
            f.write(f"BY MODALITY:\n")
            for modality in ["rgb", "hsi", "lidar"]:
                mod_data = df[df["modality"] == modality]
                if len(mod_data) > 0:
                    mod_r_success = mod_data["r_success"].sum()
                    mod_tiles_req = mod_data["tiles_requested"].sum()
                    mod_tiles_found = mod_data["tiles_found"].sum()
                    f.write(
                        f"  {modality.upper()}: {mod_tiles_found}/{mod_tiles_req} tiles ({mod_tiles_found/mod_tiles_req:.1%}), {mod_r_success} R successes\n"
                    )
            f.write("\n")

            # Failures
            failures = df[df["r_success"] == False]
            if len(failures) > 0:
                f.write(f"FAILED DOWNLOADS ({len(failures)}):\n")
                for _, row in failures.iterrows():
                    error_msg = row.get("error", "R command failed")
                    f.write(
                        f"  {row['site']} {row['year']} {row['modality']}: {error_msg}\n"
                    )
                f.write("\n")

            # Sites with missing tiles
            missing = df[df["tiles_found"] < df["tiles_requested"]]
            if len(missing) > 0:
                f.write(f"PARTIAL DOWNLOADS ({len(missing)}):\n")
                for _, row in missing.iterrows():
                    missing_count = row["tiles_requested"] - row["tiles_found"]
                    f.write(
                        f"  {row['site']} {row['year']} {row['modality']}: {missing_count} missing tiles\n"
                    )

        print(f"\n📋 FINAL LOGS SAVED:")
        print(f"  📊 Summary: {report_path}")
        print(f"  📄 CSV: {csv_path}")
        print(f"  📄 JSON: {json_path}")

        # Print quick summary to console
        print(f"\n🎯 DOWNLOAD SUMMARY:")
        print(f"  Total downloads: {total_downloads}")
        print(
            f"  R successes: {r_successes}/{total_downloads} ({r_successes/total_downloads:.1%})"
        )
        print(
            f"  Tiles found: {total_tiles_found}/{total_tiles_requested} ({total_tiles_found/total_tiles_requested:.1%})"
        )

        return csv_path, json_path, report_path


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Download NEON tiles for crowns."
    )
    parser.add_argument(
        "--coords-file",
        type=str,
        default="crown_metadata.gpkg",
        help="Path to crowns coordinates file (default: crown_metadata.gpkg in current directory)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="neon_tiles",
        help="Output directory for downloaded tiles (default: ./neon_tiles)",
    )
    parser.add_argument(
        "--site", type=str, default=None, help="NEON site code (optional)"
    )
    parser.add_argument("--year", type=int, default=None, help="Year (optional)")
    parser.add_argument(
        "--modalities",
        type=str,
        nargs="*",
        default=["rgb", "hsi", "lidar"],
        help="Modalities to download",
    )
    args = parser.parse_args()

    if not os.path.exists(args.coords_file):
        print(f"Coordinates file not found: {args.coords_file}")
        print("Please run the validate_and_extract_crown_metadata.py script first.")
        return

    coords_df = gpd.read_file(args.coords_file)

    downloader = NEONDownloader(base_output_dir=args.output_dir)

    # If site and year are specified, filter and run for that group only
    if args.site and args.year:
        site_coords = coords_df[
            (coords_df["siteID"] == args.site) & (coords_df["year"] == args.year)
        ]
        if len(site_coords) == 0:
            print(f"No coordinates found for {args.site} {args.year}")
            return
        print(f"Processing {args.site} {args.year} with {len(site_coords)} crowns")
        results = downloader.download_neon_data(
            coordinates_df=site_coords,
            site=args.site,
            year=args.year,
            modalities=args.modalities,
        )
        print(f"Done: {args.site} {args.year} - {results['total_tiles']} tiles")
    else:
        # Process all crowns, grouped by site and year
        for (site, year), group in coords_df.groupby(["siteID", "year"]):
            print(f"Processing {site} {year} with {len(group)} crowns")
            results = downloader.download_neon_data(
                coordinates_df=group,
                site=site,
                year=year,
                modalities=args.modalities,
            )
            print(f"Done: {site} {year} - {results['total_tiles']} tiles")

    # Save final consolidated log
    downloader.save_final_log()


if __name__ == "__main__":
    main()
