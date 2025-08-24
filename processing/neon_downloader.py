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
from rpy2.robjects.conversion import localconverter
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
        self.download_logs = {}  # Track download attempts and results
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

    def _get_expected_neon_file_patterns(
        self, modality: str, site: str, year: int, tile_coords: List[Tuple[int, int]]
    ) -> Dict[str, List[str]]:
        """
        Generate expected NEON file patterns for validation.
        Accounts for the nested NEON directory structure.

        Args:
            modality: Data modality ('rgb', 'hsi', 'lidar')
            site: NEON site code
            year: Year of data
            tile_coords: List of (easting, northing) tile coordinates

        Returns:
            Dict mapping tile coordinates to expected file patterns
        """
        expected_files = {}

        for x, y in tile_coords:
            patterns = []

            if modality == "rgb":
                # RGB: .../L3/Camera/Mosaic/YYYY_SITE_5_XXXXXX_YYYYYY_image.tif
                pattern = f"**/**/L3/Camera/Mosaic/{year}_{site}_*_{x}_{y}_image.tif"
                patterns.append(pattern)

            elif modality == "hsi":
                # HSI: .../L3/Spectrometer/Reflectance/NEON_D??_SITE_DP3_XXXXXX_YYYYYY_reflectance.h5
                pattern = f"**/**/L3/Spectrometer/Reflectance/NEON_*_{site}_DP3_{x}_{y}_reflectance.h5"
                patterns.append(pattern)

            elif modality == "lidar":
                # LiDAR: .../L3/DiscreteLidar/CanopyHeightModelGtif/NEON_D??_SITE_DP3_XXXXXX_YYYYYY_CHM.tif
                pattern = f"**/**/L3/DiscreteLidar/CanopyHeightModelGtif/NEON_*_{site}_DP3_{x}_{y}_CHM.tif"
                patterns.append(pattern)

            expected_files[f"{x}_{y}"] = patterns

        return expected_files

    def _validate_downloaded_files(
        self,
        output_dir: str,
        modality: str,
        site: str,
        year: int,
        tile_coords: List[Tuple[int, int]],
    ) -> Dict[str, Any]:
        """
        Validate that expected files were actually downloaded.
        Searches through the NEON nested directory structure.

        Args:
            output_dir: Output directory for this modality
            modality: Data modality
            site: NEON site code
            year: Year
            tile_coords: Expected tile coordinates

        Returns:
            Dict with validation results
        """
        import glob

        validation = {
            "modality": modality,
            "site": site,
            "year": year,
            "expected_tiles": len(tile_coords),
            "found_tiles": 0,
            "missing_tiles": [],
            "found_files": {},
            "validation_time": pd.Timestamp.now().isoformat(),
        }

        expected_patterns = self._get_expected_neon_file_patterns(
            modality, site, year, tile_coords
        )

        for x, y in tile_coords:
            tile_key = f"{x}_{y}"
            found_file = None

            # Search for files matching the expected patterns
            for pattern in expected_patterns[tile_key]:
                full_pattern = os.path.join(output_dir, pattern)
                matches = glob.glob(full_pattern, recursive=True)

                if matches:
                    found_file = matches[0]  # Take first match
                    break

            if found_file and os.path.exists(found_file):
                validation["found_files"][tile_key] = {
                    "path": found_file,
                    "size_bytes": os.path.getsize(found_file),
                    "exists": True,
                }
                validation["found_tiles"] += 1
            else:
                validation["missing_tiles"].append(
                    {
                        "tile_coord": (x, y),
                        "expected_patterns": expected_patterns[tile_key],
                    }
                )

        validation["success_rate"] = (
            validation["found_tiles"] / validation["expected_tiles"]
            if validation["expected_tiles"] > 0
            else 0
        )

        return validation

    def _log_download_attempt(
        self, site: str, year: int, modality: str, tile_coords: List[Tuple[int, int]]
    ):
        """Log a download attempt before calling R function"""
        log_key = f"{site}_{year}_{modality}"

        self.download_logs[log_key] = {
            "site": site,
            "year": year,
            "modality": modality,
            "requested_tiles": len(tile_coords),
            "tile_coordinates": tile_coords,
            "attempt_time": pd.Timestamp.now().isoformat(),
            "status": "attempting",
            "r_success": None,
            "validation_results": None,
        }

    def _save_download_logs(self, output_dir: str):
        """Save comprehensive download logs to JSON and CSV files"""
        log_dir = os.path.join(output_dir, "download_logs")
        os.makedirs(log_dir, exist_ok=True)

        # Save full logs as JSON
        json_path = os.path.join(log_dir, "full_download_log.json")
        with open(json_path, "w") as f:
            json.dump(self.download_logs, f, indent=2, default=str)

        # Create summary CSV
        summary_data = []
        for log_key, log_data in self.download_logs.items():
            validation = log_data.get("validation_results", {})

            summary_data.append(
                {
                    "site": log_data["site"],
                    "year": log_data["year"],
                    "modality": log_data["modality"],
                    "requested_tiles": log_data["requested_tiles"],
                    "r_download_success": log_data.get("r_success", False),
                    "found_tiles": validation.get("found_tiles", 0),
                    "missing_tiles": validation.get("expected_tiles", 0)
                    - validation.get("found_tiles", 0),
                    "success_rate": validation.get("success_rate", 0.0),
                    "status": log_data.get("status", "unknown"),
                }
            )

        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            csv_path = os.path.join(log_dir, "download_summary.csv")
            summary_df.to_csv(csv_path, index=False)

            print(f"\n📋 Download logs saved:")
            print(f"  Full log: {json_path}")
            print(f"  Summary: {csv_path}")

            return json_path, csv_path

        return None, None

    def generate_missing_tiles_report(self, output_dir: str) -> str:
        """Generate a detailed report of missing tiles for re-download"""
        missing_tiles = []

        for log_key, log_data in self.download_logs.items():
            validation = log_data.get("validation_results", {})

            for missing_info in validation.get("missing_tiles", []):
                missing_tiles.append(
                    {
                        "site": log_data["site"],
                        "year": log_data["year"],
                        "modality": log_data["modality"],
                        "easting": missing_info["tile_coord"][0],
                        "northing": missing_info["tile_coord"][1],
                        "expected_patterns": missing_info["expected_patterns"],
                    }
                )

        if missing_tiles:
            missing_df = pd.DataFrame(missing_tiles)
            report_path = os.path.join(
                output_dir, "download_logs", "missing_tiles_report.csv"
            )
            missing_df.to_csv(report_path, index=False)

            print(f"\n⚠️  Missing Tiles Report: {report_path}")
            print(f"📊 Total missing tiles: {len(missing_tiles)}")

            # Summary by modality
            missing_by_modality = missing_df.groupby("modality").size()
            for modality, count in missing_by_modality.items():
                print(f"  {modality.upper()}: {count} missing tiles")

            return report_path
        else:
            print("✅ No missing tiles found!")
            return None

    def _init_download_log(self, site: str, year: int) -> Dict:
        """Initialize download log structure for tracking"""
        return {
            "site": site,
            "year": year,
            "timestamp_start": datetime.now().isoformat(),
            "timestamp_end": None,
            "modalities": {},
            "summary": {
                "total_tiles_requested": 0,
                "total_tiles_successful": 0,
                "total_tiles_failed": 0,
                "success_rate": 0.0,
            },
        }

    def _log_tile_request(
        self,
        download_log: Dict,
        modality: str,
        tile_coords: List[Tuple[int, int]],
        product_code: str,
        output_dir: str,
    ) -> None:
        """Log the tiles being requested for download"""
        download_log["modalities"][modality] = {
            "product_code": product_code,
            "output_dir": output_dir,
            "tiles_requested": [{"easting": e, "northing": n} for e, n in tile_coords],
            "tiles_successful": [],
            "tiles_failed": [],
            "tiles_missing": [],
            "r_command_success": False,
            "download_timestamp": datetime.now().isoformat(),
            "error_message": None,
        }

    def _extract_coords_from_filename(self, filename: str) -> Optional[Tuple[int, int]]:
        """Extract easting/northing coordinates from NEON filename"""
        # RGB pattern: YYYY_SITE_X_XXXXXX_YYYYYY_image.tif
        match = re.search(r"_(\d+)_(\d+)_image\.tif$", filename)
        if match:
            return (int(match.group(1)), int(match.group(2)))

        # HSI pattern: NEON_D01_SITE_DP3_XXXXXX_YYYYYY_YYYY_reflectance.h5
        match = re.search(r"_(\d+)_(\d+)_\d+_reflectance\.h5$", filename)
        if match:
            return (int(match.group(1)), int(match.group(2)))

        # LiDAR pattern: NEON_D01_SITE_DP3_XXXXXX_YYYYYY_YYYY_CHM.tif
        match = re.search(r"_(\d+)_(\d+)_\d+_CHM\.tif$", filename)
        if match:
            return (int(match.group(1)), int(match.group(2)))

        return None

    def _update_download_log(
        self,
        download_log: Dict,
        modality: str,
        r_success: bool,
        successful_tiles: List[Tuple[int, int]],
        failed_tiles: List[Tuple[int, int]],
        error_msg: str = None,
    ) -> None:
        """Update download log with results"""
        modality_log = download_log["modalities"][modality]
        modality_log["r_command_success"] = r_success
        modality_log["tiles_successful"] = [
            {"easting": e, "northing": n} for e, n in successful_tiles
        ]
        modality_log["tiles_failed"] = [
            {"easting": e, "northing": n} for e, n in failed_tiles
        ]
        modality_log["error_message"] = error_msg

        # Find missing tiles (requested but not successful)
        requested_coords = set(
            (t["easting"], t["northing"]) for t in modality_log["tiles_requested"]
        )
        successful_coords = set(successful_tiles)
        missing_coords = requested_coords - successful_coords
        modality_log["tiles_missing"] = [
            {"easting": e, "northing": n} for e, n in missing_coords
        ]

    def _finalize_download_log(self, download_log: Dict, output_dir: str) -> str:
        """Finalize download log and save to files"""
        download_log["timestamp_end"] = datetime.now().isoformat()

        # Calculate summary statistics
        total_requested = 0
        total_successful = 0
        total_failed = 0

        for modality_data in download_log["modalities"].values():
            total_requested += len(modality_data["tiles_requested"])
            total_successful += len(modality_data["tiles_successful"])
            total_failed += len(modality_data["tiles_failed"])

        download_log["summary"] = {
            "total_tiles_requested": total_requested,
            "total_tiles_successful": total_successful,
            "total_tiles_failed": total_failed,
            "success_rate": (
                total_successful / total_requested if total_requested > 0 else 0.0
            ),
        }

        # Save detailed JSON log
        site = download_log["site"]
        year = download_log["year"]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        json_log_path = os.path.join(
            output_dir, f"{site}_{year}_download_log_{timestamp}.json"
        )
        with open(json_log_path, "w") as f:
            json.dump(download_log, f, indent=2)

        # Create human-readable summary report
        summary_path = os.path.join(
            output_dir, f"{site}_{year}_download_summary_{timestamp}.txt"
        )
        self._create_summary_report(download_log, summary_path)

        # Create CSV of missing tiles for easy analysis
        missing_csv_path = os.path.join(
            output_dir, f"{site}_{year}_missing_tiles_{timestamp}.csv"
        )
        self._create_missing_tiles_csv(download_log, missing_csv_path)

        return json_log_path

    def _create_summary_report(self, download_log: Dict, output_path: str) -> None:
        """Create human-readable summary report"""
        with open(output_path, "w") as f:
            f.write(f"NEON Download Summary Report\n")
            f.write(f"=" * 50 + "\n")
            f.write(f"Site: {download_log['site']}\n")
            f.write(f"Year: {download_log['year']}\n")
            f.write(f"Started: {download_log['timestamp_start']}\n")
            f.write(f"Completed: {download_log['timestamp_end']}\n\n")

            f.write(f"Overall Summary:\n")
            f.write(
                f"  Total tiles requested: {download_log['summary']['total_tiles_requested']}\n"
            )
            f.write(
                f"  Total tiles successful: {download_log['summary']['total_tiles_successful']}\n"
            )
            f.write(
                f"  Total tiles failed: {download_log['summary']['total_tiles_failed']}\n"
            )
            f.write(
                f"  Success rate: {download_log['summary']['success_rate']:.1%}\n\n"
            )

            for modality, data in download_log["modalities"].items():
                f.write(f"{modality.upper()} Modality:\n")
                f.write(f"  Product code: {data['product_code']}\n")
                f.write(f"  R command success: {data['r_command_success']}\n")
                f.write(f"  Tiles requested: {len(data['tiles_requested'])}\n")
                f.write(f"  Tiles successful: {len(data['tiles_successful'])}\n")
                f.write(f"  Tiles failed: {len(data['tiles_failed'])}\n")
                f.write(f"  Tiles missing: {len(data['tiles_missing'])}\n")

                if data["error_message"]:
                    f.write(f"  Error: {data['error_message']}\n")

                if data["tiles_missing"]:
                    f.write(f"  Missing tiles:\n")
                    for tile in data["tiles_missing"][:10]:  # Show first 10
                        f.write(f"    ({tile['easting']}, {tile['northing']})\n")
                    if len(data["tiles_missing"]) > 10:
                        f.write(f"    ... and {len(data['tiles_missing']) - 10} more\n")

                f.write("\n")

    def _create_missing_tiles_csv(self, download_log: Dict, output_path: str) -> None:
        """Create CSV file listing all missing tiles for easy reprocessing"""
        missing_tiles = []

        for modality, data in download_log["modalities"].items():
            for tile in data["tiles_missing"]:
                missing_tiles.append(
                    {
                        "site": download_log["site"],
                        "year": download_log["year"],
                        "modality": modality,
                        "product_code": data["product_code"],
                        "easting": tile["easting"],
                        "northing": tile["northing"],
                        "r_command_success": data["r_command_success"],
                        "error_message": data.get("error_message", ""),
                    }
                )

        if missing_tiles:
            df = pd.DataFrame(missing_tiles)
            df.to_csv(output_path, index=False)
            print(f"📋 Missing tiles saved to: {output_path}")
        else:
            # Create empty file to indicate no missing tiles
            with open(output_path, "w") as f:
                f.write(
                    "site,year,modality,product_code,easting,northing,r_command_success,error_message\n"
                )
                f.write("# No missing tiles - all downloads successful!\n")

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
        The validate_and_extract_crown_metadata.py script should have already done thorough validation.

        Args:
            coordinates_df: DataFrame with easting/northing coordinates
            site: NEON site code for logging

        Returns:
            Cleaned DataFrame with valid coordinates
        """
        print(f"\\n🧹 VALIDATING COORDINATES FOR {site}")
        print(f"Input coordinates: {len(coordinates_df)} entries")

        # Basic sanity check - remove any potential NaN values that might have slipped through
        initial_count = len(coordinates_df)
        coordinates_df = coordinates_df.dropna(
            subset=["center_easting", "center_northing"]
        )

        if len(coordinates_df) < initial_count:
            print(
                f"⚠️  Removed {initial_count - len(coordinates_df)} entries with NaN coordinates"
            )

        # Basic validation - ensure coordinates are reasonable UTM values
        # UTM coordinates should be positive and within global UTM ranges
        valid_mask = (
            (coordinates_df["center_easting"] > 0)
            & (coordinates_df["center_easting"] < 1000000)  # Max UTM easting ~834km
            & (coordinates_df["center_northing"] > 0)
            & (coordinates_df["center_northing"] < 10000000)  # Max UTM northing ~9329km
        )

        invalid_coords = coordinates_df[~valid_mask]
        if len(invalid_coords) > 0:
            print(
                f"⚠️  Found {len(invalid_coords)} coordinates with invalid UTM ranges:"
            )
            for idx, row in invalid_coords.head(5).iterrows():  # Show max 5 examples
                source_file = row.get("source_file", "unknown")
                print(
                    f"  {source_file}: E={row['center_easting']}, N={row['center_northing']}"
                )
            if len(invalid_coords) > 5:
                print(f"  ... and {len(invalid_coords) - 5} more")

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
        print(f"\\n📍 CONVERTING TO TILE COORDINATES")

        # FIXED: Use floor to map crowns to correct tiles (not round)
        # NEON tiles are 1km x 1km starting at bottom-left corner
        # Crown at (318500, 4880500) should map to tile (318000, 4880000), NOT (319000, 4881000)
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

    def download_neon_data(
        self,
        coordinates_df: pd.DataFrame,
        site: str,
        year: int,
        modalities: List[str] = ["rgb", "hsi", "lidar"],
        check_availability: bool = True,
        check_size: bool = False,
    ) -> Dict[str, Any]:
        """
        Download NEON data for specified coordinates and modalities.

        Args:
            coordinates_df: DataFrame with coordinate information
            site: NEON site code
            year: Year of data to download
            modalities: List of data types to download ('rgb', 'hsi', 'lidar')
            check_availability: Whether to check data availability first

        Returns:
            Dictionary with download results and summary
        """
        # Ensure year is always an integer - handle both string and int inputs
        try:
            year = int(year)
        except (ValueError, TypeError):
            raise ValueError(
                f"Year must be convertible to int, got: {year} (type: {type(year)})"
            )
        print(f"\\n  STARTING NEON DATA DOWNLOAD")
        print(f"Site: {site}, Year: {year}")
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

        results = {
            "site": site,
            "year": year,
            "total_tiles": len(tile_coords),
            "downloads": {},
            "errors": [],
        }

        # Download each modality
        for modality in modalities:
            try:
                print(f"\\n  Downloading {modality.upper()} data...")

                if modality == "hsi":
                    product_code = self._get_hsi_product_code(year)
                else:
                    product_code = self.PRODUCTS[modality]

                success = self._download_modality(
                    product_code=product_code,
                    site=site,
                    year=year,
                    tile_coords=tile_coords,
                    output_dir=output_dir,
                    modality=modality,
                    check_size=check_size,
                )

                results["downloads"][modality] = {
                    "product_code": product_code,
                    "success": success,
                    "output_dir": os.path.join(output_dir, modality),
                }

            except Exception as e:
                error_msg = f"Error downloading {modality}: {str(e)}"
                print(f"❌ {error_msg}")
                results["errors"].append(error_msg)
                results["downloads"][modality] = {"success": False, "error": error_msg}

        print(f"\\n✅ Download process completed for {site}_{year}")

        # Save comprehensive logs
        self._save_download_logs(output_dir)

        # Generate missing tiles report
        self.generate_missing_tiles_report(output_dir)

        return results

    def _download_modality(
        self,
        product_code: str,
        site: str,
        year: int,
        tile_coords: List[Tuple[int, int]],
        output_dir: str,
        modality: str,
        check_size: bool = False,
    ) -> bool:
        """
        Download a specific data modality using R neonUtilities.

        Args:
            product_code: NEON product code
            site: NEON site code
            year: Year of data
            tile_coords: List of (easting, northing) coordinates
            output_dir: Output directory
            modality: Data modality name

        Returns:
            Success status
        """
        try:
            # Log the download attempt
            self._log_download_attempt(site, year, modality, tile_coords)

            # Prepare coordinates for R
            eastings = [coord[0] for coord in tile_coords]
            northings = [coord[1] for coord in tile_coords]

            # Convert to R vectors - FIXED: use robjects.FloatVector
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

            print(f"Executing R download for {len(tile_coords)} tiles...")
            result = r(r_command)

            success = bool(result[0])

            # Validate what was actually downloaded
            validation = self._validate_downloaded_files(
                modality_output, modality, site, year, tile_coords
            )

            # Update log with results
            log_key = f"{site}_{year}_{modality}"
            if log_key in self.download_logs:
                self.download_logs[log_key]["r_success"] = success
                self.download_logs[log_key]["validation_results"] = validation
                self.download_logs[log_key]["status"] = "completed"

            if success:
                print(f"✅ R command succeeded for {modality}")
                print(
                    f"📊 Validation: {validation['found_tiles']}/{validation['expected_tiles']} tiles found ({validation['success_rate']:.1%})"
                )

                if validation["missing_tiles"]:
                    print(f"⚠️  {len(validation['missing_tiles'])} tiles are missing!")

            else:
                print(f"❌ R command failed for {modality}")

            return success

        except Exception as e:
            print(f"❌ Exception during {modality} download: {str(e)}")
            return False


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Download NEON tiles for crowns.")
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


if __name__ == "__main__":
    main()
