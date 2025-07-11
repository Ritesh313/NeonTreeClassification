# ⚠️  DEPRECATED: This file has been integrated into the main package
# Please use: from neon_tree_classification.data.shapefile_processor import ShapefileProcessor
# Or run: python scripts/test_shapefile_processor.py

print("⚠️  This script is deprecated!")
print("Please use the ShapefileProcessor class instead:")
print("  from neon_tree_classification.data.shapefile_processor import ShapefileProcessor")
print("Or run the test script:")
print("  python scripts/test_shapefile_processor.py")

import geopandas as gpd
import pandas as pd
import os
import glob
import re
import shutil
from pyproj import CRS
import warnings

# Suppress geopandas warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)


def consolidate_files(parent_dir, destination_dir):
    """
    Consolidate shapefiles from subdirectories into a single directory.
    """
    errors = []
    files_moved = []

    for root, dirs, files in os.walk(parent_dir):
        if root == parent_dir or root == destination_dir:
            continue

        print(f"Checking directory: {root}")

        all_files = []
        for ext in [".shp", ".shx", ".dbf", ".prj", ".cpg"]:
            all_files.extend(glob.glob(os.path.join(root, f"*{ext}")))

        for file_path in all_files:
            try:
                file_name = os.path.basename(file_path)
                dest_path = os.path.join(destination_dir, file_name)

                if os.path.exists(dest_path):
                    print(
                        f"Warning: File {file_name} already exists in destination. Ignoring."
                    )
                    continue

                shutil.copy2(file_path, dest_path)
                files_moved.append(file_path)
                print(f"Copied: {file_path} -> {dest_path}")
            except Exception as e:
                errors.append((file_path, str(e)))
                print(f"Error copying {file_path}: {e}")

    print("\nOperation complete!")
    print(f"Total files moved: {len(files_moved)}")
    print(f"Errors encountered: {len(errors)}")

    if errors:
        print("\nFiles that couldn't be copied:")
        for file_path, error in errors:
            print(f"- {file_path}: {error}")

    print("\nFiles in destination directory by extension:")
    for ext in [".shp", ".shx", ".dbf", ".prj", ".cpg"]:
        count = len(glob.glob(os.path.join(destination_dir, f"*{ext}")))
        print(f"{ext}: {count} files")


def extract_metadata(filename):
    """Extract site, plot, and year from filename."""
    parts = os.path.basename(filename).split(".")[0].split("_")
    if len(parts) >= 3:
        site = parts[0]
        plot = parts[1]
        year = parts[2]
        return site, plot, year
    return "unknown", "unknown", "unknown"


def get_target_utm_crs(site_code):
    """
    Determine the appropriate UTM CRS for each NEON site based on their known locations.
    Returns EPSG code as string.
    """
    # NEON site UTM zones (based on site locations)
    site_utm_zones = {
        "ABBY": "EPSG:32610",  # Zone 10N
        "BART": "EPSG:32619",  # Zone 19N
        "BONA": "EPSG:32606",  # Zone 6N
        "CLBJ": "EPSG:32614",  # Zone 14N
        "DEJU": "EPSG:32606",  # Zone 6N
        "DELA": "EPSG:32616",  # Zone 16N
        "GRSM": "EPSG:32617",  # Zone 17N
        "GUAN": "EPSG:32619",  # Zone 19N
        "HARV": "EPSG:32618",  # Zone 18N
        "HEAL": "EPSG:32606",  # Zone 6N
        "JERC": "EPSG:32616",  # Zone 16N
        "KONZ": "EPSG:32614",  # Zone 14N
        "LENO": "EPSG:32616",  # Zone 16N
        "MLBS": "EPSG:32617",  # Zone 17N
        "MOAB": "EPSG:32612",  # Zone 12N
        "NIWO": "EPSG:32613",  # Zone 13N
        "ONAQ": "EPSG:32612",  # Zone 12N
        "OSBS": "EPSG:32617",  # Zone 17N
        "PUUM": "EPSG:32605",  # Zone 5N
        "RMNP": "EPSG:32613",  # Zone 13N
        "SCBI": "EPSG:32617",  # Zone 17N
        "SERC": "EPSG:32618",  # Zone 18N
        "SJER": "EPSG:32611",  # Zone 11N
        "SOAP": "EPSG:32611",  # Zone 11N
        "SRER": "EPSG:32612",  # Zone 12N
        "TALL": "EPSG:32616",  # Zone 16N
        "TEAK": "EPSG:32611",  # Zone 11N
        "UKFS": "EPSG:32615",  # Zone 15N
        "UNDE": "EPSG:32616",  # Zone 16N
        "WREF": "EPSG:32610",  # Zone 10N
    }

    return site_utm_zones.get(site_code, "EPSG:32619")  # Default to Zone 19N if unknown


def get_centroid_coords_fixed(gdf, site_code):
    """
    Calculate centroid coordinates with proper CRS handling.
    Ensures all coordinates are in the appropriate UTM projection.
    """
    original_crs = gdf.crs

    # Get the target UTM CRS for this site
    target_crs = get_target_utm_crs(site_code)

    # Reproject if necessary
    if gdf.crs is None:
        print(f"    Warning: No CRS defined, assuming WGS84")
        gdf = gdf.set_crs("EPSG:4326")
        original_crs = gdf.crs

    if str(gdf.crs) != target_crs:
        print(f"    Reprojecting from {gdf.crs} to {target_crs}")
        gdf = gdf.to_crs(target_crs)

    # Calculate bounds in the projected coordinate system
    bounds = gdf.total_bounds  # minx, miny, maxx, maxy
    center_x = (bounds[0] + bounds[2]) / 2
    center_y = (bounds[1] + bounds[3]) / 2

    # Extract UTM zone from target CRS
    utm_zone = "unknown"
    try:
        crs_obj = CRS.from_string(target_crs)
        if "utm" in crs_obj.to_string().lower():
            utm_match = re.search(r"utm zone (\d+)", crs_obj.to_string().lower())
            if utm_match:
                utm_zone = utm_match.group(1)
            else:
                # Extract from EPSG code (e.g., EPSG:32619 -> zone 19)
                epsg_num = int(target_crs.split(":")[1])
                if 32601 <= epsg_num <= 32660:  # UTM North zones
                    utm_zone = str(epsg_num - 32600)
                elif 32701 <= epsg_num <= 32760:  # UTM South zones
                    utm_zone = str(epsg_num - 32700)
    except:
        pass

    return center_x, center_y, bounds, utm_zone, str(original_crs), target_crs


def process_shp_files_fixed(consolidated_dir):
    """
    Process shapefiles with proper CRS handling and coordinate transformation.
    """
    shp_files = glob.glob(os.path.join(consolidated_dir, "*.shp"))

    sites_data = []
    processing_summary = {
        "total_files": len(shp_files),
        "successful": 0,
        "errors": 0,
        "reprojected": 0,
        "empty_geometries": 0,
    }

    print(f"\nProcessing {len(shp_files)} shapefiles...")

    for shp_file in shp_files:
        try:
            filename = os.path.basename(shp_file)
            site, plot, year = extract_metadata(filename)

            print(f"\nProcessing: {filename}")
            print(f"  Site: {site}, Plot: {plot}, Year: {year}")

            # Read the shapefile
            gdf = gpd.read_file(shp_file)

            # Check if geometry is empty
            if len(gdf) == 0 or gdf.geometry.isna().all():
                print(f"  Warning: Empty geometry, skipping")
                processing_summary["empty_geometries"] += 1
                continue

            # Remove any invalid geometries
            gdf = gdf[gdf.geometry.notna()]
            gdf = gdf[gdf.geometry.is_valid]

            if len(gdf) == 0:
                print(f"  Warning: No valid geometries after cleaning, skipping")
                processing_summary["empty_geometries"] += 1
                continue

            print(f"  Original CRS: {gdf.crs}")
            print(f"  Number of polygons: {len(gdf)}")

            # Get coordinates with proper CRS handling
            center_x, center_y, bounds, utm_zone, original_crs, target_crs = (
                get_centroid_coords_fixed(gdf, site)
            )

            if str(original_crs) != target_crs:
                processing_summary["reprojected"] += 1

            # Store the data
            sites_data.append(
                {
                    "filename": filename,
                    "site": site,
                    "plot": plot,
                    "year": year,
                    "center_easting": center_x,
                    "center_northing": center_y,
                    "min_easting": bounds[0],
                    "min_northing": bounds[1],
                    "max_easting": bounds[2],
                    "max_northing": bounds[3],
                    "utm_zone": utm_zone,
                    "original_crs": original_crs,
                    "target_crs": target_crs,
                    "num_polygons": len(gdf),
                }
            )

            print(f"  ✅ Successfully processed")
            print(f"  Final coordinates: E={center_x:.1f}, N={center_y:.1f}")
            processing_summary["successful"] += 1

        except Exception as e:
            print(f"  ❌ Error processing {filename}: {e}")
            processing_summary["errors"] += 1

            # Add error entry with NaN coordinates
            sites_data.append(
                {
                    "filename": filename,
                    "site": site,
                    "plot": plot,
                    "year": year,
                    "center_easting": None,
                    "center_northing": None,
                    "min_easting": None,
                    "min_northing": None,
                    "max_easting": None,
                    "max_northing": None,
                    "utm_zone": "error",
                    "original_crs": "error",
                    "target_crs": "error",
                    "num_polygons": 0,
                }
            )

    # Create DataFrame
    sites_df = pd.DataFrame(sites_data)

    # Print processing summary
    print(f"\n" + "=" * 60)
    print(f"PROCESSING SUMMARY")
    print(f"=" * 60)
    print(f"Total files: {processing_summary['total_files']}")
    print(f"Successfully processed: {processing_summary['successful']}")
    print(f"Errors: {processing_summary['errors']}")
    print(f"Files reprojected: {processing_summary['reprojected']}")
    print(f"Empty geometries skipped: {processing_summary['empty_geometries']}")

    # Display coordinate summary
    valid_coords = sites_df.dropna(subset=["center_easting", "center_northing"])
    if len(valid_coords) > 0:
        print(f"\nCoordinate ranges (valid entries only):")
        print(
            f"Easting: {valid_coords['center_easting'].min():.0f} to {valid_coords['center_easting'].max():.0f}"
        )
        print(
            f"Northing: {valid_coords['center_northing'].min():.0f} to {valid_coords['center_northing'].max():.0f}"
        )

    print(f"\nFirst 5 entries:")
    print(
        sites_df[
            [
                "filename",
                "site",
                "center_easting",
                "center_northing",
                "utm_zone",
                "target_crs",
            ]
        ].head()
    )

    # Save to CSV
    csv_path = os.path.join(consolidated_dir, "neon_sites_coordinates_fixed.csv")
    sites_df.to_csv(csv_path, index=False)
    print(f"\n✅ Saved corrected coordinates to: {csv_path}")

    # Site analysis
    print(f"\nSites and coordinate systems:")
    site_analysis = (
        sites_df.groupby(["site", "target_crs"]).size().reset_index(name="count")
    )
    for _, row in site_analysis.iterrows():
        print(f"  {row['site']}: {row['count']} files in {row['target_crs']}")

    # Total polygons
    total_polygons = sites_df["num_polygons"].sum()
    print(f"\nTotal polygons across all shapefiles: {total_polygons}")

    return sites_df, processing_summary


if __name__ == "__main__":
    parent_dir = (
        "/blue/azare/riteshchowdhry/Macrosystems/Data_files/hand_annotated_neon/"
    )
    destination_dir = "/blue/azare/riteshchowdhry/Macrosystems/Data_files/hand_annotated_neon/consolidated_dir"

    # Create destination directory if it doesn't exist
    os.makedirs(destination_dir, exist_ok=True)

    print("STEP 1: Consolidating files (if needed)")
    print("=" * 60)
    # Uncomment the next line if you need to re-consolidate files
    # consolidate_files(parent_dir, destination_dir)

    print("\nSTEP 2: Processing shapefiles with CRS correction")
    print("=" * 60)
    # Process the consolidated shapefiles with proper CRS handling
    sites_df, summary = process_shp_files_fixed(destination_dir)
