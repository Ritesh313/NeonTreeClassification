"""
validate_and_extract_crown_metadata.py

Validate, deduplicate, and extract crown metadata from NEON shapefiles.

Usage:
    python validate_and_extract_crown_metadata.py /path/to/shapefile_directory /path/to/output.gpkg

This script validates all shapefiles in a directory, deduplicates them, extracts crown metadata, cleans the result, and saves as a GeoPackage.
"""

import os
import glob
import geopandas as gpd
import pandas as pd
from typing import List, Dict, Any
import re


def extract_site_plot_year_from_filename(filename):
    base = os.path.splitext(os.path.basename(filename))[0]
    base = re.sub(r" \(\d+\)$", "", base)
    parts = base.split("_")
    if len(parts) == 3 and parts[2].isdigit():
        site, plot, year = parts
        return site, plot, year
    return None, None, None


def validate_shapefile(shp_path: str) -> Dict[str, Any]:
    result = {
        "path": shp_path,
        "valid": False,
        "has_crs": False,
        "missing_sidecars": [],
        "error": None,
        "site": None,
        "plot": None,
        "year": None,
    }
    base, _ = os.path.splitext(shp_path)
    required_exts = [".shp", ".dbf", ".shx"]
    for ext in required_exts:
        if not os.path.exists(base + ext):
            result["missing_sidecars"].append(base + ext)
    try:
        gdf = gpd.read_file(shp_path)
        result["valid"] = True
        if gdf.crs is not None:
            result["has_crs"] = True
        site, plot, year = extract_site_plot_year_from_filename(shp_path)
        result["site"] = site
        result["plot"] = plot
        result["year"] = year
    except Exception as e:
        result["error"] = str(e)
    return result


def deduplicate_shapefiles(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    deduped = []
    for res in results:
        if not res["valid"] or not res["has_crs"]:
            continue
        key = (res.get("site"), res.get("plot"), res.get("year"))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(res)
    return deduped


def summarize_validation(
    results: List[Dict[str, Any]], show_invalid: bool = False
) -> Dict[str, Any]:
    total = len(results)
    valid = sum(r["valid"] and r["has_crs"] for r in results)
    missing_sidecar = sum(r["valid"] and bool(r["missing_sidecars"]) for r in results)
    missing_crs = sum(r["valid"] and not r["has_crs"] for r in results)
    unreadable = sum(not r["valid"] for r in results)
    print(
        f"Summary:\n  Total shapefiles: {total}\n  Valid (with CRS): {valid}\n  Missing sidecar files: {missing_sidecar}\n  Missing CRS: {missing_crs}\n  Unreadable: {unreadable}"
    )
    print(
        "\nTo view invalid files, set show_invalid=True when calling summarize_validation, or run the code separately to examine the paths."
    )
    if show_invalid:
        print("\nInvalid files:")
        for r in results:
            if not (r["valid"] and r["has_crs"]):
                print(
                    f"- {r['path']} | Error: {r['error']} | Missing sidecars: {r['missing_sidecars']}"
                )
    return {
        "total": total,
        "valid": valid,
        "missing_sidecar": missing_sidecar,
        "missing_crs": missing_crs,
        "unreadable": unreadable,
        "invalid_files": [r for r in results if not (r["valid"] and r["has_crs"])],
    }


def extract_crowns_from_shapefiles(shapefile_paths, target_crs="EPSG:4326"):
    # Mapping from NEON site code to UTM CRS
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

    all_gdfs = []
    for shp in shapefile_paths:
        try:
            gdf = gpd.read_file(shp)
            site, plot, year = extract_site_plot_year_from_filename(shp)
            utm_crs = SITE_UTM_ZONES.get(site)
            if utm_crs:
                if gdf.crs is None:
                    gdf = gdf.set_crs("EPSG:4326")
                if str(gdf.crs) != utm_crs:
                    gdf = gdf.to_crs(utm_crs)
                gdf["center_easting"] = gdf.geometry.centroid.x
                gdf["center_northing"] = gdf.geometry.centroid.y
            else:
                # Only print warning if site is not None (avoid noisy output for missing/invalid files)
                if site is not None:
                    print(
                        f"Warning: No UTM zone for site {site}, skipping easting/northing extraction."
                    )
                gdf["center_easting"] = None
                gdf["center_northing"] = None
            if "individual" in gdf.columns:
                gdf = gdf.rename(columns={"individual": "individual_id"})
            else:
                gdf["individual_id"] = None
            gdf["site"] = site
            gdf["plot"] = plot
            gdf["year"] = year
            gdf["source_file"] = shp
            keep_cols = [
                "individual_id",
                "geometry",
                "site",
                "plot",
                "year",
                "source_file",
                "center_easting",
                "center_northing",
            ]
            for col in keep_cols:
                if col not in gdf.columns:
                    gdf[col] = None
            gdf = gdf[keep_cols]
            # Reproject to target_crs (e.g., EPSG:4326) for concatenation, but after extracting easting/northing
            gdf = gdf.to_crs(target_crs)
            # Only append if not empty/all-NA
            if not gdf.empty and not gdf.isna().all(axis=None):
                all_gdfs.append(gdf)
        except Exception as e:
            print(f"Could not read {shp}: {e}")
    # Filter out empty/all-NA DataFrames before concatenation
    all_gdfs = [g for g in all_gdfs if not g.empty and not g.isna().all(axis=None)]
    if all_gdfs:
        combined = gpd.GeoDataFrame(
            pd.concat(all_gdfs, ignore_index=True), crs=target_crs
        )
        cleaned = combined.dropna(
            subset=["individual_id", "geometry", "site", "plot", "year"]
        )
        cleaned = cleaned.reset_index(drop=True)
        return cleaned
    else:
        return gpd.GeoDataFrame(
            columns=[
                "individual_id",
                "geometry",
                "site",
                "plot",
                "year",
                "source_file",
                "center_easting",
                "center_northing",
            ],
            crs=target_crs,
        )


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print(
            "Usage: python validate_and_extract_crown_metadata.py /path/to/shapefile_directory /path/to/output.gpkg"
        )
        sys.exit(1)
    shp_dir = sys.argv[1]
    out_path = sys.argv[2]
    shp_files = glob.glob(os.path.join(shp_dir, "**", "*.shp"), recursive=True)
    results = [validate_shapefile(f) for f in shp_files]
    deduped = deduplicate_shapefiles(results)
    summarize_validation(results)
    shapefile_paths = [rec["path"] for rec in deduped]
    crowns_gdf = extract_crowns_from_shapefiles(shapefile_paths)
    print(f"\nFinal cleaned crowns GeoDataFrame: {len(crowns_gdf)} rows")
    crowns_gdf.to_file(out_path, driver="GPKG")
    print(f"Saved cleaned crown metadata to {out_path}")
