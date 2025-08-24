# validate_and_extract_crown_metadata.py

This script validates, deduplicates, and extracts crown metadata from NEON tree crown shapefiles, outputting a clean GeoDataFrame for downstream analysis.

## Features
- Validates all shapefiles in a directory (checks readability, CRS, and required sidecar files)
- Extracts `site`, `plot`, and `year` from the filename (expects `SITE_PLOT_YEAR.shp` format)
- Deduplicates files by (site, plot, year)
- Extracts all crowns and their metadata into a single GeoDataFrame
- Cleans the data (removes rows with missing key fields, resets index)
- Saves the result as a GeoPackage (GPKG)

## Usage
```bash
python validate_and_extract_crown_metadata.py /path/to/shapefile_directory /path/to/output.gpkg
```

## Output
- A GeoPackage file containing a single layer with columns:
	- `individual_id`: unique crown identifier
	- `geometry`: crown polygon
	- `site`: 4-letter NEON site code
	- `plot`: plot identifier from filename
	- `year`: 4-digit year from filename
	- `source_file`: original shapefile path

## Notes
- Only shapefiles with valid CRS and required sidecars are processed.
- Filenames must follow the `SITE_PLOT_YEAR.shp` pattern for metadata extraction.
- Rows with missing `individual_id`, `geometry`, `site`, `plot`, or `year` are dropped.
- All geometries are reprojected to EPSG:4326 (WGS84) for consistency.

## Example
```python
import geopandas as gpd
# Load the output GeoPackage
crowns_gdf = gpd.read_file('/path/to/output.gpkg')
print(crowns_gdf.info())
print(crowns_gdf.head())
```
