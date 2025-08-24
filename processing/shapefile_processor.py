"""
Shapefile processing and coordinate system handling for NEON data.
"""

import geopandas as gpd
import pandas as pd
import os
import glob
import re
import shutil
import numpy as np
from pyproj import CRS
import warnings
from typing import Dict, Tuple, Optional, List

# Suppress geopandas warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)


class ShapefileProcessor:
    """
    Handles shapefile consolidation, CRS transformation, and coordinate extraction
    for NEON tree crown annotations.
    """
    
    # NEON site UTM zones (based on site locations)
    SITE_UTM_ZONES = {
        'ABBY': 'EPSG:32610',  # Zone 10N
        'BART': 'EPSG:32619',  # Zone 19N
        'BONA': 'EPSG:32606',  # Zone 6N
        'CLBJ': 'EPSG:32614',  # Zone 14N
        'DEJU': 'EPSG:32606',  # Zone 6N
        'DELA': 'EPSG:32616',  # Zone 16N
        'GRSM': 'EPSG:32617',  # Zone 17N
        'GUAN': 'EPSG:32619',  # Zone 19N
        'HARV': 'EPSG:32618',  # Zone 18N
        'HEAL': 'EPSG:32606',  # Zone 6N
        'JERC': 'EPSG:32616',  # Zone 16N
        'KONZ': 'EPSG:32614',  # Zone 14N
        'LENO': 'EPSG:32616',  # Zone 16N
        'MLBS': 'EPSG:32617',  # Zone 17N
        'MOAB': 'EPSG:32612',  # Zone 12N
        'NIWO': 'EPSG:32613',  # Zone 13N
        'ONAQ': 'EPSG:32612',  # Zone 12N
        'OSBS': 'EPSG:32617',  # Zone 17N
        'PUUM': 'EPSG:32605',  # Zone 5N
        'RMNP': 'EPSG:32613',  # Zone 13N
        'SCBI': 'EPSG:32617',  # Zone 17N
        'SERC': 'EPSG:32618',  # Zone 18N
        'SJER': 'EPSG:32611',  # Zone 11N
        'SOAP': 'EPSG:32611',  # Zone 11N
        'SRER': 'EPSG:32612',  # Zone 12N
        'TALL': 'EPSG:32616',  # Zone 16N
        'TEAK': 'EPSG:32611',  # Zone 11N
        'UKFS': 'EPSG:32615',  # Zone 15N
        'UNDE': 'EPSG:32616',  # Zone 16N
        'WREF': 'EPSG:32610',  # Zone 10N
    }
    
    def __init__(self):
        self.processing_summary = {
            'total_files': 0,
            'successful': 0,
            'errors': 0,
            'reprojected': 0,
            'empty_geometries': 0,
            'invalid_coordinates': 0
        }
    
    def consolidate_files(self, parent_dir: str, destination_dir: str) -> List[Tuple[str, str]]:
        """
        Consolidate shapefiles from subdirectories into a single directory.
        
        Args:
            parent_dir: Source directory containing subdirectories with shapefiles
            destination_dir: Target directory for consolidated files
            
        Returns:
            List of (file_path, error_message) tuples for any errors encountered
        """
        errors = []
        files_moved = []
        
        for root, dirs, files in os.walk(parent_dir):
            # Skip the destination directory to avoid copying files to themselves
            if root == destination_dir:
                continue
                
            print(f"Checking directory: {root}")
            
            all_files = []
            for ext in ['.shp', '.shx', '.dbf', '.prj', '.cpg']:
                all_files.extend(glob.glob(os.path.join(root, f'*{ext}')))
            
            for file_path in all_files:
                try:
                    file_name = os.path.basename(file_path)
                    dest_path = os.path.join(destination_dir, file_name)
                    
                    if os.path.exists(dest_path):
                        print(f"Warning: File {file_name} already exists in destination. Ignoring.")
                        continue
                    
                    shutil.copy2(file_path, dest_path)
                    files_moved.append(file_path)
                    print(f"Copied: {file_path} -> {dest_path}")
                except Exception as e:
                    errors.append((file_path, str(e)))
                    print(f"Error copying {file_path}: {e}")

        print("\nConsolidation complete!")
        print(f"Total files moved: {len(files_moved)}")
        print(f"Errors encountered: {len(errors)}")

        if errors:
            print("\nFiles that couldn't be copied:")
            for file_path, error in errors:
                print(f"- {file_path}: {error}")

        print("\nFiles in destination directory by extension:")
        for ext in ['.shp', '.shx', '.dbf', '.prj', '.cpg']:
            count = len(glob.glob(os.path.join(destination_dir, f'*{ext}')))
            print(f"{ext}: {count} files")
            
        return errors
    
    @staticmethod
    def extract_metadata(filename: str) -> Tuple[str, str, str]:
        """Extract site, plot, and year from filename."""
        base_filename = os.path.basename(filename).split('.')[0]
        parts = base_filename.split('_')
        
        # Debug print to help identify parsing issues
        print(f"    Parsing filename '{filename}' -> base: '{base_filename}' -> parts: {parts}")
        
        if len(parts) >= 3:
            site = parts[0].upper()  # Ensure uppercase for consistency
            plot = parts[1]
            year = parts[2]
            print(f"    Extracted: site='{site}', plot='{plot}', year='{year}'")
            return site, plot, year
        elif len(parts) >= 1:
            # If we can't parse plot/year, at least try to get the site
            site = parts[0].upper()
            print(f"    Partial extraction: site='{site}', plot='unknown', year='unknown'")
            return site, "unknown", "unknown"
        
        print(f"    Failed to extract any metadata from '{filename}'")
        return "unknown", "unknown", "unknown"
    
    def get_target_utm_crs(self, site_code: str) -> str:
        """
        Determine the appropriate UTM CRS for each NEON site based on their known locations.
        Returns EPSG code as string.
        """
        # First check if we have a mapping for this site
        if site_code in self.SITE_UTM_ZONES:
            target_crs = self.SITE_UTM_ZONES[site_code]
            print(f"    Found UTM zone mapping: {site_code} -> {target_crs}")
            return target_crs
        else:
            print(f"    Warning: No UTM zone mapping found for site '{site_code}', using default EPSG:32619")
            return 'EPSG:32619'  # Default to Zone 19N if unknown
    
    def _validate_coordinates(self, bounds: List[float]) -> bool:
        """Validate coordinate bounds for reasonable UTM values."""
        # Check for infinite or extremely large values
        if not all(np.isfinite(bounds)):
            return False
        # UTM easting typically 100k-900k, northing 0-10M (Northern hemisphere)
        min_x, min_y, max_x, max_y = bounds
        return (100000 <= min_x <= 900000 and 100000 <= max_x <= 900000 and
                0 <= min_y <= 10000000 and 0 <= max_y <= 10000000)
    
    def get_centroid_coords_fixed(self, gdf: gpd.GeoDataFrame, site_code: str) -> Tuple[float, float, List[float], str, str, str]:
        """
        Calculate centroid coordinates with proper CRS handling.
        Ensures all coordinates are in the appropriate UTM projection.
        
        Args:
            gdf: GeoDataFrame with geometries
            site_code: NEON site code for determining target UTM zone
            
        Returns:
            Tuple of (center_x, center_y, bounds, utm_zone, original_crs, target_crs)
        """
        original_crs = gdf.crs
        
        # Get the target UTM CRS for this site
        target_crs = self.get_target_utm_crs(site_code)
        
        # Reproject if necessary
        if gdf.crs is None:
            print(f"    Warning: No CRS defined, assuming WGS84")
            gdf = gdf.set_crs('EPSG:4326')
            original_crs = gdf.crs
        
        # Check if reprojection is actually needed
        if str(gdf.crs) != target_crs:
            print(f"    Reprojecting from {gdf.crs} to {target_crs}")
            gdf = gdf.to_crs(target_crs)
        else:
            print(f"    File already in target CRS ({target_crs}), no reprojection needed")
        
        # Calculate bounds in the projected coordinate system
        bounds = gdf.total_bounds  # minx, miny, maxx, maxy
        
        # Validate coordinates before proceeding
        if not self._validate_coordinates(bounds.tolist()):
            raise ValueError(f"Invalid coordinate bounds: {bounds}")
        
        center_x = (bounds[0] + bounds[2]) / 2
        center_y = (bounds[1] + bounds[3]) / 2
        
        # Extract UTM zone from target CRS
        utm_zone = "unknown"
        try:
            crs_obj = CRS.from_string(target_crs)
            if 'utm' in crs_obj.to_string().lower():
                utm_match = re.search(r'utm zone (\d+)', crs_obj.to_string().lower())
                if utm_match:
                    utm_zone = utm_match.group(1)
                else:
                    # Extract from EPSG code (e.g., EPSG:32619 -> zone 19)
                    epsg_num = int(target_crs.split(':')[1])
                    if 32601 <= epsg_num <= 32660:  # UTM North zones
                        utm_zone = str(epsg_num - 32600)
                    elif 32701 <= epsg_num <= 32760:  # UTM South zones
                        utm_zone = str(epsg_num - 32700)
        except:
            pass
        
        return center_x, center_y, bounds.tolist(), utm_zone, str(original_crs), target_crs
    
    def process_shapefiles(self, consolidated_dir: str, output_filename: str = 'neon_sites_coordinates_fixed.csv') -> Tuple[pd.DataFrame, Dict]:
        """
        Process shapefiles with proper CRS handling and coordinate transformation.
        
        Args:
            consolidated_dir: Directory containing consolidated shapefiles
            output_filename: Name for output CSV file
            
        Returns:
            Tuple of (processed_dataframe, processing_summary)
        """
        shp_files = glob.glob(os.path.join(consolidated_dir, '*.shp'))
        
        sites_data = []
        self.processing_summary = {
            'total_files': len(shp_files),
            'successful': 0,
            'errors': 0,
            'reprojected': 0,
            'empty_geometries': 0,
            'invalid_coordinates': 0
        }

        print(f"\nProcessing {len(shp_files)} shapefiles...")
        
        for shp_file in shp_files:
            try:
                filename = os.path.basename(shp_file)
                site, plot, year = self.extract_metadata(filename)
                
                print(f"\nProcessing: {filename}")
                print(f"  Site: {site}, Plot: {plot}, Year: {year}")
                
                # Read the shapefile
                gdf = gpd.read_file(shp_file)
                
                # Check if geometry is empty
                if len(gdf) == 0 or gdf.geometry.isna().all():
                    print(f"  Warning: Empty geometry, skipping")
                    self.processing_summary['empty_geometries'] += 1
                    continue
                
                # Remove any invalid geometries
                gdf = gdf[gdf.geometry.notna()]
                gdf = gdf[gdf.geometry.is_valid]
                
                if len(gdf) == 0:
                    print(f"  Warning: No valid geometries after cleaning, skipping")
                    self.processing_summary['empty_geometries'] += 1
                    continue
                
                print(f"  Original CRS: {gdf.crs}")
                print(f"  Number of polygons: {len(gdf)}")
                
                # Get coordinates with proper CRS handling
                center_x, center_y, bounds, utm_zone, original_crs, target_crs = self.get_centroid_coords_fixed(gdf, site)
                
                if str(original_crs) != target_crs:
                    self.processing_summary['reprojected'] += 1
                
                # Store the data
                sites_data.append({
                    'filename': filename,
                    'site': site,
                    'plot': plot,
                    'year': year,
                    'center_easting': center_x,
                    'center_northing': center_y,
                    'min_easting': bounds[0],
                    'min_northing': bounds[1],
                    'max_easting': bounds[2],
                    'max_northing': bounds[3],
                    'utm_zone': utm_zone,
                    'original_crs': original_crs,
                    'target_crs': target_crs,
                    'num_polygons': len(gdf)
                })
                
                print(f"  ✅ Successfully processed")
                print(f"  Final coordinates: E={center_x:.1f}, N={center_y:.1f}")
                self.processing_summary['successful'] += 1
                
            except ValueError as e:
                if "Invalid coordinate bounds" in str(e):
                    print(f"  ⚠️  Invalid coordinates, skipping")
                    self.processing_summary['invalid_coordinates'] += 1
                else:
                    print(f"  ❌ Error processing {filename}: {e}")
                    self.processing_summary['errors'] += 1
            except Exception as e:
                print(f"  ❌ Error processing {filename}: {e}")
                self.processing_summary['errors'] += 1
                
                # Add error entry with NaN coordinates
                sites_data.append({
                    'filename': filename,
                    'site': site,
                    'plot': plot,
                    'year': year,
                    'center_easting': None,
                    'center_northing': None,
                    'min_easting': None,
                    'min_northing': None,
                    'max_easting': None,
                    'max_northing': None,
                    'utm_zone': 'error',
                    'original_crs': 'error',
                    'target_crs': 'error',
                    'num_polygons': 0
                })

        # Create DataFrame
        sites_df = pd.DataFrame(sites_data)
        
        # Print processing summary
        self._print_processing_summary(sites_df)
        
        # Save to CSV in the parent directory 
        parent_dir = os.path.dirname(consolidated_dir)
        csv_path = os.path.join(parent_dir, output_filename)
        sites_df.to_csv(csv_path, index=False)
        print(f"\n✅ Saved corrected coordinates to: {csv_path}")
        
        return sites_df, self.processing_summary
    
    def _print_processing_summary(self, sites_df: pd.DataFrame):
        """Print detailed processing summary."""
        print(f"\n" + "="*60)
        print(f"PROCESSING SUMMARY")
        print(f"="*60)
        print(f"Total files: {self.processing_summary['total_files']}")
        print(f"Successfully processed: {self.processing_summary['successful']}")
        print(f"Errors: {self.processing_summary['errors']}")
        print(f"Files reprojected: {self.processing_summary['reprojected']}")
        print(f"Empty geometries skipped: {self.processing_summary['empty_geometries']}")
        print(f"Invalid coordinates skipped: {self.processing_summary['invalid_coordinates']}")
        
        # Display coordinate summary
        if len(sites_df) > 0 and 'center_easting' in sites_df.columns:
            valid_coords = sites_df.dropna(subset=['center_easting', 'center_northing'])
            if len(valid_coords) > 0:
                print(f"\nCoordinate ranges (valid entries only):")
                print(f"Easting: {valid_coords['center_easting'].min():.0f} to {valid_coords['center_easting'].max():.0f}")
                print(f"Northing: {valid_coords['center_northing'].min():.0f} to {valid_coords['center_northing'].max():.0f}")
        else:
            print(f"\nNo coordinate data available (no files processed)")
        
        if len(sites_df) > 0:
            print(f"\nFirst 5 entries:")
            available_cols = [col for col in ['filename', 'site', 'center_easting', 'center_northing', 'utm_zone', 'target_crs'] if col in sites_df.columns]
            print(sites_df[available_cols].head())
        else:
            print(f"\nNo data to display (no files processed)")
        
        # Site analysis
        if len(sites_df) > 0 and 'site' in sites_df.columns and 'target_crs' in sites_df.columns:
            print(f"\nSites and coordinate systems:")
            site_analysis = sites_df.groupby(['site', 'target_crs']).size().reset_index(name='count')
            for _, row in site_analysis.iterrows():
                print(f"  {row['site']}: {row['count']} files in {row['target_crs']}")
        
        # Total polygons
        if len(sites_df) > 0 and 'num_polygons' in sites_df.columns:
            total_polygons = sites_df['num_polygons'].sum()
            print(f"\nTotal polygons across all shapefiles: {total_polygons}")
        else:
            print(f"\nNo polygon data available")


def main():
    """Main function for command-line usage."""
    parent_dir = '/blue/azare/riteshchowdhry/Macrosystems/Data_files/hand_annotated_neon/'
    destination_dir = '/blue/azare/riteshchowdhry/Macrosystems/Data_files/hand_annotated_neon/consolidated_dir'
    
    # Create destination directory if it doesn't exist
    os.makedirs(destination_dir, exist_ok=True)
    
    processor = ShapefileProcessor()
    
    print("STEP 1: Consolidating files (if needed)")
    print("="*60)
    # Uncomment the next line if you need to re-consolidate files
    # processor.consolidate_files(parent_dir, destination_dir)
    
    print("\nSTEP 2: Processing shapefiles with CRS correction")
    print("="*60)
    # Process the consolidated shapefiles with proper CRS handling
    sites_df, summary = processor.process_shapefiles(destination_dir)


if __name__ == "__main__":
    main()
