"""
NEON data downloader with support for RGB, HSI, and LiDAR data products.
"""

import os
import pandas as pd
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri, r
from rpy2.robjects.conversion import localconverter
import warnings

# Suppress R warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)


class NEONDownloader:
    """
    Downloads NEON airborne data products (RGB, HSI, LiDAR) for specified coordinates.
    """
    
    # NEON data product codes
    PRODUCTS = {
        'rgb': 'DP3.30010.001',  # RGB orthophotos
        'hsi_pre2022': 'DP3.30006.001',  # HSI without BRDF correction (pre-2022)
        'hsi_post2022': 'DP3.30006.002',  # HSI with BRDF correction (2022+)
        'lidar': 'DP3.30015.001',  # LiDAR CHM
    }
    
    def __init__(self, base_output_dir: str = '/tmp/neon_downloads'):
        """
        Initialize the NEON downloader.
        
        Args:
            base_output_dir: Base directory for downloaded data
        """
        self.base_output_dir = base_output_dir
        self._setup_r_environment()
    
    def _setup_r_environment(self):
        """Setup R environment and load required packages."""
        try:
            # Activate pandas2ri for automatic conversion
            pandas2ri.activate()
            
            # Load required R packages
            r('library(neonUtilities)')
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
        if year >= 2022:
            return self.PRODUCTS['hsi_post2022']
        else:
            return self.PRODUCTS['hsi_pre2022']
    
    def _clean_coordinates(self, coordinates_df: pd.DataFrame, site: str) -> pd.DataFrame:
        """
        Clean and validate coordinates for a specific site.
        
        Args:
            coordinates_df: DataFrame with easting/northing coordinates
            site: NEON site code for validation
            
        Returns:
            Cleaned DataFrame with valid coordinates
        """
        print(f"\\n🧹 CLEANING COORDINATES FOR {site}")
        print(f"Original coordinates: {len(coordinates_df)} entries")
        
        # Remove rows with missing coordinates
        initial_count = len(coordinates_df)
        coordinates_df = coordinates_df.dropna(subset=['center_easting', 'center_northing'])
        print(f"After removing NaN coordinates: {len(coordinates_df)} entries")
        
        # Site-specific coordinate validation ranges
        site_ranges = {
            'BART': {'e_min': 310000, 'e_max': 325000, 'n_min': 4870000, 'n_max': 4890000},
            'HARV': {'e_min': 720000, 'e_max': 740000, 'n_min': 4700000, 'n_max': 4720000},
            # Add more sites as needed
        }
        
        if site in site_ranges:
            ranges = site_ranges[site]
            valid_mask = (
                (coordinates_df['center_easting'] >= ranges['e_min']) &
                (coordinates_df['center_easting'] <= ranges['e_max']) &
                (coordinates_df['center_northing'] >= ranges['n_min']) &
                (coordinates_df['center_northing'] <= ranges['n_max'])
            )
            
            invalid_coords = coordinates_df[~valid_mask]
            if len(invalid_coords) > 0:
                print(f"Removing {len(invalid_coords)} invalid coordinates:")
                for _, row in invalid_coords.iterrows():
                    print(f"  {row['filename']}: E={row['center_easting']}, N={row['center_northing']}")
            
            coordinates_df = coordinates_df[valid_mask]
            print(f"After coordinate validation: {len(coordinates_df)} entries")
        
        return coordinates_df
    
    def _convert_to_tile_coordinates(self, coordinates_df: pd.DataFrame) -> List[Tuple[int, int]]:
        """
        Convert center coordinates to NEON tile coordinates (1000m grid).
        
        Args:
            coordinates_df: DataFrame with center coordinates
            
        Returns:
            List of unique (easting, northing) tile coordinate pairs
        """
        print(f"\\n📍 CONVERTING TO TILE COORDINATES")
        
        # Round to nearest 1000m (NEON tile size)
        tile_eastings = (coordinates_df['center_easting'] / 1000).round().astype(int) * 1000
        tile_northings = (coordinates_df['center_northing'] / 1000).round().astype(int) * 1000
        
        # Get unique coordinate pairs
        tile_coords = list(set(zip(tile_eastings, tile_northings)))
        
        print(f"Generated {len(tile_coords)} unique tile coordinates")
        print(f"Easting range: {min(e for e, n in tile_coords)} to {max(e for e, n in tile_coords)}")
        print(f"Northing range: {min(n for e, n in tile_coords)} to {max(n for e, n in tile_coords)}")
        
        return tile_coords
    
    def download_neon_data(self, 
                          coordinates_df: pd.DataFrame, 
                          site: str, 
                          year: int,
                          modalities: List[str] = ['rgb', 'hsi', 'lidar'],
                          check_availability: bool = True) -> Dict[str, Any]:
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
        print(f"\\n🚀 STARTING NEON DATA DOWNLOAD")
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
            'site': site,
            'year': year,
            'total_tiles': len(tile_coords),
            'downloads': {},
            'errors': []
        }
        
        # Download each modality
        for modality in modalities:
            try:
                print(f"\\n📥 Downloading {modality.upper()} data...")
                
                if modality == 'hsi':
                    product_code = self._get_hsi_product_code(year)
                else:
                    product_code = self.PRODUCTS[modality]
                
                success = self._download_modality(
                    product_code=product_code,
                    site=site,
                    year=year,
                    tile_coords=tile_coords,
                    output_dir=output_dir,
                    modality=modality
                )
                
                results['downloads'][modality] = {
                    'product_code': product_code,
                    'success': success,
                    'output_dir': os.path.join(output_dir, modality)
                }
                
            except Exception as e:
                error_msg = f"Error downloading {modality}: {str(e)}"
                print(f"❌ {error_msg}")
                results['errors'].append(error_msg)
                results['downloads'][modality] = {
                    'success': False,
                    'error': error_msg
                }
        
        print(f"\\n✅ Download process completed for {site}_{year}")
        return results
    
    def _download_modality(self, 
                          product_code: str, 
                          site: str, 
                          year: int,
                          tile_coords: List[Tuple[int, int]], 
                          output_dir: str,
                          modality: str) -> bool:
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
            robjects.globalenv['eastings'] = r_eastings
            robjects.globalenv['northings'] = r_northings
            robjects.globalenv['site_code'] = site
            robjects.globalenv['year_val'] = year
            robjects.globalenv['product_code'] = product_code
            robjects.globalenv['output_path'] = modality_output
            
            # R download command
            r_command = '''
            tryCatch({
                cat("Downloading with parameters:\\n")
                cat("Product:", product_code, "\\n")
                cat("Site:", site_code, "\\n") 
                cat("Year:", year_val, "\\n")
                cat("Coordinates:", length(eastings), "tile pairs\\n")
                cat("Output:", output_path, "\\n")
                
                byTileAOP(dpID = product_code,
                         site = site_code,
                         year = year_val,
                         easting = eastings,
                         northing = northings,
                         savepath = output_path,
                         check.size = FALSE)
                
                cat("Download completed successfully\\n")
                return(TRUE)
            }, error = function(e) {
                cat("Error in download:", e$message, "\\n")
                return(FALSE)
            })
            '''
            
            print(f"Executing R download for {len(tile_coords)} tiles...")
            result = r(r_command)
            
            success = bool(result[0])
            if success:
                print(f"✅ Successfully downloaded {modality} data")
                # List downloaded files
                downloaded_files = []
                for root, dirs, files in os.walk(modality_output):
                    downloaded_files.extend([os.path.join(root, f) for f in files])
                print(f"Downloaded {len(downloaded_files)} files to {modality_output}")
            else:
                print(f"❌ Download failed for {modality}")
            
            return success
            
        except Exception as e:
            print(f"❌ Exception during {modality} download: {str(e)}")
            return False


def main():
    """Example usage of NEONDownloader."""
    # Example coordinates file
    coords_file = "/blue/azare/riteshchowdhry/Macrosystems/Data_files/hand_annotated_neon/consolidated_dir/neon_sites_coordinates_fixed.csv"
    
    if not os.path.exists(coords_file):
        print(f"Coordinates file not found: {coords_file}")
        print("Please run the shapefile processor first.")
        return
    
    # Load coordinates
    coords_df = pd.read_csv(coords_file)
    
    # Example: Download BART 2019 data
    site = 'BART'
    year = 2019
    
    # Filter for specific site and year
    site_coords = coords_df[
        (coords_df['site'] == site) & 
        (coords_df['year'] == year)
    ]
    
    if len(site_coords) == 0:
        print(f"No coordinates found for {site} {year}")
        return
    
    # Initialize downloader
    downloader = NEONDownloader(base_output_dir='/blue/azare/riteshchowdhry/Macrosystems/Data_files/neon_tiles')
    
    # Download data
    results = downloader.download_neon_data(
        coordinates_df=site_coords,
        site=site,
        year=year,
        modalities=['rgb', 'hsi', 'lidar']
    )
    
    print("\\nDownload Results:")
    print(f"Site: {results['site']}")
    print(f"Year: {results['year']}")
    print(f"Total tiles: {results['total_tiles']}")
    
    for modality, info in results['downloads'].items():
        if info['success']:
            print(f"✅ {modality.upper()}: Successfully downloaded to {info['output_dir']}")
        else:
            print(f"❌ {modality.upper()}: Failed - {info.get('error', 'Unknown error')}")


if __name__ == "__main__":
    main()
