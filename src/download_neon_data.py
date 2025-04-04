import argparse
import os
import site
import rpy2
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from rpy2.rinterface_lib.callbacks import logger as rpy2_logger
import logging
rpy2_logger.setLevel(logging.ERROR)

base = importr('base')
utils = importr('utils')
stats = importr('stats')
neonUtilities = importr('neonUtilities')


def rgb_data_download(easting, northing, site, year, output_dir):
    """
    Download RGB data from NEON for a specific site and year. Pass a list of easting and northing coordinates to download all the tiles that span the area. 
    Args:
        easting: List of easting coordinates
        northing: List of northing coordinates
        site: NEON site code (e.g., 'HARV')
        year: Year of data to download (e.g., '2022')
        output_dir: Directory to save the downloaded data
    """
    # Download the RGB data
    neonUtilities.byTileAOP(dpID=rgb_data_product, site=site, year=year,
                            check_size=False,
                            easting=easting, northing=northing,
                            include_provisional = True,
                            savepath=output_dir);
    
    
def hsi_withbrdf_data_download(easting, northing, site, year, output_dir):
    """
    Download HSI data from NEON for a specific site and year. Pass a list of easting and northing coordinates to download all the tiles that span the area.
    Args:
        easting: List of easting coordinates
        northing: List of northing coordinates
        site: NEON site code (e.g., 'HARV')
        year: Year of data to download (e.g., '2022')
        output_dir: Directory to save the downloaded data
    """
    # Download the HSI data with BRDF correction
    neonUtilities.byTileAOP(dpID=hsi_withbrdf_2022, site=site, year=year,
                            check_size=False,
                            easting=easting, northing=northing,
                            include_provisional = True,
                            savepath=output_dir);
    
def lidar_chm_data_download(easting, northing, site, year, output_dir):
    """
    Download LiDAR CHM data from NEON for a specific site and year. Pass a list of easting and northing coordinates to download all the tiles that span the area.
    Args:
        easting: List of easting coordinates
        northing: List of northing coordinates
        site: NEON site code (e.g., 'HARV')
        year: Year of data to download (e.g., '2022')
        output_dir: Directory to save the downloaded data
    """
    # Download the LiDAR CHM data
    neonUtilities.byTileAOP(dpID=lidar, site=site, year=year,
                            check_size=False,
                            easting=easting, northing=northing,
                            include_provisional = True,
                            savepath=output_dir);
    
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Download NEON data for a specific site and year.')
    parser.add_argument('--site', type=str, required=True, help='NEON site code (e.g., HARV)')
    parser.add_argument('--year', type=str, required=True, help='Year of data to download (e.g., 2022)')
    parser.add_argument('--easting_start', type=int, required=True, help='Starting easting coordinate')
    parser.add_argument('--easting_end', type=int, required=True, help='Ending easting coordinate')
    parser.add_argument('--northing_start', type=int, required=True, help='Starting northing coordinate')
    parser.add_argument('--northing_end', type=int, required=True, help='Ending northing coordinate')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save the downloaded data')
    args = parser.parse_args()
    site = args.site
    year = args.year
    easting_start = args.easting_start
    easting_end = args.easting_end
    northing_start = args.northing_start
    northing_end = args.northing_end
    output_dir = args.output_dir
    
    
    rgb_data_product = 'DP3.30010.001'
    hsi_withbrdf_2022 = 'DP3.30006.002'
    lidar = 'DP3.30015.001'
    
    # to do: add a function to get the easting and northing coordinates from the site name
    
    # rgb_path = '/blue/azare/riteshchowdhry/Macrosystems/Data_files/unlabeled_data/HARV/RGB'
    # rgb_data_download(easting, northing, site, year, rgb_path)
    
    hsi_path = '/blue/azare/riteshchowdhry/Macrosystems/Data_files/unlabeled_data/HARV/HSI'
    hsi_withbrdf_data_download(easting, northing, site, year, hsi_path)