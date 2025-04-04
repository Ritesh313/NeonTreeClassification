# NeonTreeClassification

National Ecological Observatory Network (NEON) offers a variety of data products, including airborne data from different forest sites. Airborne data includes RGB orthophotos, LiDAR (CHM) airborne data, and 426 band hyperspectral data. All products are available on https://data.neonscience.org/data-products/, the following are the airborne data products used in this repository:

rgb_data_product = 'DP3.30010.001'
hsi_withbrdf_2022 = 'DP3.30006.002'
lidar = 'DP3.30015.001' #CHM

# Workflow
## 1. Download NEON data
- Given the Northing, Easting, Year and Site, download the NEON data using the `download_neon_data.py` script. There are functions to download the RGB, HSI, and LiDAR data. The data is downloaded to a specified directory.
### To do:
- Merge this script in neon_utils.py
- Look into using Google Earth Engine

## 2. Generate crowns using deepforest
- The `deepforest_parallel.py` script uses the deepforest package to generate tree crowns from the RGB data. The script is parallelized on SLURM using Dask. It can run on a given list of RGB tiles and save a pandas dataframe with the tree crowns.

# Citations

## NEON Airborne Data Products
NEON (National Ecological Observatory Network). High-resolution orthorectified camera imagery mosaic (DP3.30010.001), RELEASE-2025. https://doi.org/10.48443/gdgn-3r69. Dataset accessed from https://data.neonscience.org/data-products/DP3.30010.001/RELEASE-2025 on April 3, 2025.

NEON (National Ecological Observatory Network). Spectrometer orthorectified surface bidirectional reflectance - mosaic (DP3.30006.002), provisional data. Dataset accessed from https://data.neonscience.org/data-products/DP3.30006.002 on April 3, 2025. Data archived at [your DOI].

NEON (National Ecological Observatory Network). Ecosystem structure (DP3.30015.001), RELEASE-2025. https://doi.org/10.48443/jqqd-1n30. Dataset accessed from https://data.neonscience.org/data-products/DP3.30015.001/RELEASE-2025 on April 3, 2025.

