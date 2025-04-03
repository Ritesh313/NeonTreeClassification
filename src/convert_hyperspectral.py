import os
import h5py
import numpy as np
import warnings
import rasterio
from rasterio.transform import Affine

def convert_h5_to_tif(h5_file, output_dir, year, site, coords):
    """
    Convert a single HSI H5 file to GeoTIFF format
    
    Parameters:
    -----------
    h5_file : str
        Path to the H5 file
    output_dir : str
        Directory to save the output TIF file
    suffix : str, optional
        Additional suffix for the output filename
    
    Returns:
    --------
    str
        Path to the created TIF file
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    # Generate output filename from h5 file path
    output_file = f"{year}_{site}_{coords}_HSI.tif"

    
    # Extract metadata and reflectance data
    with h5py.File(h5_file, 'r') as hdf:
        # Get sitename from file attributes
        sitename = list(hdf.keys())[0]
        
        # Get reflectance data and metadata
        refl_group = hdf[sitename]['Reflectance']
        refl_data = refl_group['Reflectance_Data'][:]
        
        # Get metadata
        epsg = str(refl_group['Metadata']['Coordinate_System']['EPSG Code'][()])
        epsg = epsg.split("'")[1] if "'" in epsg else epsg
        
        map_info = str(refl_group['Metadata']['Coordinate_System']['Map_Info'][()])
        map_info = map_info.split(",")
        
        # Get resolution
        pixel_width = float(map_info[5])
        pixel_height = float(map_info[6])
        
        # Get corner coordinates
        x_min = float(map_info[3])
        y_max = float(map_info[4])
        
        # Calculate extent
        shape = refl_data.shape
        x_max = x_min + (shape[1] * pixel_width)
        y_min = y_max - (shape[0] * pixel_height)
        
        # Delete water absorption bands
        band_indices = np.r_[0:425]
        band_indices = np.delete(band_indices, np.r_[419:425])
        band_indices = np.delete(band_indices, np.r_[283:315])
        band_indices = np.delete(band_indices, np.r_[192:210])
        refl_data = refl_data[:, :, band_indices]
        
        # Scale factor
        scale_factor = float(refl_group['Reflectance_Data'].attrs['Scale_Factor'])
        no_data_val = float(refl_group['Reflectance_Data'].attrs['Data_Ignore_Value'])
    
    # Create geotransform for the raster
    transform = Affine.translation(x_min, y_max) * Affine.scale(pixel_width, -pixel_height)
    
    # Rearrange dimensions for rasterio (bands, rows, cols)
    refl_data = np.moveaxis(refl_data, 2, 0)
    
    # Write to GeoTIFF
    output_path = os.path.join(output_dir, output_file)
    with rasterio.open(
        output_path,
        'w',
        driver='GTiff',
        height=refl_data.shape[1],
        width=refl_data.shape[2],
        count=refl_data.shape[0],
        dtype=refl_data.dtype,
        crs=f'EPSG:{epsg}',
        transform=transform,
        nodata=no_data_val
    ) as dst:
        dst.write(refl_data)
    
    return output_path

def batch_convert_h5_to_tif(mapping_list, output_dir):
    """
    Convert all HSI H5 files to GeoTIFF using existing mapping. Make sure RGB file name has site and year
    and easting and northing coordinates.
    
    Parameters:
    -----------
    mapping_list : list of dict
        List of dictionaries with 'rgb_file', 'hsi_file', 'easting', 'northing' keys
    output_dir : str
        Directory to save the output TIF files
    bands : str
        Band selection strategy: "all" or "no_water"
    
    Returns:
    --------
    list
        List of created TIF filenames
    """
    created_files = []
    # generate a warning about the code using the RGB file name to extract year and site
    warnings.warn("Using RGB file name to extract year and site. Ensure the naming follows the format: YYYY_SITE_*_*.tif")
    
    for item in mapping_list:
        hsi_file = item['hsi_file']
        rgb_file = item['rgb_file']
        easting = item['easting']
        northing = item['northing']
        # Extract year and site from the rgb name
        year = rgb_file.split('/')[-1].split('_')[0]
        site = rgb_file.split('/')[-1].split('_')[1]
        coords = f"{easting}_{northing}"
        
        try:
            output_file = convert_h5_to_tif(
                h5_file=hsi_file,
                output_dir=output_dir,
                year=year,
                site=site,  
                coords=coords
            )
            
            created_files.append(output_file)
            print(f"Converted: {hsi_file} -> {output_file}")
        except Exception as e:
            print(f"Error converting {hsi_file}: {str(e)}")
    
    print(f"Successfully converted {len(created_files)} of {len(mapping_list)} files")
    return created_files


if __name__ == "__main__":
    # example to run the conversion code
    mapping_list = [
        {'rgb_file': '/blue/azare/riteshchowdhry/Macrosystems/Data_files/unlabeled_data/HARV/RGB/Mosaic/2022_HARV_7_734000_4709000_image.tif', 
         'hsi_file': '/blue/azare/riteshchowdhry/Macrosystems/Data_files/unlabeled_data/HARV/HSI/Reflectance/NEON_D01_HARV_DP3_734000_4709000_bidirectional_reflectance.h5', 
         'easting': 734000, 
         'northing': 4709000}
    ]
    output_dir = '/blue/azare/riteshchowdhry/Macrosystems/Data_files/unlabeled_data/HARV/HSI/HSI_tif'
    converted_files = batch_convert_h5_to_tif(mapping_list, output_dir)
    print(f"Successfully converted {len(converted_files)} HSI files to TIF format")