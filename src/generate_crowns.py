import numpy as np
import os
import glob
import rasterio
import matplotlib.pyplot as plt

from deepforest import main

def convert_rgb_to_hsi_bbox(rgb_bbox, rgb_transform, hsi_transform):
    """
    Convert RGB bounding box to HSI pixel coordinates.
    
    Args:
        rgb_bbox: Tuple of (xmin, ymin, xmax, ymax) in RGB pixel coordinates
        rgb_transform: Affine transform for RGB data
        hsi_transform: Affine transform for HSI data
        
    Returns:
        Tuple of (xmin, ymin, xmax, ymax) in HSI pixel coordinates
    """
    # Unpack RGB bbox
    xmin, ymin, xmax, ymax = rgb_bbox
    
    # Convert RGB pixel coordinates to geospatial coordinates
    geo_xmin, geo_ymin = rgb_transform * (xmin, ymin)
    geo_xmax, geo_ymax = rgb_transform * (xmax, ymax)
    
    # Convert geospatial coordinates to HSI pixel coordinates using inverse transform
    hsi_transform_inv = ~hsi_transform
    hsi_xmin, hsi_ymin = hsi_transform_inv * (geo_xmin, geo_ymin)
    hsi_xmax, hsi_ymax = hsi_transform_inv * (geo_xmax, geo_ymax)
    
    # Round to nearest pixel
    hsi_bbox = (
        int(round(hsi_xmin)),
        int(round(hsi_ymin)),
        int(round(hsi_xmax)),
        int(round(hsi_ymax))
    )
    
    return hsi_bbox

def extract_tree_crowns(rgb_file, hsi_file, tree_bbox, visualize=False):
    """
    Extract corresponding tree crown from RGB and HSI imagery.
    
    Args:
        rgb_file: Path to RGB GeoTIFF file
        hsi_file: Path to HSI GeoTIFF file
        tree_bbox: Tuple of (xmin, ymin, xmax, ymax) in RGB pixel coordinates
        visualize: Whether to display the extracted crowns
        
    Returns:
        Tuple of (rgb_tree, hsi_tree) as numpy arrays
    """
    # Extract RGB crown
    with rasterio.open(rgb_file) as rgb_src:
        rgb_transform = rgb_src.transform
        
        # Create window from bbox (row_start, row_stop), (col_start, col_stop)
        rgb_window = ((tree_bbox[1], tree_bbox[3]), (tree_bbox[0], tree_bbox[2]))
        rgb_tree = rgb_src.read(window=rgb_window)
        
    # Extract HSI crown
    with rasterio.open(hsi_file) as hsi_src:
        hsi_transform = hsi_src.transform
        
        # Convert RGB bbox to HSI bbox
        hsi_bbox = convert_rgb_to_hsi_bbox(tree_bbox, rgb_transform, hsi_transform)
        
        # Create window from bbox
        hsi_window = ((hsi_bbox[1], hsi_bbox[3]), (hsi_bbox[0], hsi_bbox[2]))
        hsi_tree = hsi_src.read(window=hsi_window)
    
    # Optional visualization
    if visualize:
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        # Display RGB (channels, height, width) -> (height, width, channels)
        rgb_display = np.moveaxis(rgb_tree, 0, -1)
        axes[0].imshow(rgb_display)
        axes[0].set_title(f'RGB Tree Crown {rgb_tree.shape}')
        axes[0].axis('off')
        
        # Display HSI using false color (select representative bands)
        # Assuming HSI has at least these bands, otherwise adjust indices
        if hsi_tree.shape[0] >= 3:
            # Simple false color display using 3 HSI bands
            band_indices = [min(50, hsi_tree.shape[0]-1), 
                           min(25, hsi_tree.shape[0]-1), 
                           min(15, hsi_tree.shape[0]-1)]
            hsi_display = np.stack([hsi_tree[i] for i in band_indices], axis=-1)
            # Normalize for display
            hsi_display = hsi_display / (hsi_display.max() + 1e-10)
        else:
            # If fewer than 3 bands, just use first band grayscale
            hsi_display = hsi_tree[0]
            
        axes[1].imshow(hsi_display)
        axes[1].set_title(f'HSI Tree Crown {hsi_tree.shape}')
        axes[1].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    return rgb_tree, hsi_tree

def run_deepforest(rgb_tile, save_path, patch_size=400, patch_overlap=0.25):
    """
    Predict tree crowns using a pretrained deepforest model given a rgb tile
    Args:
        rgb_tile: Path to RGB tile image
        save_path: Path to save the predictions
        patch_size: DeepForest divides the big tile into smaller patches, default is 400
        patch_overlap: Overlap between patches, default is 0.25
    returns:
        filename: Path to the CSV of saved predictions
    """
    model = main.deepforest()
    # Load a pretrained tree detection model from Hugging Face
    model.load_model(model_name="weecology/deepforest-tree", revision="main")
    predicted_raster = model.predict_tile(rgb_tile, patch_size=patch_size, patch_overlap=patch_overlap)
    filename = os.path.join(save_path, 'predicted_boxes_' + os.path.basename(rgb_tile).split('.')[0]+'.csv')
    predicted_raster.to_csv(filename)
    print(f"Predicted boxes for {rgb_tile} saved to {filename}")
    return filename

if __name__ == "__main__":
    
    rgb_tiles_dir = '/blue/azare/riteshchowdhry/Macrosystems/Data_files/unlabeled_data/HARV/RGB/Mosaic/'
    all_rgb_tiles = glob.glob(os.path.join(rgb_tiles_dir, '*.tif'))
    
    # given a bbox (from deepforest) in RGB coordinates 
    rgb_file = '/blue/azare/riteshchowdhry/Macrosystems/Data_files/unlabeled_data/HARV/RGB/Mosaic/2022_HARV_7_734000_4709000_image.tif'
    hsi_file = '/blue/azare/riteshchowdhry/Macrosystems/Data_files/unlabeled_data/HARV/HSI/HSI_tif/2022_HARV_7_734000_4709000_image_hyperspectral.tif'
    xmin = 7872.0
    ymin = 2750.0
    xmax = 7950.0
    ymax = 2821.0
    tree_bbox = (xmin, ymin, xmax, ymax)
    # Extract tree crowns
    rgb_tree, hsi_tree = extract_tree_crowns(rgb_file, hsi_file, tree_bbox)