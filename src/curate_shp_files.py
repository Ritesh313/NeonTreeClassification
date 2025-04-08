import geopandas as gpd
import pandas as pd
import os
import glob
import re
import shutil

def consolidate_files(parent_dir, destination_dir):
    """
    Consolidate shapefiles from subdirectories into a single directory.
    """
    errors = []
    # List to keep track of files moved
    files_moved = []
    # Walk through all subdirectories
    for root, dirs, files in os.walk(parent_dir):
        # Skip the parent directory itself and the destination directory
        if root == parent_dir or root == destination_dir:
            continue
            
        print(f"Checking directory: {root}")
        
        # Find all shapefile-related files in this directory
        # (looking for all extensions: .shp, .shx, .dbf, .prj, .cpg)
        all_files = []
        for ext in ['.shp', '.shx', '.dbf', '.prj', '.cpg']:
            all_files.extend(glob.glob(os.path.join(root, f'*{ext}')))
        
        # Copy each file to the destination
        for file_path in all_files:
            try:
                file_name = os.path.basename(file_path)
                dest_path = os.path.join(destination_dir, file_name)
                
                # Check if file already exists at destination
                if os.path.exists(dest_path):
                    print(f"Warning: File {file_name} already exists in destination. Ignoring.")
                    continue
                
                shutil.copy2(file_path, dest_path)
                files_moved.append(file_path)
                print(f"Copied: {file_path} -> {dest_path}")
            except Exception as e:
                errors.append((file_path, str(e)))
                print(f"Error copying {file_path}: {e}")

    # Print summary
    print("\nOperation complete!")
    print(f"Total files moved: {len(files_moved)}")
    print(f"Errors encountered: {len(errors)}")

    if errors:
        print("\nFiles that couldn't be copied:")
        for file_path, error in errors:
            print(f"- {file_path}: {error}")

    # Count files in destination by extension
    print("\nFiles in destination directory by extension:")
    for ext in ['.shp', '.shx', '.dbf', '.prj', '.cpg']:
        count = len(glob.glob(os.path.join(destination_dir, f'*{ext}')))
        print(f"{ext}: {count} files")

# Function to extract site, plot, and year from filename
def extract_metadata(filename):
    # Assuming filename format like: SOAP_052_2019.shp
    parts = os.path.basename(filename).split('.')[0].split('_')
    if len(parts) >= 3:
        site = parts[0]
        plot = parts[1]
        year = parts[2]
        return site, plot, year
    return "unknown", "unknown", "unknown"

# Function to calculate centroid coordinates
def get_centroid_coords(gdf):
    # Handle potential MultiPolygon geometries by getting the centroid of all geometries
    bounds = gdf.total_bounds  # minx, miny, maxx, maxy
    center_x = (bounds[0] + bounds[2]) / 2
    center_y = (bounds[1] + bounds[3]) / 2
    
    # Get the UTM zone if available from projection info
    utm_zone = "unknown"
    if 'utm' in gdf.crs.name.lower():
        utm_match = re.search(r'utm zone (\d+)', gdf.crs.name.lower())
        if utm_match:
            utm_zone = utm_match.group(1)
    
    return center_x, center_y, bounds, utm_zone

def process_shp_files(consolidated_dir):
    """
    Process shapefiles in the consolidated directory to extract metadata and coordinates.
    """
    # Find all shapefiles
    shp_files = glob.glob(os.path.join(consolidated_dir, '*.shp'))

    # List to store metadata
    sites_data = []

    # Process each shapefile
    for shp_file in shp_files:
        try:
            # Extract metadata from filename
            site, plot, year = extract_metadata(shp_file)
            
            # Read the shapefile
            gdf = gpd.read_file(shp_file)
            
            # Get centroid coordinates and bounds
            center_x, center_y, bounds, utm_zone = get_centroid_coords(gdf)
            
            # Store the data
            sites_data.append({
                'filename': os.path.basename(shp_file),
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
                'crs': str(gdf.crs),
                'num_polygons': len(gdf)
            })
            
            print(f"Processed: {os.path.basename(shp_file)}")
            
        except Exception as e:
            print(f"Error processing {shp_file}: {e}")

    # Create a DataFrame with the collected data
    sites_df = pd.DataFrame(sites_data)

    # Display the DataFrame
    print("\nSites and Coordinates Summary:")
    print(sites_df.head())

    # Save to CSV
    csv_path = os.path.join(consolidated_dir, 'neon_sites_coordinates.csv')
    sites_df.to_csv(csv_path, index=False)
    print(f"\nSaved coordinates to: {csv_path}")

    # Additional analysis - group by site
    print("\nNumber of plots per site:")
    site_counts = sites_df.groupby('site').size()
    print(site_counts)

    # Check which coordinate reference system (CRS) is used for each site
    print("\nCoordinate Reference Systems used:")
    crs_counts = sites_df.groupby(['site', 'crs']).size().reset_index(name='count')
    print(crs_counts)
    
    total_polygons = sites_df['num_polygons'].sum()
    print(f"\nTotal number of polygons across all shapefiles: {total_polygons}")

if __name__ == "__main__":
    parent_dir = '/blue/azare/riteshchowdhry/Macrosystems/Data_files/hand_annotated_neon/'
    destination_dir = '/blue/azare/riteshchowdhry/Macrosystems/Data_files/hand_annotated_neon/consolidated_dir'
    
    # Create destination directory if it doesn't exist
    os.makedirs(destination_dir, exist_ok=True)
    
    # If required, consolidate files from parent directory to destination directory
    consolidate_files(parent_dir, destination_dir)
    
    # Process the consolidated shapefiles and create a csv with all the metadata
    process_shp_files(destination_dir)
    