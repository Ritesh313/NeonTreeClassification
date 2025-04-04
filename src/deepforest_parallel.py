import os
import glob
import time
from dask_jobqueue import SLURMCluster
from dask.distributed import Client, wait


from deepforest import main

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
    print(f"\nPredicted boxes for {rgb_tile} saved to {filename}")
    return filename

if __name__ == "__main__":
    
    rgb_tiles_dir = '/blue/azare/riteshchowdhry/Macrosystems/Data_files/unlabeled_data/HARV/RGB/Mosaic/'
    all_rgb_tiles = glob.glob(os.path.join(rgb_tiles_dir, '*.tif'))
    output_dir = "/blue/azare/riteshchowdhry/Macrosystems/Data_files/unlabeled_data/HARV/deepforest_crowns"
    df_patch_size = 400
    df_patch_overlap = 0.25
    num_workers = 40 # be careful with this, memory will be allocated for each worker, so if you have 40 workers and each worker has 35GB of memory, you will need 1.4TB of memory on the node.
    
    slurm_args = [
        "--job-name=deepforest",
        "--account=azare",
        "--mail-type=END,FAIL",
        "--mail-user=riteshchowdhry@ufl.edu",
        "--output=/home/riteshchowdhry/logs/macrosystems/dask_deepforest/harv2022%j.out",
        "--partition=gpu",
        "--constraint=ai",
        "--gpus=1",
        "--time=24:00:00",  
    ]
    
    cluster = SLURMCluster(
        cores=6,
        memory='35GB',
        processes=1,
        walltime='24:00:00',
        scheduler_options={"dashboard_address": ":8787"},
        job_extra_directives=slurm_args,
        local_directory='/home/riteshchowdhry/logs/macrosystems/dask_deepforest/',
        death_timeout=300,
    )
    
    print("Job script template:")
    print(cluster.job_script())
    
    print(f"Scaling cluster to {num_workers} workers")
    cluster.scale(num_workers)
    
    # Allow time for workers to start
    print("Waiting for workers to start...")
    time.sleep(30)

    # Connect client to cluster
    print("Connecting client to cluster")
    dask_client = Client(cluster)
    print(f"Dashboard link: {dask_client.dashboard_link}")
    
    futures = []
    for i, rgb_tile in enumerate(all_rgb_tiles):
        print(f"Submitting task {i+1}/{len(all_rgb_tiles)}: {rgb_tile}")
        future = dask_client.submit(run_deepforest, rgb_tile, output_dir, df_patch_size, df_patch_overlap)
        futures.append(future)
    
    
    # Wait for tasks with progress reporting
    print(f"Waiting for {len(futures)} tasks to complete...")
    completed = 0
    for future in futures:
        try:
            result = future.result()
            completed += 1
            print(f"Task completed: {completed}/{len(futures)}")
        except Exception as e:
            print(f"{completed} Task failed with error: {str(e)}")
    
    print("All tasks completed or failed.")
    
    # Close connections
    print("Closing Dask client")
    dask_client.close()
    print("Closing cluster")
    cluster.close()
    print("Script completed.")
