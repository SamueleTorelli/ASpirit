import os
import pandas as pd

def merge_hdfs_multi(folderdata_list):
    """
    Merge HDF5 datasets from multiple folders into a single pandas DataFrame.

    Parameters:
    -----------
    folderdata_list : list of str
        A list of directory paths. Each directory is searched recursively for HDF5 files.

    Returns:
    --------
    pd.DataFrame
        A single DataFrame obtained by concatenating all '/DST/Events' datasets
        found in the `.h5` or `.hdf5` files located in the specified folders.
    """
    dfs = []

    # Loop through each folder in the input list
    for folderdata in folderdata_list:
        # Recursively walk through all files in the folder
        for root, _, files in os.walk(folderdata):
            for f in files:
                # Check if the file is an HDF5 file
                if f.endswith(('.h5', '.hdf5')):
                    file_path = os.path.join(root, f)
                    # Read the HDF5 dataset at the path '/DST/Events'
                    df = pd.read_hdf(file_path, '/DST/Events')
                    dfs.append(df)

    # Concatenate all collected DataFrames into one
    return pd.concat(dfs, ignore_index=True)


def merge_hdfs_multi_reco(folderdata_list, event_list):
    """
    Merge HDF5 datasets from multiple folders into a single pandas DataFrame.

    Parameters:
    -----------
    folderdata_list : list of str
        A list of directory paths. Each directory is searched recursively for HDF5 files.

    Returns:
    --------
    pd.DataFrame
        A single DataFrame obtained by concatenating all '/DST/Events' datasets
        found in the `.h5` or `.hdf5` files located in the specified folders.
    """
    dfs = []

    # Loop through each folder in the input list
    for folderdata in folderdata_list:
        # Recursively walk through all files in the folder
        for root, _, files in os.walk(folderdata):
            for f in files:
                # Check if the file is an HDF5 file
                if f.endswith(('.h5', '.hdf5')):
                    file_path = os.path.join(root, f)
                    # Read the HDF5 dataset at the path '/DST/Events'
                    print(file_path)
                    df = pd.read_hdf(file_path, '/RECO/Events')
                    df = df[df["event"].isin(event_list)]
                    dfs.append(df)

    # Concatenate all collected DataFrames into one
    return pd.concat(dfs, ignore_index=True)
