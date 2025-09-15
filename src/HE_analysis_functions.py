import os
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN

module_dir = os.path.abspath('.')

from analysis_functions import MapPar
import parser_fun as pf
import analysis_functions as af



def remove_scatter_and_reweight(df: pd.DataFrame) -> pd.DataFrame:

    # Compute Eslice for each (event, Z)
    eslice = df.groupby(['event', 'Z'])['E'].sum().rename("Eslice")
    df = df.merge(eslice, on=['event', 'Z'])

    # Mask for signal rows (cluster >= 0)
    signal_mask = df['cluster'] >= 0

    # Compute Qsum for signal rows grouped by (event, Z)
    qsum = df[signal_mask].groupby(['event', 'Z'])['Q'].sum().rename("Qsum")

    # Merge Qsum back to original DataFrame; use left join to preserve all rows
    df = df.merge(qsum, on=['event', 'Z'], how='left')

    # Compute Erw only for signal rows
    df['Erw'] = np.where(
        signal_mask,
        df['Eslice'] * df['Q'] / df['Qsum'],
        np.nan
    )

    df = df.drop(columns=['Eslice', 'Qsum'])      
    return df


def correct_Hits(
    df: pd.DataFrame,
    krmap: MapPar,
    rmax: float = 480.0,
    zmax: float = 1350.0,
    var: str = 'E'
) -> pd.DataFrame:
    """
    Apply position-based E correction using a 3D histogram map, safely guarding against divide-by-zero.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with columns 'X', 'Y', 'Z', 'E'.
    krmap : MapPar
        Correction map object containing hmap and bin edges.
    rmax : float, optional
        Max radius cut.
    zmax : float, optional
        Max drift time cut.

    Returns
    -------
    pd.DataFrame
        DataFrame with added 'Ec' and 'corrections' columns.
    """
    
    x_vals = df['X'].to_numpy()
    y_vals = df['Y'].to_numpy()
    z_vals = df['Z'].to_numpy()
    ene_vals = df[var].to_numpy()

    x_bins = np.digitize(x_vals, krmap.xedges) - 1
    y_bins = np.digitize(y_vals, krmap.yedges) - 1
    z_bins = np.digitize(z_vals, krmap.zedges) - 1

    hmap = krmap.hmap
    if hmap is None:
        raise ValueError("krmap.hmap is None. You must set krmap.hmap before calling correct_Ene().")

    valid_mask = (
        (x_bins >= 0) & (x_bins < hmap.shape[0]) &
        (y_bins >= 0) & (y_bins < hmap.shape[1]) &
        (z_bins >= 0) & (z_bins < hmap.shape[2])
    )

    corrections = np.full_like(ene_vals, np.nan)
    corrections[valid_mask] = hmap[x_bins[valid_mask], y_bins[valid_mask], z_bins[valid_mask]]

    # Avoid divide-by-zero and negative corrections
    safe_mask = corrections > 1e-6
    norm_ene = np.full_like(ene_vals, np.nan)
    norm_ene[safe_mask] = ene_vals[safe_mask] / corrections[safe_mask]

    df['Ec'] = norm_ene

    df = df.dropna(subset=['Ec'])

    return df



def clusterize_hits(df_pe_peak: pd.DataFrame, eps=2.3, npt=5)-> pd.DataFrame:

    """
    Cluster hits in 3D space for each event using DBSCAN.
    
    The coordinates are scaled to account for detector geometry differences 
    in samplig 
    
    Parameters
    ----------
    df_pe_peak : pd.DataFrame
    DataFrame containing hit information with columns 'X', 'Y', 'Z', and 'event'.
    
    Returns
    -------
    pd.DataFrame
    Modified DataFrame with an added 'cluster' column indicating the cluster label 
    for each hit (-1 for noise).
    """
    
    a = 14.55  # XY scale
    b = 3.7  # Z scale

    # Pre-allocate array for cluster labels
    cluster_labels = np.full(len(df_pe_peak), -9999, dtype=int)

    # Get values once (faster than repeatedly accessing DataFrame columns)
    coords = df_pe_peak[['X', 'Y', 'Z']].to_numpy()
    events = df_pe_peak['event'].to_numpy()
    
    # Use np.unique to get sorted event IDs
    unique_events = np.unique(events)
    
    for event_id in unique_events:
        mask = (events == event_id)
        X = coords[mask].copy()
        
        # Scale
        X[:, :2] /= a
        X[:, 2] /= b
        
        labels = DBSCAN(eps=eps, min_samples=npt).fit_predict(X)
        cluster_labels[mask] = labels

    df_pe_peak['cluster'] = cluster_labels

    return df_pe_peak


def compute_cluster_stats(df: pd.DataFrame)-> pd.DataFrame:

    slice_cluster_stats=[]
    
    for ev, df_ev in df.groupby('event'):
        z_bounds = df_ev.groupby('cluster')['Z'].agg(['min', 'max'])
        z_bounds['min'] -= 1
        z_bounds['max'] += 1
        z_slices = np.sort(np.unique(np.concatenate([z_bounds['min'].values, z_bounds['max'].values])))
        
        for i in range(len(z_slices) - 1):
            z_low = z_slices[i]
            z_high = z_slices[i + 1]
            slice_df = df_ev[(df_ev['Z'] >= z_low) & (df_ev['Z'] <= z_high)]
            
            for cluster_id, sub_df in slice_df.groupby('cluster'):
                
                stats = {
                    'event': ev,
                    'Z_min': z_low,
                    'Z_max': z_high,
                    'cluster': cluster_id,
                    'Ec_sum': sub_df['Ec'].sum(),
                    'X_mean': df_ev[df_ev['cluster'] == cluster_id]['X'].mean(),
                    'X_min': df_ev[df_ev['cluster'] == cluster_id]['X'].min(),
                    'X_max': df_ev[df_ev['cluster'] == cluster_id]['X'].max(),
                    'Y_mean': df_ev[df_ev['cluster'] == cluster_id]['Y'].mean(),
                    'Y_min': df_ev[df_ev['cluster'] == cluster_id]['Y'].min(),
                    'Y_max': df_ev[df_ev['cluster'] == cluster_id]['Y'].max()
                }
                slice_cluster_stats.append(stats)

    df_slices_clustered = pd.DataFrame(slice_cluster_stats)

    return df_slices_clustered

def preprocess_df_hits(folderlist: [str],
                       ev_list: [int],
                       path_to_kr_map: str)-> pd.DataFrame:

    kr_map = af.load_kr_map(path_to_kr_map)
    print('recomended to pass 1 ldc per time in the list!! [../ldc1/,../ldc2,...]')

    dfs =[]
    
    for a_folder in folderlist:
        print("opening...")
        df_pe_peak = pf.open_reco_event(a_folder,ev_list)
        df_pe_peak = df_pe_peak[(df_pe_peak['Z']>0) & (df_pe_peak['Z']<1500)]

        print("clustering...")
        df_pe_peak = clusterize_hits(df_pe_peak)

        print("correcting hits...")
        df_pe_peak = correct_Hits(df_pe_peak,kr_map)

        print("calculating stats...")
        df_pe_peak = compute_cluster_stats(df_pe_peak)

        dfs.append(df_pe_peak)
        
    return pd.concat(dfs, ignore_index=True)





