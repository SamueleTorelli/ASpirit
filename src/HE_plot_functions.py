import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from plot_functions import color_sequence
from sklearn.cluster import DBSCAN

module_dir = os.path.abspath('.')

import HE_analysis_functions as afHE


def display_event_cluster(df_reco_event):

    color_sequence = ("k", "m", "g", "b", "r",
                  "gray", "aqua", "gold", "lime", "purple",
                  "brown", "lawngreen", "tomato", "lightgray", "lightpink")

    # Group total energy for coloring
    df_grouped_xy = df_reco_event.groupby(['X', 'Y'], as_index=False)['E'].sum()
    df_grouped_zy = df_reco_event.groupby(['Z', 'Y'], as_index=False)['E'].sum()
    df_grouped_xz = df_reco_event.groupby(['X', 'Z'], as_index=False)['E'].sum()
    
    # Group by cluster (includes Scattered)
    df_clustered_xy = df_reco_event.groupby(['X', 'Y', 'cluster'], as_index=False)['E'].sum()
    df_clustered_zy = df_reco_event.groupby(['Z', 'Y', 'cluster'], as_index=False)['E'].sum()
    df_clustered_xz = df_reco_event.groupby(['X', 'Z', 'cluster'], as_index=False)['E'].sum()
    
    # Create 3x2 plot
    fig, axes = plt.subplots(3, 2, figsize=(30, 30), dpi=180)
    
    # --- TOP LEFT: All hits X vs Y ---
    sc0 = axes[0, 0].scatter(df_grouped_xy['X'], df_grouped_xy['Y'], c=df_grouped_xy['E'],
                             cmap='jet', s=12, marker='o')
    axes[0, 0].set_title("All Hits: Q vs X,Y")
    axes[0, 0].set_xlabel("X [mm]")
    axes[0, 0].set_ylabel("Y [mm]")
    axes[0, 0].set_xlim(-500, 500)
    axes[0, 0].set_ylim(-500, 500)
    axes[0, 0].set_aspect('equal')
    axes[0, 0].grid()
    axes[0, 0].set_facecolor("whitesmoke")
    fig.colorbar(sc0, ax=axes[0, 0], label="Total Q")
    
    # --- TOP RIGHT: Clustered hits X vs Y ---
    for cl in sorted(df_clustered_xy['cluster'].unique()):
        cluster_df = df_clustered_xy[df_clustered_xy['cluster'] == cl]
        color = color_sequence[-3] if cl == -1 else color_sequence[cl]
        label = 'Scattered' if cl == -1 else f'Cluster {cl}'
        axes[0, 1].scatter(cluster_df['X'], cluster_df['Y'], s=12, marker='o', label=label, c=color)
    axes[0, 1].set_title("Clustered Hits: Q vs X,Y")
    axes[0, 1].set_xlabel("X [mm]")
    axes[0, 1].set_ylabel("Y [mm]")
    axes[0, 1].set_xlim(-500, 500)
    axes[0, 1].set_ylim(-500, 500)
    axes[0, 1].set_aspect('equal')
    axes[0, 1].set_facecolor("whitesmoke")
    axes[0, 1].grid()
    axes[0, 1].legend(markerscale=2, fontsize='small')
    
    # --- MIDDLE LEFT: All hits Z vs Y ---
    sc2 = axes[1, 0].scatter(df_grouped_zy['Z'], df_grouped_zy['Y'], c=df_grouped_zy['E'],
                             cmap='jet', s=12, marker='o')
    axes[1, 0].set_title("All Hits: Q vs Z,Y")
    axes[1, 0].set_xlabel("Z [mm]")
    axes[1, 0].set_ylabel("Y [mm]")
    axes[1, 0].set_ylim(-500, 500)
    axes[1, 0].set_facecolor("whitesmoke")
    axes[1, 0].grid()
    fig.colorbar(sc2, ax=axes[1, 0], label="Total Q")
    
    # --- MIDDLE RIGHT: Clustered hits Z vs Y ---
    for cl in sorted(df_clustered_zy['cluster'].unique()):
        cluster_df = df_clustered_zy[df_clustered_zy['cluster'] == cl]
        color = color_sequence[-3] if cl == -1 else color_sequence[cl]
        label = 'Scattered' if cl == -1 else f'Cluster {cl}'
        axes[1, 1].scatter(cluster_df['Z'], cluster_df['Y'], s=12, marker='o', label=label, c=color)
    axes[1, 1].set_title("Clustered Hits: Q vs Z,Y")
    axes[1, 1].set_xlabel("Z [mm]")
    axes[1, 1].set_ylabel("Y [mm]")
    axes[1, 1].set_ylim(-500, 500)
    axes[1, 1].set_facecolor("whitesmoke")
    axes[1, 1].grid()
    axes[1, 1].legend(markerscale=2, fontsize='small')
    
    # --- BOTTOM LEFT: All hits X vs Z ---
    sc4 = axes[2, 0].scatter(df_grouped_xz['Z'], df_grouped_xz['X'], c=df_grouped_xz['E'],
                             cmap='jet', s=12, marker='o')
    axes[2, 0].set_title("All Hits: Q vs X,Z")
    axes[2, 0].set_xlabel("X [mm]")
    axes[2, 0].set_ylabel("Z [mm]")
    axes[2, 0].set_ylim(-500, 500)
    axes[2, 0].set_facecolor("whitesmoke")
    axes[2, 0].grid()
    fig.colorbar(sc4, ax=axes[2, 0], label="Total Q")
    
    # --- BOTTOM RIGHT: Clustered hits X vs Z ---
    for cl in sorted(df_clustered_xz['cluster'].unique()):
        cluster_df = df_clustered_xz[df_clustered_xz['cluster'] == cl]
        color = color_sequence[-3] if cl == -1 else color_sequence[cl]
        label = 'Scattered' if cl == -1 else f'Cluster {cl}'
        axes[2, 1].scatter(cluster_df['Z'],cluster_df['X'], s=12, marker='o', label=label, c=color)
    axes[2, 1].set_title("Clustered Hits: Q vs X,Z")
    axes[2, 1].set_xlabel("Z [mm]")
    axes[2, 1].set_ylabel("X [mm]")
    #axes[2, 1].set_xlim(-500, 500)
    axes[1, 1].set_ylim(-500, 500)
    axes[2, 1].set_facecolor("whitesmoke")
    axes[2, 1].grid()
    axes[2, 1].legend(markerscale=2, fontsize='small')

    plt.tight_layout()
    plt.show()


