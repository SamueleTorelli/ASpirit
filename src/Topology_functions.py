import numpy as np
import networkx as nx
from scipy.spatial import distance_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist, squareform


def compute_primary_path(df):
    """
    Computes the 'primary path' of a set of 3D points:
    - longest among all shortest paths in the MST
    - if multiple, selects the one with minimum deflection
    """
    points = df[['X', 'Y', 'Z']].to_numpy()
    
    # --- 1. Compute MST ---
    D = distance_matrix(points, points)
    mst = minimum_spanning_tree(D).toarray()

    # --- 2. Build graph ---
    G = nx.Graph()
    for i in range(len(points)):
        for j in range(len(points)):
            if mst[i, j] > 0:
                G.add_edge(i, j, weight=mst[i, j])

    # --- 3. Compute all pairs shortest paths ---
    lengths = dict(nx.all_pairs_dijkstra_path_length(G))
    paths = dict(nx.all_pairs_dijkstra_path(G, weight='distance'))

    # --- 4. Find all paths with maximum length ---
    max_len = 0
    max_paths = []
    for i in lengths:
        for j in lengths[i]:
            l = lengths[i][j]
            if l > max_len:
                max_len = l
                max_paths = [paths[i][j]]
            elif l == max_len:
                max_paths.append(paths[i][j])

    # --- 5. If multiple, choose path with minimum deflection ---
    def compute_deflection(path_pts):
        """
        Compute a simple deflection measure:
        sum of angles between consecutive segments
        """
        if len(path_pts) < 3:
            return 0
        def angle(v1, v2):
            cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1)*np.linalg.norm(v2))
            cos_theta = np.clip(cos_theta, -1, 1)
            return np.arccos(cos_theta)
        total_angle = 0
        for k in range(1, len(path_pts)-1):
            v1 = path_pts[k] - path_pts[k-1]
            v2 = path_pts[k+1] - path_pts[k]
            total_angle += angle(v1, v2)
        return total_angle

    # Select path with minimum deflection
    best_path = max_paths[0]
    min_deflection = compute_deflection(points[best_path])
    for p in max_paths[1:]:
        defl = compute_deflection(points[p])
        if defl < min_deflection:
            best_path = p
            min_deflection = defl

    # --- 6. Return ---
    primary_path_points = points[best_path]
    return primary_path_points, max_len


def compute_primary_path_q_weigth(df):
    """
    Computes the 'primary path' of a set of 3D points:
    - longest among all shortest paths in the MST
    - if multiple, selects the one with minimum deflection
    """
    points = df[['X', 'Y', 'Z']].to_numpy()
    Q = df['Q'].to_numpy()
    
    def effective_cost(i, j):
        d = np.linalg.norm(points[i] - points[j])
        q_avg = (Q[i] + Q[j]) / 2.0
        alpha = 1.0   # tune this
        return d * (1 + alpha * q_avg)

    # --- 1. Compute MST ---
    N = len(points)
    D = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i != j:
                D[i, j] = effective_cost(i, j)

    mst = minimum_spanning_tree(D).toarray()

    # --- 2. Build graph ---
    G = nx.Graph()
    for i in range(len(points)):
        for j in range(len(points)):
            if mst[i, j] > 0:
                G.add_edge(i, j, weight=mst[i, j])

    # --- 3. Compute all pairs shortest paths ---
    lengths = dict(nx.all_pairs_dijkstra_path_length(G))
    paths = dict(nx.all_pairs_dijkstra_path(G, weight='distance'))

    # --- 4. Find all paths with maximum length ---
    max_len = 0
    max_paths = []
    for i in lengths:
        for j in lengths[i]:
            l = lengths[i][j]
            if l > max_len:
                max_len = l
                max_paths = [paths[i][j]]
            elif l == max_len:
                max_paths.append(paths[i][j])

    # --- 5. If multiple, choose path with minimum deflection ---
    def compute_deflection(path_pts):
        """
        Compute a simple deflection measure:
        sum of angles between consecutive segments
        """
        if len(path_pts) < 3:
            return 0
        def angle(v1, v2):
            cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1)*np.linalg.norm(v2))
            cos_theta = np.clip(cos_theta, -1, 1)
            return np.arccos(cos_theta)
        total_angle = 0
        for k in range(1, len(path_pts)-1):
            v1 = path_pts[k] - path_pts[k-1]
            v2 = path_pts[k+1] - path_pts[k]
            total_angle += angle(v1, v2)
        return total_angle

    # Select path with minimum deflection
    best_path = max_paths[0]
    min_deflection = compute_deflection(points[best_path])
    for p in max_paths[1:]:
        defl = compute_deflection(points[p])
        if defl < min_deflection:
            best_path = p
            min_deflection = defl

    # --- 6. Return ---
    primary_path_points = points[best_path]
    return primary_path_points, max_len




def reconstruct_path(df, primary_path_points, radius=40):
    """
    Smooth the primary path by replacing each point with the Q-weighted centroid
    of points within a given radius.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Must contain columns 'X', 'Y', 'Z', 'Q'.
    primary_path_points : np.ndarray
        Array of shape (N,3) containing points along the primary path.
    radius : float
        Neighborhood radius to consider for the barycentre (same units as X,Y,Z).
    
    Returns
    -------
    reconstructed_path : np.ndarray
        Smoothed path points (N,3).
    """
    # Build KDTree of all points
    pts = df[['X','Y','Z']].to_numpy()
    Q = df['Q'].to_numpy()
    tree = cKDTree(pts)

    smoothed_path = []

    for p in primary_path_points:
        # Find all points within radius
        idx = tree.query_ball_point(p, radius)
        if len(idx) == 0:
            # No points nearby: keep original
            smoothed_path.append(p)
        else:
            # Energy-weighted centroid
            local_pts = pts[idx]
            local_Q = Q[idx]
            centroid = np.average(local_pts, axis=0, weights=local_Q)
            smoothed_path.append(centroid)

    return np.array(smoothed_path)

def mean_filter_path(path_points, window=5):
    """
    Applies a mean (moving average) filter to a 3D path.

    Parameters
    ----------
    path_points : (N,3) array
        Input polyline (e.g., smoothed_path).
    window : int
        Number of points in the moving average window (must be odd).

    Returns
    -------
    filtered : (N,3) array
        Mean-filtered path.
    """
    if window < 1:
        return path_points.copy()

    if window % 2 == 0:
        raise ValueError("Window size must be odd.")

    N = len(path_points)
    half = window // 2

    filtered = np.zeros_like(path_points)

    # Pad edges to preserve length
    padded = np.pad(path_points, ((half, half), (0, 0)), mode="edge")

    # Mean filter
    for i in range(N):
        filtered[i] = padded[i:i+window].mean(axis=0)

    return filtered



def compute_min_axis_spacing(df):
    """
    Computes the minimum spacing between unique coordinate values
    along X, Y, and Z.
    
    Returns
    -------
    (dx_min, dy_min, dz_min)
    """

    def min_diff(values):
        """Return the smallest difference between sorted unique values."""
        uniq = np.sort(np.unique(values))
        if len(uniq) < 2:
            return np.nan
        diffs = np.diff(uniq)
        return np.min(diffs)

    dx = min_diff(df["X"].to_numpy())
    dy = min_diff(df["Y"].to_numpy())
    dz = min_diff(df["Z"].to_numpy())

    return dx, dy, dz


def remove_isolated_points(df, radius=1.2):
    """
    Removes points that have no neighbors within a given radius.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Must contain columns 'X', 'Y', 'Z'.
    radius : float
        Distance threshold to consider a point 'connected'.

    Returns
    -------
    df_filtered : pandas.DataFrame
        DataFrame with isolated points removed.
    """

    dx, dy, dz = compute_min_axis_spacing(df)

    # Build KD-tree
    points = df[['X', 'Y', 'Z']].to_numpy()

    points[:, 0] /= dx  # X
    points[:, 1] /= dy  # Y
    points[:, 2] /= dz  # Z

    tree = cKDTree(points)

    # Count neighbors for each point (including itself, so ≥2 means not isolated)
    neighbor_lists = tree.query_ball_tree(tree, r=radius)

    # A point is isolated if len(neighbors) == 1 (only itself)
    mask = np.array([len(neigh) > 2 for neigh in neighbor_lists])

    # Return only points that are not isolated
    return df[mask].copy()

