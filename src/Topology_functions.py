import numpy as np
import networkx as nx
from scipy.spatial import distance_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist, squareform


def compute_primary_path(df, thresh):
    """
    Computes the 'primary path' of a set of 3D points:
    - longest among all shortest paths in the MST
    - if multiple, selects the one with minimum deflection
    """
    points = df[['X', 'Y', 'Z']].to_numpy()
    
    # --- 1. Compute MST ---
    D = distance_matrix(points, points)
    D[D > thresh] = 999
    mst = minimum_spanning_tree(D).toarray()

    # --- 2. Build graph ---
    G = nx.Graph()
    for i in range(len(points)):
        for j in range(len(points)):
            if mst[i, j] > 0:
                G.add_edge(i, j, weight=mst[i, j])

    # --- 3. Compute all pairs shortest paths ---
    lengths = dict(nx.all_pairs_dijkstra_path_length(G))
    paths = dict(nx.all_pairs_dijkstra_path(G))

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


def compute_primary_path_fast(df, thresh):
    """
    Compute the centerline through a 3D point cloud using:
    1. Minimum Spanning Tree (MST)
    2. Tree diameter (two-pass Dijkstra)
    
    Returns:
        path_indices: ordered list of indices along the best path
        length: total geodesic length of the path
        endpoints: (start_node, end_node)
    """

    points = df[['X', 'Y', 'Z']].to_numpy()
    
    # --- 1. Distance matrix ----
    D = distance_matrix(points, points)
    D[D > thresh] = 999

    # --- 2. Compute MST ----
    mst = minimum_spanning_tree(D).toarray()

    # --- 3. Build graph from MST ----
    G = nx.Graph()
    N = len(points)
    for i in range(N):
        for j in range(N):
            if mst[i, j] > 0:
                G.add_edge(i, j, weight=mst[i, j])

    # --- 4. First Dijkstra: from arbitrary node (0) to find farthest node A ---
    lengths_0 = nx.single_source_dijkstra_path_length(G, 0)
    A = max(lengths_0, key=lengths_0.get)

    # --- 5. Second Dijkstra: from A to find farthest node B (diameter endpoint) ---
    lengths_A, paths_A = nx.single_source_dijkstra(G, A)
    B = max(lengths_A, key=lengths_A.get)

    # --- 6. The diameter path is the path from A to B ---
    best_path = paths_A[B]
    best_length = lengths_A[B]

    return df[['X', 'Y', 'Z']].to_numpy()[best_path]


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


def reconstruct_path_ellipse(df, primary_path_points, ellipse_size):
    """
    Smooth the primary path by replacing each point with the Q-weighted centroid
    of points inside an ellipsoidal neighborhood (axis `a` along Z, `b` on X/Y).

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain columns 'X', 'Y', 'Z', 'Q'.
    primary_path_points : np.ndarray
        Array of shape (N,3) containing points along the primary path.
    a : float
        Semi-axis of the ellipsoid along Z.
    b : float
        Semi-axis of the ellipsoid along X and Y (assumed equal for both).

    Returns
    -------
    reconstructed_path : np.ndarray
        Smoothed path points (N,3).
    """
    # Build KDTree in scaled coordinate system so ellipsoid check becomes spherical
    pts = df[['X', 'Y', 'Z']].to_numpy()
    Q = df['Q'].to_numpy()

    a=ellipse_size[0]
    b=ellipse_size[1]

    scaled_pts = np.column_stack((pts[:, 0] / b, pts[:, 1] / b, pts[:, 2] / a))
    tree = cKDTree(scaled_pts)

    smoothed_path = []

    for p in primary_path_points:
        scaled_p = np.array([p[0] / b, p[1] / b, p[2] / a])
        # radius 1 in scaled space corresponds to ellipsoid in original space
        idx = tree.query_ball_point(scaled_p, 1.0)
        if len(idx) == 0:
            smoothed_path.append(p)
        else:
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


def prune_edges(df, distance_threshold=1.5):
    """
    Iteratively remove edge points without breaking connectivity.
    
    Parameters
    ----------
    df : pd.DataFrame
        Must have columns ['X','Y','Z','Q'].
    distance_threshold : float
        Maximum distance to consider points as neighbors.
        
    Returns
    -------
    pd.DataFrame
        Pruned DataFrame.
    """
    # Copy input
    df_pruned = df.copy()
    
    # Build k-d tree for neighbors
    coords = df_pruned[['X','Y','Z']].values
    tree = cKDTree(coords)
    
    # Build graph: edges between points within distance_threshold
    G = nx.Graph()
    for i, coord in enumerate(coords):
        neighbors = tree.query_ball_point(coord, r=distance_threshold)
        for j in neighbors:
            if i != j:
                G.add_edge(i, j)
    
    # Function to check if a node is a boundary (has less than max neighbors)
    def is_edge_node(node):
        return len(list(G.neighbors(node))) < 6  # heuristic
    
    # Iteratively remove edge nodes if connectivity is preserved
    nodes_to_check = list(G.nodes)
    while True:
        removed_any = False
        for node in nodes_to_check:
            if node not in G:
                continue
            if is_edge_node(node):
                # Remove node and check connectivity
                G.remove_node(node)
                if nx.is_connected(G):
                    removed_any = True
                else:
                    # Restore
                    G.add_node(node)
                    # Reconnect edges
                    coord = coords[node]
                    neighbors = tree.query_ball_point(coord, r=distance_threshold)
                    for j in neighbors:
                        if j != node and j in G:
                            G.add_edge(node, j)
        if not removed_any:
            break
    
    # Return pruned DataFrame
    remaining_indices = list(G.nodes)
    return df_pruned.iloc[remaining_indices].reset_index(drop=True)


def skeletonize_point_cloud_from_df(df, distance_threshold=2.0, min_degree=1):
    """
    Skeletonize 3D point cloud from a dataframe with columns ['X','Y','Z','Q'].
    """
    coords = df[['X','Y','Z']].to_numpy()
    Q = df['Q'].to_numpy() if 'Q' in df.columns else None
    
    N = len(coords)
    G = nx.Graph()

    # Add nodes with indices 0..N-1
    for i in range(N):
        G.add_node(i, pos=coords[i], Q=Q[i] if Q is not None else 1.0)

    # Build KD-tree
    tree = cKDTree(coords)

    for i, pt in enumerate(coords):
        neighbors = tree.query_ball_point(pt, r=distance_threshold)
        for j in neighbors:
            if i >= j:
                continue
            dist = np.linalg.norm(pt - coords[j])
            weight = dist #/ (G.nodes[i]['Q'] + G.nodes[j]['Q'])
            G.add_edge(i, j, weight=weight)

    # Extremities and shortest paths (same as before)
    extremities = [n for n in G.nodes if G.degree[n] <= min_degree]
    if len(extremities) < 2:
        extremities = list(G.nodes)
    
    skeleton_edges = set()
    for i, src in enumerate(extremities):
        for dst in extremities[i+1:]:
            try:
                path = nx.shortest_path(G, source=src, target=dst, weight='distance')
                skeleton_edges.update([(path[k], path[k+1]) for k in range(len(path)-1)])
            except nx.NetworkXNoPath:
                continue
    
    G_skel = nx.Graph()
    G_skel.add_nodes_from(range(N))
    G_skel.add_edges_from(skeleton_edges)
    
    # Optional pruning leaves
    while True:
        leaves = [n for n in G_skel.nodes if G_skel.degree[n] == 1]
        if not leaves:
            break
        G_skel.remove_nodes_from(leaves)
    
    skeleton_nodes = coords[list(G_skel.nodes)]
    skeleton_edges = list(G_skel.edges)
    
    return skeleton_nodes, skeleton_edges


def filter_points(filtered_nodes, max_distance=5.0):
    """
    Filter consecutive points in ordered array - remove point if it's too far from previous point
    """
    if len(filtered_nodes) == 0:
        return filtered_nodes
    
    filtered_result = [filtered_nodes[0]]  # Always keep first point
    
    for i in range(1, len(filtered_nodes)):
        current_point = filtered_nodes[i]
        previous_point = filtered_result[-1]  # Last point we kept
        
        dist = np.linalg.norm(current_point - previous_point)
        
        if dist <= max_distance:
            filtered_result.append(current_point)
        # else: skip this point (don't add it to filtered_result)
    
    return np.array(filtered_result)


def sort_skeleton_points(skeleton_nodes, skeleton_edges):
    """
    Sort skeleton points in order between the two extremities
    """
    if len(skeleton_nodes) == 0:
        return skeleton_nodes
    
    # Create graph from skeleton
    G = nx.Graph()
    for i in range(len(skeleton_nodes)):
        G.add_node(i, pos=skeleton_nodes[i])
    G.add_edges_from(skeleton_edges)
    
    # Find extremities (degree 1 nodes)
    extremities = [n for n in G.nodes if G.degree(n) == 1]
    
    if len(extremities) != 2:
        # If not exactly 2 extremities, can't sort linearly
        return skeleton_nodes
    
    # Find path between the two extremities
    start, end = extremities[0], extremities[1]
    path = nx.shortest_path(G, source=start, target=end)
    
    # Return nodes in order along the path
    sorted_nodes = np.array([skeleton_nodes[i] for i in path])
    return sorted_nodes

# Usage:
#skeleton_nodes, skeleton_edges = skeletonize_point_cloud_from_df(df, distance_threshold=2.0)
#sorted_skeleton = sort_skeleton_points(skeleton_nodes, skeleton_edges)


def skeletonize_point_cloud_from_dfV2(df, distance_threshold=2.0, min_degree=1):
    """
    Skeletonize 3D point cloud from a dataframe with columns ['X','Y','Z','Q'].
    Only connects points whose pairwise distance <= distance_threshold.
    """
    coords = df[['X','Y','Z']].to_numpy()
    Q = df['Q'].to_numpy() if 'Q' in df.columns else None
    
    N = len(coords)
    G = nx.Graph()

    # Add nodes
    for i in range(N):
        G.add_node(i, pos=coords[i], Q=Q[i] if Q is not None else 1.0)

    # Build KD-tree
    tree = cKDTree(coords)

    # Build graph with hard distance cutoff
    for i, pt in enumerate(coords):
        neighbors = tree.query_ball_point(pt, r=distance_threshold)
        for j in neighbors:
            if i >= j:
                continue
            dist = np.linalg.norm(pt - coords[j])
            if dist <= distance_threshold:           # <-- strict local threshold
                G.add_edge(i, j, weight=dist)        # attribute name: 'weight'

    # Extremities
    extremities = [n for n in G.nodes if G.degree[n] <= min_degree]
    if len(extremities) < 2:
        extremities = list(G.nodes)

    skeleton_edges = set()

    # Shortest paths between extremities using correct weight
    for i, src in enumerate(extremities):
        for dst in extremities[i+1:]:
            max_endpoint_distance = 3.0  # or whatever
            if np.linalg.norm(coords[src] - coords[dst]) > max_endpoint_distance:
                continue
            try:
                path = nx.shortest_path(G, source=src, target=dst, weight='weight')
                # add all edges of the path, they are already <= distance_threshold
                for k in range(len(path) - 1):
                    u, v = path[k], path[k+1]
                    # optional: safety check (should always pass)
                    if G[u][v]['weight'] <= distance_threshold:
                        skeleton_edges.add((u, v))
            except nx.NetworkXNoPath:
                continue

    # Build skeleton graph
    G_skel = nx.Graph()
    G_skel.add_nodes_from(range(N))
    G_skel.add_edges_from(skeleton_edges)

    # Optional: prune leaves iteratively
    #while True:
    #    leaves = [n for n in G_skel.nodes if G_skel.degree[n] == 1]
    #    if not leaves:
    #        break
    #    G_skel.remove_nodes_from(leaves)

    skeleton_nodes = coords[list(G_skel.nodes)]
    skeleton_edges = list(G_skel.edges)

    return skeleton_nodes, skeleton_edges


def order_points_longest_greedy(points):
    """
    Sort points by:
      1) finding the pair of points with maximal Euclidean distance
      2) starting from one endpoint
      3) repeatedly choosing the nearest unused point

    Returns:
        ordered_points: array of shape (N, 3)
        order_indices : list of indices in the order visited
    """
    pts = np.asarray(points)
    N = len(pts)

    # ---- 1. Find the farthest pair (A, B) ----
    D = np.linalg.norm(pts[:,None,:] - pts[None,:,:], axis=2)
    A, B = np.unravel_index(np.argmax(D), D.shape)

    # ---- 2. Greedy nearest-neighbor walk from A ----
    unused = set(range(N))
    unused.remove(A)

    order = [A]
    current = A

    while unused:
        # find nearest unused point
        next_idx = min(unused, key=lambda j: np.linalg.norm(pts[current] - pts[j]))
        order.append(next_idx)
        unused.remove(next_idx)
        current = next_idx

    # ---- 3. Ensure the path ends at the farthest point B ----
    if order[-1] != B:
        order.reverse()

    return pts[order]


from skimage.morphology import skeletonize
def skeleton_voxel_coordinates(hist_bins, edges_x, edges_y, edges_z):
    """
    hist_bins: boolean 3D histogram (Nx, Ny, Nz) or any shape
    edges_x, edges_y, edges_z: histogram bin edges
    """

    # 1. Reorder for skimage (Z, Y, X)
    volume = hist_bins.transpose(2, 1, 0)

    # 2. Skeleton
    skel = skeletonize(volume, method='lee')

    # 3. Get voxel indices
    z_idx, y_idx, x_idx = np.where(skel)

    # 4. Compute bin centers
    xs = 0.5 * (edges_x[:-1] + edges_x[1:])
    ys = 0.5 * (edges_y[:-1] + edges_y[1:])
    zs = 0.5 * (edges_z[:-1] + edges_z[1:])

    # 5. Convert voxel indices → real coordinates
    X = xs[x_idx]
    Y = ys[y_idx]
    Z = zs[z_idx]

    # Stack
    coords = np.vstack([X, Y, Z]).T

    return coords

def skeleton_voxel_coordinates2D(hist_bins, edges_x, edges_y):
    """
    hist_bins: boolean 2D histogram (Nx, Ny)
    edges_x, edges_y: histogram bin edges
    """

    # 1. Reorder for skimage (Y, X)
    volume = hist_bins.transpose(1, 0)

    # 2. Skeleton
    skel = skeletonize(volume, method='lee')

    # 3. Get voxel indices
    y_idx, x_idx = np.where(skel)

    # 4. Compute bin centers
    xs = 0.5 * (edges_x[:-1] + edges_x[1:])
    ys = 0.5 * (edges_y[:-1] + edges_y[1:])

    # 5. Convert voxel indices → real coordinates
    X = xs[x_idx]
    Y = ys[y_idx]

    # Stack
    coords = np.vstack([X, Y]).T

    return coords