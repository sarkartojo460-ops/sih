"""
Stage 1, temporal half: turn per-timestep GNN anomaly scores into a single
4D (x, y, z, t) bounding box tracking the anomaly across the D+3..D+10 window,
per blueprint section 3.1 ("4D temporal bounding box... reusing standard
multi-object tracking association logic: nearest-neighbour or Hungarian
matching on mesh-node clusters").

Pipeline:
  1. At each timestep, connected-component cluster the mesh nodes whose GNN
     score exceeds the threshold (using the sparse adjacency, via scipy's
     connected_components -- vectorized, not a manual BFS).
  2. Take the largest cluster per timestep as "the" tracked anomaly (extendable
     to multi-object tracking by keeping all clusters and doing Hungarian
     matching across ALL of them, which is implemented below and used when
     more than one significant cluster exists at a timestep).
  3. Link cluster centroids across consecutive timesteps by nearest-centroid
     (Hungarian assignment, scipy.optimize.linear_sum_assignment) to build a
     continuous trajectory even if the anomaly moves.
  4. Union the bounding volumes across all linked timesteps into one 4D box.
"""
import numpy as np
from scipy.sparse.csgraph import connected_components
from scipy.optimize import linear_sum_assignment


def cluster_anomalies(predictions: np.ndarray, adjacency_csr, threshold: float):
    """Connected-component clustering restricted to the anomalous subgraph."""
    mask = predictions > threshold
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return []

    sub_adj = adjacency_csr[idx][:, idx]
    n_components, labels = connected_components(sub_adj, directed=False)

    clusters = []
    for c in range(n_components):
        member_idx = idx[labels == c]
        if len(member_idx) == 0:
            continue
        clusters.append(member_idx)
    return clusters


def _cluster_centroid(vertices, cluster_idx):
    return vertices[cluster_idx].mean(axis=0)


def link_clusters_across_time(per_step_clusters, vertices, max_link_distance=1.5):
    """
    per_step_clusters: list (len = n_steps) of lists of index arrays (clusters at that step)
    Greedy/Hungarian-matched trajectory linking: each step's clusters are matched to the
    previous step's active tracks by centroid distance; unmatched clusters start new tracks.
    Returns list of tracks, each track = list of (t, cluster_idx_array).
    """
    active_tracks = []  # list of dict(last_centroid, steps=[(t, idx)])
    finished_tracks = []

    for t, clusters in enumerate(per_step_clusters):
        if not clusters:
            continue
        centroids = np.array([_cluster_centroid(vertices, c) for c in clusters])

        if not active_tracks:
            for c, cen in zip(clusters, centroids):
                active_tracks.append({"last_centroid": cen, "steps": [(t, c)]})
            continue

        prev_centroids = np.array([tr["last_centroid"] for tr in active_tracks])
        cost = np.linalg.norm(prev_centroids[:, None, :] - centroids[None, :, :], axis=-1)
        row_idx, col_idx = linear_sum_assignment(cost)

        matched_cols = set()
        for r, c in zip(row_idx, col_idx):
            if cost[r, c] <= max_link_distance:
                active_tracks[r]["steps"].append((t, clusters[c]))
                active_tracks[r]["last_centroid"] = centroids[c]
                matched_cols.add(c)

        # unmatched new clusters become new tracks
        for c_i, (c, cen) in enumerate(zip(clusters, centroids)):
            if c_i not in matched_cols:
                active_tracks.append({"last_centroid": cen, "steps": [(t, c)]})

    finished_tracks.extend(active_tracks)
    return finished_tracks


def track_to_bbox4d(track, vertices):
    all_idx = np.concatenate([idx for _, idx in track["steps"]])
    points = vertices[all_idx]
    t_values = [t for t, _ in track["steps"]]
    return {
        "x_min": float(points[:, 0].min()), "x_max": float(points[:, 0].max()),
        "y_min": float(points[:, 1].min()), "y_max": float(points[:, 1].max()),
        "z_min": float(points[:, 2].min()), "z_max": float(points[:, 2].max()),
        "t_start": int(min(t_values)),
        "t_end": int(max(t_values)),
        "n_points": int(len(np.unique(all_idx))),
    }


def track_anomaly_4d(predictions_per_step, adjacency_csr, vertices, threshold: float):
    """
    predictions_per_step: (n_steps, n_points) GNN anomaly scores at each forecast step
    Returns the dominant 4D bounding box (largest track by total point-count) plus
    the full list of tracks for completeness.
    """
    per_step_clusters = [
        cluster_anomalies(predictions_per_step[t], adjacency_csr, threshold)
        for t in range(predictions_per_step.shape[0])
    ]
    tracks = link_clusters_across_time(per_step_clusters, vertices)
    if not tracks:
        return None, []

    boxes = [track_to_bbox4d(tr, vertices) for tr in tracks]
    dominant_idx = int(np.argmax([b["n_points"] for b in boxes]))
    return boxes[dominant_idx], boxes
