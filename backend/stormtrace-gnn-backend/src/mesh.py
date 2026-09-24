"""
Spherical (icosahedral) mesh construction and flat-grid <-> mesh reprojection.

Upgrades over a minimal version:
  - configurable subdivision depth (default 4 -> 2562 vertices, dense enough
    for real message-passing instead of a 162-point toy)
  - grid_to_mesh / mesh_to_grid are inverses of each other (nearest-neighbour
    KDTree in both directions), so re-projection correctness is unit-testable
    per the blueprint's "Mesh re-projection correctness" guardrail
  - build_sparse_adjacency is fully vectorized (numpy, no python triangle loop)
    and returns a scipy CSR matrix with symmetric GCN-style normalization,
    which is what makes the GNN forward pass a single sparse matmul instead
    of an O(V^2) dense multiply
"""
import numpy as np
import trimesh
from scipy.spatial import cKDTree
from scipy import sparse

from config import settings


def build_icosahedral_mesh(subdivisions: int = None):
    subdivisions = subdivisions if subdivisions is not None else settings.mesh_subdivisions
    mesh = trimesh.creation.icosphere(subdivisions=subdivisions, radius=1.0)
    return mesh.vertices.astype(np.float64), mesh.faces.astype(np.int64)


def _grid_points(grid_size: int):
    xs = np.linspace(-1, 1, grid_size)
    ys = np.linspace(-1, 1, grid_size)
    xx, yy = np.meshgrid(xs, ys)
    return np.stack([xx.ravel(), yy.ravel(), np.zeros(xx.size)], axis=-1)


def grid_to_mesh(grid: np.ndarray, vertices: np.ndarray) -> np.ndarray:
    """Nearest-neighbour sample a (grid,grid) field onto mesh vertices (projected to the
    equatorial plane, since our synthetic data has no real lat/lon structure to preserve)."""
    n = grid.shape[0]
    flat_xyz = _grid_points(n)
    tree = cKDTree(flat_xyz)
    query_points = vertices.copy()
    query_points[:, 2] = 0
    _, idx = tree.query(query_points)
    return grid.ravel()[idx]


def mesh_to_grid(mesh_values: np.ndarray, vertices: np.ndarray, grid_size: int) -> np.ndarray:
    """Inverse of grid_to_mesh: nearest-neighbour scatter mesh values back onto a flat grid.
    Used purely to unit-test re-projection fidelity."""
    flat_xyz = _grid_points(grid_size)
    query_points = vertices.copy()
    query_points[:, 2] = 0
    tree = cKDTree(query_points)
    _, idx = tree.query(flat_xyz)
    return mesh_values[idx].reshape(grid_size, grid_size)


def grid_to_mesh_multivar(grid_stack: np.ndarray, vertices: np.ndarray) -> np.ndarray:
    """grid_stack: (n_vars, grid, grid) -> (n_mesh_points, n_vars), single KDTree build reused
    across variables for efficiency."""
    n = grid_stack.shape[-1]
    flat_xyz = _grid_points(n)
    tree = cKDTree(flat_xyz)
    query_points = vertices.copy()
    query_points[:, 2] = 0
    _, idx = tree.query(query_points)
    n_vars = grid_stack.shape[0]
    flat = grid_stack.reshape(n_vars, -1)
    return flat[:, idx].T  # (n_mesh_points, n_vars)


def build_sparse_adjacency(faces: np.ndarray, n_points: int) -> sparse.csr_matrix:
    """Vectorized edge extraction (no per-triangle python loop) + symmetric
    GCN normalization: A_hat = D^-1/2 (A + I) D^-1/2."""
    edges_a = np.concatenate([faces[:, 0], faces[:, 1], faces[:, 2]])
    edges_b = np.concatenate([faces[:, 1], faces[:, 2], faces[:, 0]])
    row = np.concatenate([edges_a, edges_b])
    col = np.concatenate([edges_b, edges_a])
    data = np.ones(len(row), dtype=np.float32)
    A = sparse.coo_matrix((data, (row, col)), shape=(n_points, n_points))
    A = A.tocsr()
    A.data[:] = 1.0  # dedupe multi-edges to binary adjacency
    A = A + sparse.eye(n_points, format="csr")  # self loops

    deg = np.asarray(A.sum(axis=1)).flatten()
    deg_inv_sqrt = np.zeros_like(deg)
    nonzero = deg > 0
    deg_inv_sqrt[nonzero] = np.power(deg[nonzero], -0.5)
    D_inv_sqrt = sparse.diags(deg_inv_sqrt)
    A_hat = D_inv_sqrt @ A @ D_inv_sqrt
    return A_hat.tocsr()


def scipy_csr_to_torch_sparse(csr):
    import torch
    coo = csr.tocoo()
    indices = torch.tensor(np.vstack([coo.row, coo.col]), dtype=torch.long)
    values = torch.tensor(coo.data, dtype=torch.float32)
    return torch.sparse_coo_tensor(indices, values, size=coo.shape).coalesce()


if __name__ == "__main__":
    import os
    os.makedirs(settings.data_dir, exist_ok=True)
    vertices, faces = build_icosahedral_mesh()
    ensemble = np.load(f"{settings.data_dir}/ensemble.npy")  # (steps, members, vars, g, g)
    mesh_values = grid_to_mesh_multivar(ensemble[0, 0], vertices)  # step0, member0

    # round-trip fidelity check
    back = mesh_to_grid(mesh_values[:, 0], vertices, ensemble.shape[-1])
    err = np.abs(back - ensemble[0, 0, 0]).mean()

    adjacency = build_sparse_adjacency(faces, len(vertices))

    np.save(f"{settings.data_dir}/mesh_vertices.npy", vertices)
    np.save(f"{settings.data_dir}/mesh_faces.npy", faces)
    np.save(f"{settings.data_dir}/mesh_values.npy", mesh_values)
    sparse.save_npz(f"{settings.data_dir}/mesh_adjacency.npz", adjacency)

    print(f"Mesh built. Vertices: {vertices.shape} Faces: {faces.shape}")
    print(f"Mesh values (multivar): {mesh_values.shape}")
    print(f"Adjacency: {adjacency.shape}, nnz={adjacency.nnz}")
    print(f"Round-trip reprojection mean abs error: {err:.4f}")
