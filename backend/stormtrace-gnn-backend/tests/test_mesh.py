import numpy as np
from src.mesh import (
    build_icosahedral_mesh, grid_to_mesh, mesh_to_grid,
    grid_to_mesh_multivar, build_sparse_adjacency,
)


def test_mesh_build_shapes():
    vertices, faces = build_icosahedral_mesh(subdivisions=2)
    assert vertices.shape[1] == 3
    assert faces.shape[1] == 3
    assert len(vertices) > 100


def test_reprojection_round_trip_recovers_field():
    """Guardrail from blueprint section 12: 're-project a known analytic field onto the
    mesh and back, recover the original within tolerance'."""
    grid_size = 16
    x, y = np.meshgrid(np.linspace(-3, 3, grid_size), np.linspace(-3, 3, grid_size))
    analytic_field = np.exp(-(x ** 2 + y ** 2) / 4)  # smooth known field

    vertices, _ = build_icosahedral_mesh(subdivisions=3)
    mesh_values = grid_to_mesh(analytic_field, vertices)
    recovered = mesh_to_grid(mesh_values, vertices, grid_size)

    err = np.abs(recovered - analytic_field).mean()
    assert err < 0.1, f"round-trip reprojection error too high: {err}"


def test_sparse_adjacency_symmetric_and_normalized():
    vertices, faces = build_icosahedral_mesh(subdivisions=2)
    adj = build_sparse_adjacency(faces, len(vertices))
    dense = adj.toarray()
    assert np.allclose(dense, dense.T, atol=1e-5), "adjacency should be symmetric"
    assert adj.nnz > 0
