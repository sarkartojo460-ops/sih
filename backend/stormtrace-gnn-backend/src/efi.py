"""
Extreme Forecast Index (EFI): how unusual is the forecast at each mesh point
compared to that SAME point's 30-year climatological distribution.

Fix vs a naive version: EFI must be computed per-point against that point's
own baseline distribution, not against a single grid-wide scalar (averaging
away the baseline's spatial distribution would silently break the metric,
since a hotspot could easily look "normal" against a flattened baseline).

Vectorized: no python loop over mesh points. For P points and Y baseline
years this is an (P, Y) broadcasted comparison done in one numpy call.
"""
import numpy as np

from src.mesh import grid_to_mesh_multivar


def compute_efi_field(forecast_mesh_values: np.ndarray, baseline_mesh_values: np.ndarray) -> np.ndarray:
    """
    forecast_mesh_values: (n_points,) forecast at each mesh point, single variable
    baseline_mesh_values: (n_years, n_points) baseline at each mesh point, same variable
    returns: (n_points,) EFI in [-1, 1], close to 1 = unusually extreme high
    """
    # percentile[i] = fraction of baseline years at point i that are <= forecast value at point i
    percentile = (baseline_mesh_values <= forecast_mesh_values[None, :]).mean(axis=0)
    return 2 * percentile - 1


def compute_efi_multivar(forecast_mesh: np.ndarray, baseline_years_grid: np.ndarray, vertices: np.ndarray) -> dict:
    """
    forecast_mesh: (n_points, n_vars) forecast field already projected to the mesh
    baseline_years_grid: (n_years, n_vars, grid, grid) raw baseline (still on the flat grid)
    vertices: mesh vertices, used to reproject each baseline year onto the same mesh

    Returns dict variable_name -> efi array (n_points,), plus a combined "max_efi"
    field taking the element-wise max across variables (an anomaly can dominate
    via temperature OR rainfall OR wind).
    """
    from config import settings

    n_years = baseline_years_grid.shape[0]
    n_vars = baseline_years_grid.shape[1]
    n_points = vertices.shape[0]

    baseline_mesh = np.zeros((n_years, n_vars, n_points), dtype=np.float32)
    for y in range(n_years):
        baseline_mesh[y] = grid_to_mesh_multivar(baseline_years_grid[y], vertices).T

    efi_per_var = {}
    stacked = np.zeros((n_vars, n_points), dtype=np.float32)
    for v in range(n_vars):
        efi_v = compute_efi_field(forecast_mesh[:, v], baseline_mesh[:, v, :])
        name = settings.variable_names[v]
        efi_per_var[name] = efi_v
        stacked[v] = efi_v

    efi_per_var["combined_max"] = stacked.max(axis=0)
    return efi_per_var
