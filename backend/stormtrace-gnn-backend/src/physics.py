"""
Physics-informed constraints (blueprint section 3.3 / 8): the model is
penalized, not hard-constrained, if it generates a physically implausible
field -- e.g. a sharp rainfall/temperature spike with no matching wind
convergence feeding it.

Two checks are implemented:
  1. moisture_convergence_penalty  -- a differentiable torch penalty used
     INSIDE the diffusion training loss (diffusion.py imports this).
  2. physics_consistency_score     -- a post-hoc numpy score (0..1, higher
     is better) used for the physics-check panel / API report, comparing the
     generated field's peak location against the ensemble's own agreement
     (low spread = ensemble agreed something unusual was there = trustworthy).

A field with a sharp local peak that has no support in the negative Laplacian
(i.e. no local convergence pattern) is penalized -- this is a lightweight
stand-in for real moisture-convergence-from-wind-divergence physics, as the
blueprint itself notes should start as ONE conservation check before adding
the full fluid-dynamics set.
"""
import numpy as np
import torch
import torch.nn.functional as F


_LAPLACIAN_KERNEL = torch.tensor(
    [[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]]
).view(1, 1, 3, 3)


def moisture_convergence_penalty(generated_field: torch.Tensor, coarse_condition: torch.Tensor) -> torch.Tensor:
    """
    generated_field, coarse_condition: (B,1,H,W)
    Penalizes locations where the generated field has a strong positive peak
    (rain/heat spike) but the coarse-scale condition shows no local convergence
    (negative Laplacian) supporting it -- i.e. the model "invented" an extreme.
    """
    kernel = _LAPLACIAN_KERNEL.to(generated_field.dtype)
    lap_condition = F.conv2d(coarse_condition, kernel, padding=1)
    convergence = torch.relu(-lap_condition)  # positive where mass/heat is converging
    convergence_norm = convergence / (convergence.amax(dim=(-1, -2), keepdim=True) + 1e-6)

    peak_norm = generated_field / (generated_field.amax(dim=(-1, -2), keepdim=True).abs() + 1e-6)
    unsupported_peak = torch.relu(peak_norm - convergence_norm)
    return unsupported_peak.mean()


def physics_consistency_score(generated_field: np.ndarray, ensemble_var_field: np.ndarray) -> float:
    """
    generated_field: (1,1,H,W) or (H,W) diffusion output
    ensemble_var_field: (n_members, grid, grid) raw ensemble for the same variable
    Returns 0..1, higher = more physically consistent with ensemble agreement at the peak.
    """
    from scipy.ndimage import zoom

    generated = np.asarray(generated_field).squeeze()
    ensemble_spread = ensemble_var_field.std(axis=0)

    scale = generated.shape[0] / ensemble_spread.shape[0]
    spread_resized = zoom(ensemble_spread, scale)

    peak_idx = np.unravel_index(generated.argmax(), generated.shape)
    agreement_at_peak = 1 - min(spread_resized[peak_idx] / (spread_resized.max() + 1e-6), 1.0)
    return float(np.clip(agreement_at_peak, 0.0, 1.0))


def moisture_convergence_score_numpy(generated_field: np.ndarray, coarse_condition: np.ndarray) -> float:
    """Numpy mirror of the torch training penalty, for the post-hoc physics-check panel."""
    with torch.no_grad():
        g = torch.tensor(np.asarray(generated_field), dtype=torch.float32).reshape(1, 1, *generated_field.shape[-2:])
        c = torch.tensor(np.asarray(coarse_condition), dtype=torch.float32).reshape(1, 1, *coarse_condition.shape[-2:])
        penalty = moisture_convergence_penalty(g, c).item()
    return float(np.clip(1.0 - penalty, 0.0, 1.0))  # convert penalty -> score, higher is better
