import numpy as np
import torch
from src.physics import moisture_convergence_penalty, physics_consistency_score


def test_unsupported_peak_penalized_more_than_supported_peak():
    """Guardrail from blueprint section 13: a synthetic field with rainfall but no
    moisture convergence is penalized more than a physically consistent field."""
    # a coarse condition field with a real local convergence pocket (negative Laplacian)
    # centered at the same location as the generated peak
    yy, xx = torch.meshgrid(torch.arange(16), torch.arange(16), indexing="ij")
    convergent_condition = torch.exp(-(((xx - 8) ** 2 + (yy - 8) ** 2) / 6.0)).unsqueeze(0).unsqueeze(0)
    flat_condition = torch.zeros(1, 1, 16, 16)  # no convergence anywhere

    spike = torch.zeros(1, 1, 16, 16)
    spike[0, 0, 8, 8] = 10.0  # sharp peak at the same location for both cases

    penalty_unsupported = moisture_convergence_penalty(spike, flat_condition).item()
    penalty_supported = moisture_convergence_penalty(spike, convergent_condition).item()

    assert penalty_unsupported > penalty_supported


def test_physics_consistency_score_bounded():
    generated = np.random.rand(16, 16)
    ensemble = np.random.rand(5, 8, 8)
    score = physics_consistency_score(generated, ensemble)
    assert 0.0 <= score <= 1.0
