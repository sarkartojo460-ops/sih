import numpy as np
from src.efi import compute_efi_field


def test_efi_extreme_value_near_one():
    baseline = np.full((30, 5), 25.0)  # 30 years, 5 points, all normal = 25
    forecast = np.array([25.0, 25.0, 25.0, 25.0, 60.0])  # last point wildly extreme
    efi = compute_efi_field(forecast, baseline)
    assert efi[-1] > 0.9, "an extreme value far above the whole baseline should score near +1"


def test_efi_normal_value_near_zero():
    rng = np.random.default_rng(0)
    baseline = rng.normal(25, 2, (1000, 3))
    forecast = np.array([25.0, 25.0, 25.0])  # right at the baseline mean
    efi = compute_efi_field(forecast, baseline)
    assert np.all(np.abs(efi) < 0.2), "a forecast at the baseline mean should score near 0"
