"""
Pinpoint centroid + severity tier, per blueprint section 3 final stage.

Upgrades over a peak-value-only heuristic: severity is a calibrated blend of
  - normalized peak amplitude of the generated 5km field
  - the EFI at that location (how extreme vs 30yr baseline)
  - the physics consistency score (down-weights alerts physics disagrees with)
so a high peak value that fails the physics check does not get over-alerted
as "severe" -- it is capped down, with the cap itself reported as `confidence`.
"""
import numpy as np

from config import settings


def compute_alert(
    generated_field: np.ndarray,
    efi_field_at_mesh_peak: float,
    physics_score: float,
) -> dict:
    field = np.asarray(generated_field).squeeze()
    peak_idx = np.unravel_index(field.argmax(), field.shape)
    peak_val = float(field.max())
    normalized_amp = float((peak_val - field.min()) / (field.max() - field.min() + 1e-6))

    efi_norm = float(np.clip((efi_field_at_mesh_peak + 1) / 2, 0, 1))  # EFI in [-1,1] -> [0,1]

    # weighted blend: amplitude matters most, EFI corroborates, physics gates confidence
    raw_score = 0.55 * normalized_amp + 0.35 * efi_norm + 0.10 * physics_score
    confidence = float(np.clip(0.5 * raw_score + 0.5 * physics_score, 0, 1))

    if raw_score < settings.severity_low_thresh:
        severity = "low"
    elif raw_score < settings.severity_high_thresh:
        severity = "moderate"
    else:
        severity = "severe"

    # a severe call with low physics confidence is down-graded one tier, so the
    # alert API never emits "severe" purely off an unsupported generative peak
    if severity == "severe" and physics_score < 0.35:
        severity = "moderate"

    return {
        "centroid_row": int(peak_idx[0]),
        "centroid_col": int(peak_idx[1]),
        "peak_value": peak_val,
        "severity": severity,
        "confidence": round(confidence, 3),
        "radius_km": 5.0,
        "raw_score": round(float(raw_score), 3),
    }
