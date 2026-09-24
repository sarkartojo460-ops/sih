"""
Central configuration for STORMTRACE-GNN backend.
All tunables are env-overridable so the same code runs identically
in dev (toy grid) and a heavier deployment (larger grid / mesh / ensemble)
with zero code changes -- only these numbers move.
"""
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    # --- storage ---
    database_url: str = Field(
        default="postgresql://stormtrace:stormtrace@localhost:5432/stormtrace",
        alias="DATABASE_URL",
    )
    data_dir: str = "data"

    # --- synthetic ensemble geometry (stand-in for NEPS-G 12km grid) ---
    grid_size: int = Field(default=32, alias="GRID_SIZE")
    n_ensemble_members: int = Field(default=8, alias="N_ENSEMBLE_MEMBERS")
    n_baseline_years: int = Field(default=30, alias="N_BASELINE_YEARS")
    # variables tracked per grid cell: t2m, rainfall_proxy, wind_u, wind_v
    n_variables: int = 4
    variable_names: tuple = ("t2m", "rainfall", "wind_u", "wind_v")

    # --- temporal window: D+3 .. D+10 -> 8 forecast steps ---
    n_forecast_steps: int = Field(default=8, alias="N_FORECAST_STEPS")

    # --- spherical mesh (icosahedral) ---
    mesh_subdivisions: int = Field(default=4, alias="MESH_SUBDIVISIONS")

    # --- GNN ---
    gnn_hidden: int = 32
    gnn_layers: int = 3
    gnn_epochs: int = 120
    gnn_lr: float = 5e-3

    # --- EFI ---
    efi_anomaly_threshold: float = 0.5  # >0.5 => flagged anomalous

    # --- downscaling ---
    upscale_factor: int = Field(default=2, alias="UPSCALE_FACTOR")
    unet_epochs: int = 60
    diffusion_steps: int = Field(default=50, alias="DIFFUSION_STEPS")
    diffusion_epochs: int = 400
    diffusion_lr: float = 2e-3
    physics_loss_weight: float = 0.15

    # --- severity tiers ---
    severity_low_thresh: float = 0.45
    severity_high_thresh: float = 0.75

    model_config = SettingsConfigDict(env_file=".env", populate_by_name=True, extra="ignore")


settings = Settings()
