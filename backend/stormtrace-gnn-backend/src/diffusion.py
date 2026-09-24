"""
Conditional Denoising Diffusion Probabilistic Model (DDPM) for amplitude-
preserving 12km -> 5km downscaling. This is the direct fix for the spectral
smoothing the U-Net baseline demonstrates: instead of regressing to a single
mean field, the model learns to denoise samples from the true conditional
distribution, so a sampled extreme stays extreme.

Upgrades over a minimal single-shot conv net:
  - a proper DDPM formulation: fixed linear beta schedule, closed-form
    q_sample (forward noising at any timestep in one shot, not a step loop),
    sinusoidal timestep embeddings (not a constant channel), and ancestral
    (not just noise-subtraction) sampling with the correct posterior variance
  - the denoiser is conditioned on the coarse-resolution field (upsampled)
    as an extra channel, so it is a genuine super-resolution/downscaling
    model, not an unconditional generator
  - physics-informed loss: alongside the standard noise-prediction MSE, a
    moisture-convergence penalty (see physics.py) is added with weight
    settings.physics_loss_weight, so the model is trained -- not just
    evaluated -- against the conservation-law check
"""
import math
import torch
import torch.nn as nn
import numpy as np

from config import settings
from src.physics import moisture_convergence_penalty


def sinusoidal_embedding(t: torch.Tensor, dim: int = 16):
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, dtype=torch.float32) / half)
    args = t[:, None].float() * freqs[None, :]
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class ConditionalDenoiser(nn.Module):
    """Input channels: [noisy_target, coarse_condition] + broadcast timestep embedding."""

    def __init__(self, t_emb_dim: int = 16, hidden: int = 32):
        super().__init__()
        self.t_emb_dim = t_emb_dim
        self.t_mlp = nn.Sequential(nn.Linear(t_emb_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden))

        self.in_conv = nn.Conv2d(2, hidden, 3, padding=1)
        self.block1 = nn.Sequential(nn.Conv2d(hidden, hidden, 3, padding=1), nn.SiLU())
        self.block2 = nn.Sequential(nn.Conv2d(hidden, hidden, 3, padding=1), nn.SiLU())
        self.out_conv = nn.Conv2d(hidden, 1, 3, padding=1)

    def forward(self, noisy, condition, t):
        x = torch.cat([noisy, condition], dim=1)
        h = self.in_conv(x)
        t_emb = self.t_mlp(sinusoidal_embedding(t, self.t_emb_dim))  # (B, hidden)
        h = h + t_emb[:, :, None, None]
        h = self.block1(h)
        h = self.block2(h)
        return self.out_conv(h)


class DDPMSchedule:
    def __init__(self, n_steps: int, beta_start=1e-4, beta_end=2e-2):
        self.n_steps = n_steps
        self.betas = torch.linspace(beta_start, beta_end, n_steps)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def q_sample(self, x0, t, noise):
        """Closed-form forward diffusion to step t (vectorized, no step loop)."""
        ab = self.alpha_bars[t].view(-1, 1, 1, 1)
        return torch.sqrt(ab) * x0 + torch.sqrt(1 - ab) * noise


def train_diffusion(coarse_condition: torch.Tensor, target_high_res: torch.Tensor, epochs: int = None):
    """
    coarse_condition: (1,1,H,W) already upsampled (bilinear) 12km->5km grid, the conditioning signal
    target_high_res:  (1,1,H,W) the "true" high-amplitude field to learn to generate
    """
    epochs = epochs or settings.diffusion_epochs
    n_steps = settings.diffusion_steps
    schedule = DDPMSchedule(n_steps)
    model = ConditionalDenoiser()
    optimizer = torch.optim.Adam(model.parameters(), lr=settings.diffusion_lr)

    for epoch in range(epochs):
        t = torch.randint(0, n_steps, (1,))
        noise = torch.randn_like(target_high_res)
        noisy = schedule.q_sample(target_high_res, t, noise)

        optimizer.zero_grad()
        predicted_noise = model(noisy, coarse_condition, t)
        noise_loss = nn.functional.mse_loss(predicted_noise, noise)

        # physics-informed term: reconstruct an approximate x0 estimate from the
        # current noise prediction and penalize conservation-law violation in it,
        # so the training signal -- not just post-hoc evaluation -- includes physics
        with torch.no_grad():
            ab = schedule.alpha_bars[t].view(-1, 1, 1, 1)
        x0_estimate = (noisy - torch.sqrt(1 - ab) * predicted_noise) / torch.sqrt(ab).clamp(min=1e-4)
        physics_penalty = moisture_convergence_penalty(x0_estimate, coarse_condition)

        loss = noise_loss + settings.physics_loss_weight * physics_penalty
        loss.backward()
        optimizer.step()

    return model, schedule


@torch.no_grad()
def sample_diffusion(model: ConditionalDenoiser, schedule: DDPMSchedule, coarse_condition: torch.Tensor):
    """Ancestral DDPM sampling: start from pure noise, iteratively denoise using the
    correct posterior mean/variance (not a naive linear noise subtraction)."""
    shape = coarse_condition.shape
    x = torch.randn(shape)

    for step in reversed(range(schedule.n_steps)):
        t = torch.tensor([step])
        beta_t = schedule.betas[step]
        alpha_t = schedule.alphas[step]
        alpha_bar_t = schedule.alpha_bars[step]

        predicted_noise = model(x, coarse_condition, t)
        mean = (1 / torch.sqrt(alpha_t)) * (x - (beta_t / torch.sqrt(1 - alpha_bar_t)) * predicted_noise)

        if step > 0:
            noise = torch.randn(shape)
            variance = beta_t
            x = mean + torch.sqrt(variance) * noise
        else:
            x = mean

    return x


def train_and_generate(ensemble_field: np.ndarray, variable_index: int = 0):
    """
    ensemble_field: (n_members, n_vars, grid, grid) for a single forecast timestep.
    Uses the ensemble mean of `variable_index` as the coarse condition and the
    highest-amplitude member as the training target (the one closest to the true
    extreme the diffusion model should learn to preserve).

    Fields are z-score normalized before entering the diffusion process and
    de-normalized after sampling. DDPM's noise schedule assumes roughly unit-
    scale data (q_sample blends x0 with N(0,1) noise); real meteorological
    fields sit at an offset (e.g. ~25-40 degC), so without this the reverse
    process converges toward the noise prior's own scale rather than the
    field's true amplitude -- which would silently sabotage the exact
    peak-retention comparison against the U-Net baseline that is this
    project's core claim.
    """
    var_field = ensemble_field[:, variable_index]  # (n_members, grid, grid)
    extreme_member = var_field[np.abs(var_field - var_field.mean(axis=0)).sum(axis=(1, 2)).argmax()]
    target = torch.tensor(extreme_member, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    coarse = torch.tensor(var_field.mean(axis=0), dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    from src.unet import upscale_bilinear
    coarse_upsampled = upscale_bilinear(coarse)
    target_upsampled = upscale_bilinear(target)  # our synthetic grid has no separate high-res truth,
    # so both baselines are trained/evaluated at the same upsampled resolution for a fair comparison

    mean = target_upsampled.mean()
    std = target_upsampled.std().clamp(min=1e-3)
    coarse_norm = (coarse_upsampled - mean) / std
    target_norm = (target_upsampled - mean) / std

    model, schedule = train_diffusion(coarse_norm, target_norm)
    generated_norm = sample_diffusion(model, schedule, coarse_norm)
    generated = generated_norm * std + mean

    return generated, coarse_upsampled, target_upsampled
