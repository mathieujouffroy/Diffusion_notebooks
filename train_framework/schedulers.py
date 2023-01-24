import torch
import torch.nn.functional as F

def cosine_beta_schedule(timesteps, s=0.008):
    """Cosine schedule as proposed in https://arxiv.org/abs/2102.09672."""
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


def linear_beta_schedule(timesteps):
    """Linear schedule for the noise scheduler."""
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)


def quadratic_beta_schedule(timesteps):
    """Quadratic schedule for the noise scheduler."""
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start**0.5, beta_end**0.5, timesteps) ** 2


def sigmoid_beta_schedule(timesteps):
    """Sigmoid schedule for the noise scheduler."""
    beta_start = 0.0001
    beta_end = 0.02
    betas = torch.linspace(-6, 6, timesteps)
    return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start


def get_components_list(betas):
    """Retrieve the list of components for the noise scheduler."""
    # define alphas 
    alphas = 1. - betas # α
    alphas_cumprod = torch.cumprod(alphas, axis=0) # α_bar
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0) # α_bar(t-1)
    sqrt_recip_alphas = torch.sqrt(1.0 / alphas) # √(1/α)

    # calculations for diffusion q(x_t | x_{t-1}) and others
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod) # √(α_bar)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod) # √(1 - α_bar)

    # calculations for posterior q(x_{t-1} | x_t, x_0)
    posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
    component_list = [sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, sqrt_recip_alphas, posterior_variance]
    return component_list
