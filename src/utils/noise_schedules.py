"""
Noise scheduling functions for diffusion models.

This module contains various noise scheduling strategies for diffusion models,
including cosine, Karras, linear, and exponential schedules. These schedules control the
noise levels added during the forward diffusion process.
"""

import torch


def cosine_beta_schedule(timesteps, s=0.008):
    """Create cosine noise schedule for diffusion.
    
    This schedule provides a smooth, monotonic noise progression that preserves
    signal structure early in the diffusion process. Well-tested for various
    applications and provides stable training dynamics.
    
    Args:
        timesteps: Number of diffusion timesteps
        s: Offset parameter controlling the shape of the schedule (default: 0.008)
        
    Returns:
        torch.Tensor: Beta schedule for diffusion process
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


def linear_beta_schedule(timesteps, beta_start=0.0001, beta_end=0.015):
    """Create linear noise schedule for diffusion.
    
    This is the original schedule from the DDPM paper, providing a simple
    linear increase in noise levels over time. Best suited for simple,
    stationary time series or as a baseline for comparison.
    
    Args:
        timesteps: Number of diffusion timesteps
        beta_start: Starting beta value (default: 0.0001)
        beta_end: Ending beta value (default: 0.015)
        
    Returns:
        torch.Tensor: Beta schedule for diffusion process
    """
    return torch.linspace(beta_start, beta_end, timesteps)


def exponential_beta_schedule(timesteps, beta_start=0.0001, beta_end=0.02, gamma=2.0):
    """Create exponential noise schedule for diffusion.
    
    This schedule provides exponential decay in noise levels, which is particularly
    effective for time series data where early timesteps should preserve more
    signal structure and later timesteps can be more aggressive.
    
    Args:
        timesteps: Number of diffusion timesteps
        beta_start: Starting beta value (default: 0.0001)
        beta_end: Ending beta value (default: 0.02)
        gamma: Exponential decay rate (default: 2.0)
            - Higher values (2-3): More aggressive early noise reduction
            - Lower values (1-1.5): Gentler progression
            - gamma=1: Linear schedule
        
    Returns:
        torch.Tensor: Beta schedule for diffusion process
    """
    # Create exponential progression
    t = torch.linspace(0, 1, timesteps)
    
    # Exponential decay: beta = beta_start + (beta_end - beta_start) * (1 - exp(-gamma * t))
    betas = beta_start + (beta_end - beta_start) * (1 - torch.exp(-gamma * t))
    
    # Ensure monotonic increase
    betas = torch.cummax(betas, dim=0)[0]
    
    return torch.clip(betas, 0.0001, 0.9999)


def get_noise_schedule(schedule_type="cosine", timesteps=1000, **kwargs):
    """Get noise schedule based on the specified type.
    
    This is the main interface for creating noise schedules. It provides
    a unified API for all supported scheduling strategies.
    
    Args:
        schedule_type: Type of noise schedule ("cosine", "linear", "exponential")
        timesteps: Number of timesteps
        **kwargs: Additional parameters for specific schedulers
            - cosine: s (offset parameter)
            - linear: beta_start, beta_end
            - exponential: beta_start, beta_end, gamma
        
    Returns:
        dict: Dictionary containing:
            - beta_all: Beta values with zero padding
            - alpha_all: Alpha values (1 - beta)
            - alpha_bar_all: Cumulative product of alphas
            - schedule_type: Name of the schedule used
            - parameters: Parameters used for the schedule
            
    Raises:
        ValueError: If schedule_type is not recognized
    """
    if schedule_type == "cosine":
        s = kwargs.get("s", 0.008)
        betas = cosine_beta_schedule(timesteps, s=s)
    elif schedule_type == "linear":
        beta_start = kwargs.get("beta_start", 0.0001)
        beta_end = kwargs.get("beta_end", 0.02)
        betas = linear_beta_schedule(timesteps, beta_start=beta_start, beta_end=beta_end)
    elif schedule_type == "exponential":
        beta_start = kwargs.get("beta_start", 0.0001)
        beta_end = kwargs.get("beta_end", 0.02)
        gamma = kwargs.get("gamma", 2.0)
        betas = exponential_beta_schedule(timesteps, beta_start=beta_start, beta_end=beta_end, gamma=gamma)
    else:
        raise ValueError(f"Unknown schedule type: {schedule_type}. Choose from: 'cosine', 'linear', 'exponential'")
    
    # Compute derived parameters
    beta_all = torch.zeros(timesteps + 1)
    beta_all[1:] = betas
    alpha_all = 1 - beta_all
    alpha_bar_all = torch.cumprod(alpha_all, dim=0)
    alpha_bar_all[0] = 1
    
    return {
        "beta_all": beta_all,
        "alpha_all": alpha_all,
        "alpha_bar_all": alpha_bar_all,
        "schedule_type": schedule_type,
        "parameters": kwargs
    }

