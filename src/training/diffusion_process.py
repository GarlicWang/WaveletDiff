"""
Diffusion process implementations for sampling (DDPM/DDIM).

This module contains the core sampling methods used during generation
and reconstruction, separated from the main model for cleaner architecture.
"""

import torch
from abc import ABC, abstractmethod
from typing import Optional, Dict, List, Union


class DiffusionSampler(ABC):
    """Abstract base class for diffusion sampling methods."""
    
    def __init__(self, model, T=1000):
        self.model = model
        self.T = T
        self.device = next(model.parameters()).device
    
    @abstractmethod
    def sample(self, x_t_initial: torch.Tensor, 
               store_intermediates: bool = False,
               store_specific_timesteps: Optional[List[int]] = None,
               show_progress: bool = True) -> Union[torch.Tensor, Dict[int, torch.Tensor]]:
        """Generate samples using the diffusion process."""
        pass
    
    def generate(self, n_samples: int, input_dim: int, num_features: int, show_progress: bool = True, **kwargs) -> Union[torch.Tensor, Dict[int, torch.Tensor]]:
        """Generate new samples from random noise."""
        x_t_initial = torch.randn(n_samples, input_dim, num_features, device=self.device)
        return self.sample(x_t_initial, show_progress=show_progress, **kwargs)
    
    def reconstruct(self, x_0_original: torch.Tensor, show_progress: bool = True, **kwargs) -> torch.Tensor:
        """Reconstruct samples by adding noise then denoising."""
        batch_size = x_0_original.shape[0]
        
        # Add noise to original samples (forward process to timestep T)
        t_start = torch.full((batch_size,), self.T, device=self.device)
        x_t_initial, _ = self.model.compute_forward_process(x_0_original, t_start)
        
        # Run reverse process starting from noisy original samples
        result = self.sample(x_t_initial, store_intermediates=False, show_progress=show_progress, **kwargs)
        
        # Return just the final tensor if dict is returned
        if isinstance(result, dict):
            return result[0]
        return result


class DDPMSampler(DiffusionSampler):
    """Standard DDPM sampling (stochastic)."""
    
    def _ddpm_step(self, x_t: torch.Tensor, prediction: torch.Tensor, t: int, t_prev: int) -> torch.Tensor:
        """Perform a single DDPM sampling step."""
        alpha_t = self.model.alpha_all[t]
        alpha_bar_t = self.model.alpha_bar_all[t]
        alpha_bar_t_prev = self.model.alpha_bar_all[t_prev] if t_prev > 0 else torch.tensor(1.0, device=self.device)
        beta_t = self.model.beta_all[t]
        
        if self.model.prediction_target == "noise":
            # Original noise prediction approach
            coeff1 = 1 / torch.sqrt(alpha_t)
            coeff2 = beta_t / torch.sqrt(1 - alpha_bar_t)
            x_t_next = coeff1 * (x_t - coeff2 * prediction)
        elif self.model.prediction_target == "coefficient":
            # Direct coefficient prediction approach
            coeff1 = (torch.sqrt(alpha_bar_t_prev) * beta_t) / (1 - alpha_bar_t)
            coeff2 = (torch.sqrt(alpha_t) * (1 - alpha_bar_t_prev)) / (1 - alpha_bar_t)
            x_t_next = coeff1 * prediction + coeff2 * x_t
        else:
            raise ValueError(f"Unknown prediction target: {self.model.prediction_target}")
        
        # Add noise for all timesteps except the last one
        if t_prev > 0:
            noise = torch.randn_like(x_t)
            x_t_next = x_t_next + torch.sqrt(beta_t) * noise
        
        return x_t_next
    
    def sample(self, x_t_initial: torch.Tensor, 
               store_intermediates: bool = False,
               store_specific_timesteps: Optional[List[int]] = None,
               show_progress: bool = True) -> Union[torch.Tensor, Dict[int, torch.Tensor]]:
        """Sample using DDPM."""
        x_t = x_t_initial.clone()  # (batch_size, total_coeffs, num_features)
        timesteps = list(range(self.T, 0, -1))
        total_steps = len(timesteps)
        
        # Initialize storage
        stored_samples = {}
        if store_intermediates or store_specific_timesteps is not None:
            stored_samples[self.T] = x_t.clone()
        
        if show_progress:
            print(f"Starting DDPM sampling with {total_steps} steps...")
        
        # Track progress milestones
        progress_milestones = [0.2, 0.4, 0.6, 0.8, 1.0]  # 20%, 40%, 60%, 80%, 100%
        milestone_idx = 0
        
        # Perform reverse diffusion
        for i, t in enumerate(timesteps):
            # Determine next timestep
            t_prev = timesteps[i + 1] if i < len(timesteps) - 1 else 0
            
            # Get model prediction
            batch_size = x_t.shape[0]
            t_tensor = torch.full((batch_size,), t, device=self.device)
            t_norm = t_tensor.float() / self.T
            
            with torch.no_grad():
                prediction = self.model(x_t, t_norm)

            # Perform denoising step
            x_t = self._ddpm_step(x_t, prediction, t, t_prev)
            
            # Check progress and print milestones
            if show_progress and milestone_idx < len(progress_milestones):
                current_progress = (i + 1) / total_steps
                if current_progress >= progress_milestones[milestone_idx]:
                    percentage = int(progress_milestones[milestone_idx] * 100)
                    print(f"DDPM Sampling: {percentage}% complete ({i + 1}/{total_steps} steps)")
                    milestone_idx += 1
            
            # Store if needed
            should_store = False
            if store_specific_timesteps is not None and t in store_specific_timesteps:
                should_store = True
            elif store_intermediates and (t % 100 == 0 or t == timesteps[-1]):
                should_store = True
            
            if should_store:
                stored_samples[t] = x_t.clone()
        
        # Store final result
        if store_intermediates or store_specific_timesteps is not None:
            stored_samples[0] = x_t.clone()
            return stored_samples
        else:
            return x_t


class DDIMSampler(DiffusionSampler):
    """DDIM sampling (deterministic when eta=0)."""
    
    def __init__(self, model, T=1000, eta=0.0, ddim_steps=None):
        super().__init__(model, T)
        self.eta = eta
        self.ddim_steps = ddim_steps
    
    def _get_sampling_timesteps(self) -> List[int]:
        """Get the timesteps to use for DDIM sampling."""
        if self.ddim_steps is not None and self.ddim_steps < self.T:
            # Use subset of timesteps for accelerated DDIM
            step_size = self.T // self.ddim_steps
            timesteps = list(range(self.T, 0, -step_size))
            if timesteps[-1] != 1:
                timesteps.append(1)  # Ensure we end at timestep 1
            timesteps = sorted(timesteps, reverse=True)
        else:
            # Use all timesteps
            timesteps = list(range(self.T, 0, -1))
        return timesteps
    
    def _predict_x0_from_noise(self, x_t: torch.Tensor, noise_pred: torch.Tensor, alpha_bar_t: torch.Tensor) -> torch.Tensor:
        """Predict x_0 from predicted noise using the reparameterization."""
        return (x_t - torch.sqrt(1 - alpha_bar_t) * noise_pred) / torch.sqrt(alpha_bar_t)
    
    def _ddim_step(self, x_t: torch.Tensor, prediction: torch.Tensor, t: int, t_prev: int) -> torch.Tensor:
        """Perform a single DDIM sampling step."""
        alpha_bar_t = self.model.alpha_bar_all[t]
        alpha_bar_t_prev = self.model.alpha_bar_all[t_prev] if t_prev > 0 else torch.tensor(1.0, device=self.device)
        beta_t = self.model.beta_all[t]
        
        # Get x_0 prediction
        if self.model.prediction_target == "noise":
            x_0_pred = self._predict_x0_from_noise(x_t, prediction, alpha_bar_t)
            # For DDIM, we use the noise prediction in the deterministic part
            noise_pred = prediction
        elif self.model.prediction_target == "coefficient":
            x_0_pred = prediction
            # For coefficient prediction, we need to compute the noise prediction
            noise_pred = (x_t - torch.sqrt(alpha_bar_t) * x_0_pred) / torch.sqrt(1 - alpha_bar_t)
        else:
            raise ValueError(f"Unknown prediction target: {self.model.prediction_target}")
        
        # DDIM update equation
        sqrt_alpha_bar_t_prev = torch.sqrt(alpha_bar_t_prev)
        sqrt_one_minus_alpha_bar_t_prev = torch.sqrt(1 - alpha_bar_t_prev)
        
        # Deterministic part
        x_t_next = sqrt_alpha_bar_t_prev * x_0_pred + sqrt_one_minus_alpha_bar_t_prev * noise_pred
        
        # Add stochastic part if eta > 0
        if self.eta > 0.0 and t_prev > 0:
            sigma_t = self.eta * torch.sqrt((1 - alpha_bar_t_prev) / (1 - alpha_bar_t)) * torch.sqrt(beta_t)
            noise = torch.randn_like(x_t)
            x_t_next = x_t_next + sigma_t * noise
        
        return x_t_next
    
    def sample(self, x_t_initial: torch.Tensor, 
               store_intermediates: bool = False,
               store_specific_timesteps: Optional[List[int]] = None,
               show_progress: bool = True) -> Union[torch.Tensor, Dict[int, torch.Tensor]]:
        """Sample using DDIM."""
        x_t = x_t_initial.clone()  # (batch_size, total_coeffs, num_features)
        timesteps = self._get_sampling_timesteps()
        total_steps = len(timesteps)
        
        # Initialize storage
        stored_samples = {}
        if store_intermediates or store_specific_timesteps is not None:
            stored_samples[self.T] = x_t.clone()
        
        if show_progress:
            ddim_type = "accelerated" if self.ddim_steps and self.ddim_steps < self.T else "full"
            print(f"Starting DDIM sampling ({ddim_type}) with {total_steps} steps...")
        
        # Track progress milestones
        progress_milestones = [0.2, 0.4, 0.6, 0.8, 1.0]  # 20%, 40%, 60%, 80%, 100%
        milestone_idx = 0
        
        # Perform reverse diffusion
        for i, t in enumerate(timesteps):
            # Determine next timestep
            t_prev = timesteps[i + 1] if i < len(timesteps) - 1 else 0
            
            # Get model prediction
            batch_size = x_t.shape[0]
            t_tensor = torch.full((batch_size,), t, device=self.device)
            t_norm = t_tensor.float() / self.T
            
            with torch.no_grad():
                prediction = self.model(x_t, t_norm)
            
            # Perform denoising step
            x_t = self._ddim_step(x_t, prediction, t, t_prev)
            
            # Check progress and print milestones
            if show_progress and milestone_idx < len(progress_milestones):
                current_progress = (i + 1) / total_steps
                if current_progress >= progress_milestones[milestone_idx]:
                    percentage = int(progress_milestones[milestone_idx] * 100)
                    print(f"DDIM Sampling: {percentage}% complete ({i + 1}/{total_steps} steps)")
                    milestone_idx += 1
            
            # Store if needed
            should_store = False
            if store_specific_timesteps is not None and t in store_specific_timesteps:
                should_store = True
            elif store_intermediates and (t % 100 == 0 or t == timesteps[-1]):
                should_store = True
            
            if should_store:
                stored_samples[t] = x_t.clone()
        
        # Store final result
        if store_intermediates or store_specific_timesteps is not None:
            stored_samples[0] = x_t.clone()
            return stored_samples
        else:
            return x_t


class DiffusionTrainer:
    """Utility class for training and evaluation with different sampling methods."""
    
    def __init__(self, model):
        self.model = model
        self.ddpm_sampler = DDPMSampler(model, T=model.T)
        self.ddim_sampler = DDIMSampler(model, T=model.T, eta=model.ddim_eta, ddim_steps=model.ddim_steps)
    
    def generate_samples(self, n_samples: int, use_ddim: bool = False, show_progress: bool = True, **kwargs) -> torch.Tensor:
        """Generate samples using either DDPM or DDIM."""
        sampler = self.ddim_sampler if use_ddim else self.ddpm_sampler
        input_dim = self.model.input_dim
        num_features = self.model.num_features

        result = sampler.generate(n_samples, input_dim, num_features, show_progress=show_progress, **kwargs)

        # Handle both dict and tensor returns
        if isinstance(result, dict):
            # Return final samples
            return result[0]
        return result
    
    def reconstruct_samples(self, x_0_original: torch.Tensor, use_ddim: bool = True, show_progress: bool = True, **kwargs) -> torch.Tensor:
        """Reconstruct samples using either DDPM or DDIM."""
        sampler = self.ddim_sampler if use_ddim else self.ddpm_sampler
        return sampler.reconstruct(x_0_original, show_progress=show_progress, **kwargs)
