"""
Wavelet-specific loss functions for diffusion models.

This module provides various loss computation strategies that handle
the imbalanced nature of wavelet coefficient representations across
different decomposition levels.
"""

import torch
import torch.nn.functional as F
from typing import List, Callable


class WaveletBalancedLoss:
    """Balanced loss computation for wavelet coefficients across multiple levels.
    
    Addresses the issue where levels with more coefficients dominate the loss
    computation in standard MSE loss, leading to imbalanced training.
    """
    
    def __init__(self, 
                 level_dims: List[int], 
                 level_start_indices: List[int], 
                 strategy: str = "coefficient_weighted",
                 approximation_weight: float = 2.0,
                 use_energy_term: bool = False,
                 energy_weight: float = 0.0):
        """
        Initialize the balanced loss function with energy term.
        
        Args:
            level_dims: List of coefficient counts for each wavelet level
            level_start_indices: Starting indices for each level in flattened array
            strategy: Loss balancing strategy
            approximation_weight: Weight multiplier for approximation level (only used for strategy "coefficient_weighted")
            energy_weight: Weight for the energy preservation term
        """
        # Validate inputs
        if len(level_dims) == 0:
            raise ValueError("level_dims cannot be empty")
        if len(level_dims) != len(level_start_indices):
            raise ValueError("level_dims and level_start_indices must have the same length")
        if any(dim <= 0 for dim in level_dims):
            raise ValueError("All level dimensions must be positive")
        
        # Validate indices are consistent
        expected_start = 0
        for i, (start_idx, dim) in enumerate(zip(level_start_indices, level_dims)):
            if start_idx != expected_start:
                raise ValueError(f"Inconsistent start index at level {i}: expected {expected_start}, got {start_idx}")
            expected_start += dim
        
        self.level_dims = level_dims
        self.level_start_indices = level_start_indices
        self.strategy = strategy
        self.num_levels = len(level_dims)
        self.approximation_weight = approximation_weight

        # Energy term parameters
        self.use_energy_term = use_energy_term
        self.energy_weight = energy_weight
        
        
        # Initialize weights based on strategy
        self.level_weights = self._initialize_weights()
        
        
        print(f"Initialized {strategy} wavelet loss:")
        for i, (dim, weight) in enumerate(zip(level_dims, self.level_weights)):
            print(f"  Level {i}: {dim} coeffs, weight={weight:.4f}")
    
    def _validate_tensor_shape(self, tensor: torch.Tensor, tensor_name: str):
        """Validate that tensor is 3D with expected shape."""
        if tensor.dim() != 3:
            raise ValueError(f"{tensor_name} must be 3D tensor [batch_size, total_coeffs_per_feature, num_features], "
                           f"got {tensor.dim()}D tensor with shape {tensor.shape}")

    def _initialize_weights(self) -> List[float]:
        """Initialize level weights based on the chosen strategy."""
        if self.strategy == "standard":
            # Standard MSE - all coefficients have equal weight
            return [1.0] * self.num_levels
        
        elif self.strategy == "level_weighted":
            # Equal weight per level (inversely proportional to level size)
            weights = [1.0 / dim for dim in self.level_dims]
            total_weight = sum(weights)
            return [w / total_weight for w in weights]
        
        elif self.strategy == "coefficient_weighted":
            # Higher weight for approximation coefficients, normalized by level size
            weights = []
            for i, dim in enumerate(self.level_dims):
                if i == 0:
                    # Approximation level gets boosted weight, normalized by dimension
                    weights.append(self.approximation_weight / dim)
                else:
                    # Detail levels get standard weight, normalized by dimension
                    weights.append(1.0 / dim)
            
            total_weight = sum(weights)
            return [w / total_weight for w in weights]
        
        
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

    def _compute_energy(self, coeffs: torch.Tensor) -> torch.Tensor:
        """
        Compute energy of wavelet coefficients using Parseval's theorem.
        
        Args:
            coeffs: Coefficients tensor [batch_size, total_coeffs_per_feature, num_features]
            
        Returns:
            Energy values [batch_size, num_levels, num_features]
        """
        # Energy = sum of squared coefficients
        squared_coeffs = coeffs ** 2
        
        # Energy per level per feature
        batch_size, _, num_features = coeffs.shape
        level_feature_energies = []
        
        for i, (start_idx, dim) in enumerate(zip(self.level_start_indices, self.level_dims)):
            end_idx = start_idx + dim
            level_coeffs = squared_coeffs[:, start_idx:end_idx, :]  # [batch_size, dim, num_features]
            level_energy = torch.sum(level_coeffs, dim=1)  # [batch_size, num_features]
            level_feature_energies.append(level_energy)
        
        return torch.stack(level_feature_energies, dim=1)  # [batch_size, num_levels, num_features]

    def _compute_energy_loss(self, target: torch.Tensor, prediction: torch.Tensor) -> torch.Tensor:
        """
        Compute energy preservation loss between target and prediction.
        Uses absolute normalization: |E_target - E_pred|
        
        Args:
            target: Target coefficients [batch_size, total_coeffs_per_feature, num_features]
            prediction: Predicted coefficients [batch_size, total_coeffs_per_feature, num_features]
            
        Returns:
            Energy loss scalar
        """
        # Compute energies
        energy_target = self._compute_energy(target)
        energy_pred = self._compute_energy(prediction)
        
        # Use absolute normalization
        energy_diff = torch.abs(energy_target - energy_pred)
        
        # Use MEAN to get scalar loss
        energy_loss = torch.mean(energy_diff)
        
        return energy_loss

    
    def compute_loss(self, 
                    target: torch.Tensor, 
                    prediction: torch.Tensor) -> torch.Tensor:
        """
        Compute balanced loss across wavelet levels.
        
        Args:
            target: Ground truth coefficients [batch_size, total_coeffs]
            prediction: Predicted coefficients [batch_size, total_coeffs]
            
        Returns:
            Computed loss value
        """
        # Validate tensor shapes
        self._validate_tensor_shape(target, "target")
        self._validate_tensor_shape(prediction, "prediction")

        if self.strategy == "standard":
            reconstruction_loss = F.mse_loss(target, prediction)
        else:
            reconstruction_loss = self._compute_weighted_loss(target, prediction)
        
        # Compute energy preservation loss
        if self.use_energy_term:
            energy_loss = self._compute_energy_loss(target, prediction)

            # Combine losses with current energy weight
            total_loss = reconstruction_loss + self.energy_weight * energy_loss
            return total_loss
        else:
            return reconstruction_loss
    
    def _compute_weighted_loss(self, target: torch.Tensor, prediction: torch.Tensor) -> torch.Tensor:
        """Compute weighted loss across levels."""
        batch_size, total_coeffs_per_feature, num_features = target.shape
        total_loss = 0.0

        # Compute loss for each feature separately, then average
        for feature_idx in range(num_features):
            feature_total_loss = 0.0
            target_feature = target[:, :, feature_idx]  # [batch_size, total_coeffs_per_feature]
            pred_feature = prediction[:, :, feature_idx]  # [batch_size, total_coeffs_per_feature]
            
            for i, (start_idx, dim, weight) in enumerate(zip(
                self.level_start_indices, self.level_dims, self.level_weights
            )):
                end_idx = start_idx + dim
                level_target = target_feature[:, start_idx:end_idx]
                level_pred = pred_feature[:, start_idx:end_idx]
                
                level_loss = F.mse_loss(level_target, level_pred)
                feature_total_loss += weight * level_loss

            total_loss += feature_total_loss

        # Average across features
        return total_loss / num_features
    
    
    def get_level_losses(self, target: torch.Tensor, prediction: torch.Tensor) -> List[torch.Tensor]:
        """Get individual losses for each level (useful for logging)."""
        # Validate tensor shapes
        self._validate_tensor_shape(target, "target")
        self._validate_tensor_shape(prediction, "prediction")
        
        batch_size, total_coeffs_per_feature, num_features = target.shape
        level_losses = []

        for i, (start_idx, dim) in enumerate(zip(self.level_start_indices, self.level_dims)):
            end_idx = start_idx + dim
            level_loss_total = 0.0
            
            for feature_idx in range(num_features):
                target_feature = target[:, :, feature_idx]
                pred_feature = prediction[:, :, feature_idx]
                level_target = target_feature[:, start_idx:end_idx]
                level_pred = pred_feature[:, start_idx:end_idx]
                level_loss = F.mse_loss(level_target, level_pred)
                level_loss_total += level_loss
            
            # Average across features
            level_losses.append(level_loss_total / num_features)
        
        return level_losses
    
    def get_weights(self) -> List[float]:
        """Get current level weights."""
        return self.level_weights.copy()
    
    def get_energy_loss(self, target: torch.Tensor, prediction: torch.Tensor) -> torch.Tensor:
        """Get energy preservation loss."""
        return self._compute_energy_loss(target, prediction)
    
    def get_energy_stats(self, target: torch.Tensor, prediction: torch.Tensor) -> dict:
        """Get detailed energy statistics for analysis."""
        self._validate_tensor_shape(target, "target")
        self._validate_tensor_shape(prediction, "prediction")
        
        energy_target = self._compute_energy(target)
        energy_pred = self._compute_energy(prediction)
        
        stats = {
            "energy_target_mean": torch.mean(energy_target).item(),
            "energy_pred_mean": torch.mean(energy_pred).item(),
            "energy_target_std": torch.std(energy_target).item(),
            "energy_pred_std": torch.std(energy_pred).item(),
            "energy_absolute_error": torch.mean(
                torch.abs(energy_target - energy_pred)
            ).item()
        }
        
        return stats


def get_wavelet_loss_fn(level_dims: List[int], 
                       level_start_indices: List[int], 
                       strategy: str = "level_weighted",
                       **kwargs) -> Callable:
    """
    Factory function to create a wavelet loss function.
    
    Args:
        level_dims: List of coefficient counts for each wavelet level
        level_start_indices: Starting indices for each level
        strategy: Loss balancing strategy
        **kwargs: Additional arguments for WaveletBalancedLoss
    
    Returns:
        Loss function that takes (target, prediction) and returns loss
    """
    loss_computer = WaveletBalancedLoss(level_dims, level_start_indices, strategy, **kwargs)
    
    def loss_fn(target: torch.Tensor, prediction: torch.Tensor) -> torch.Tensor:
        return loss_computer.compute_loss(target, prediction)
    
    # Attach methods for convenience
    loss_fn.get_level_losses = loss_computer.get_level_losses
    loss_fn.get_weights = loss_computer.get_weights
    loss_fn.get_energy_loss = loss_computer.get_energy_loss
    loss_fn.get_energy_stats = loss_computer.get_energy_stats
    loss_fn.loss_computer = loss_computer
    
    return loss_fn
