"""Wavelet energy computation utilities for WaveletDiff."""

import numpy as np
import torch


def wavelet_energy(coeffs):
    """
    Compute the energy of wavelet coefficients using Parseval's theorem.

    Args:
        coeffs (np.ndarray or torch.Tensor): Wavelet coefficients, shape (..., N)
    Returns:
        float: Energy of the coefficients
    """
    if hasattr(coeffs, 'detach'):  # torch.Tensor
        coeffs = coeffs.detach().cpu().numpy()
    energy = np.sum(np.square(coeffs), axis=1)
    return energy


def compute_wavelet_energy(coeffs, level_wise=False):
    """
    Compute wavelet energy with optional level-wise breakdown.
    
    Args:
        coeffs: Wavelet coefficients
        level_wise: If True, return energy per level
        
    Returns:
        Energy values (total or per-level)
    """
    if level_wise:
        # Compute energy for each level separately
        if isinstance(coeffs, (list, tuple)):
            energies = []
            for level_coeffs in coeffs:
                if hasattr(level_coeffs, 'detach'):
                    level_coeffs = level_coeffs.detach().cpu().numpy()
                energy = np.sum(np.square(level_coeffs))
                energies.append(energy)
            return energies
        else:
            return wavelet_energy(coeffs)
    else:
        return wavelet_energy(coeffs)



