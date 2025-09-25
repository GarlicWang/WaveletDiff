"""
Dataset loading utilities for various time series datasets.

Extracted from the main data module to keep it clean and focused.
"""

import pandas as pd
import numpy as np
import torch
import os
from typing import Dict, Any, Tuple

def create_sliding_windows(data: np.ndarray, 
                          seq_len: int, 
                          stride: int = 1,
                          normalize: bool = True) -> Tuple[np.ndarray, dict]:
    """
    Create sliding window samples from long time series data.
    
    Args:
        data: Time series data of shape (total_timesteps, n_features)
        seq_len: Length of each window/sample
        stride: Step size between windows.
        normalize: Whether to apply feature-wise standardization
        
    Returns:
        Tuple of (windowed_data, norm_stats) where:
        - windowed_data: shape (n_samples, seq_len, n_features)
        - norm_stats: dict with 'mean' and 'std' arrays if normalize=True, else None
    """

    total_timesteps, n_features = data.shape
    
    if seq_len > total_timesteps:
        raise ValueError(f"seq_len ({seq_len}) cannot be larger than total timesteps ({total_timesteps})")
    
    norm_stats = None
    if normalize:
        # Ensure data is float type to avoid numpy issues
        data = data.astype(np.float32)
        
        # Store normalization statistics for later reconstruction
        data_mean = np.mean(data, axis=0)
        data_std = np.std(data, axis=0)
        
        # Handle zero variance features to avoid division by zero
        data_std = np.where(data_std == 0, 1.0, data_std)
        
        norm_stats = {
            'mean': data_mean,
            'std': data_std
        }
        
        # Standardization
        data = (data - data_mean) / data_std

    # Calculate number of possible windows
    n_samples = (total_timesteps - seq_len) // stride + 1
    
    # Create windows
    windows = []
    for i in range(n_samples):
        start_idx = i * stride
        end_idx = start_idx + seq_len
        
        if end_idx <= total_timesteps:
            windows.append(data[start_idx:end_idx])
    
    return np.array(windows), norm_stats


def load_ett_data(dataset_name: str, data_dir: str, seq_len: int = 24, normalize_data: bool = True) -> Tuple[torch.Tensor, dict]:
    """Load ETT dataset."""
    ett_path = os.path.join(data_dir, "ETT-small", f"ETT{dataset_name[-2:]}.csv")
    if not os.path.exists(ett_path):
        raise FileNotFoundError(f"ETT data not found at: {ett_path}")
    
    df = pd.read_csv(ett_path)
    data, norm_stats = create_sliding_windows(df.values[:, 1:], seq_len=seq_len, stride=1, normalize=normalize_data)
    data = data.astype(np.float32)
    return torch.FloatTensor(data), norm_stats


def load_fmri_data(data_dir: str, seq_len: int = 24, normalize_data: bool = True) -> Tuple[torch.Tensor, dict]:
    """Load fMRI dataset."""
    fmri_path = os.path.join(data_dir, "fMRI", "sim4.mat")
    if not os.path.exists(fmri_path):
        raise FileNotFoundError(f"fMRI data not found at: {fmri_path}")

    from scipy.io import loadmat
    data = loadmat(fmri_path)

    # Diffusion-TS's implementation: (1, 10000, 50)
    data, norm_stats = create_sliding_windows(data['ts'], seq_len=seq_len, stride=1, normalize=normalize_data)
    data = data.astype(np.float32)
    
    return torch.FloatTensor(data), norm_stats


def load_exchange_rate_data(data_dir: str, seq_len: int = 24, normalize_data: bool = True) -> Tuple[torch.Tensor, dict]:
    """Load Exchange Rate dataset."""
    exchange_rate_path = os.path.join(data_dir, "exchange_rate", "exchange_rate.txt")
    if not os.path.exists(exchange_rate_path):
        raise FileNotFoundError(f"Exchange rate data not found at: {exchange_rate_path}")

    df = pd.read_csv(exchange_rate_path, header=None)
    data, norm_stats = create_sliding_windows(df.values, seq_len=seq_len, stride=1, normalize=normalize_data)
    data = data.astype(np.float32)
    
    return torch.FloatTensor(data), norm_stats


def load_stocks_data(data_dir: str, seq_len: int = 24, normalize_data: bool = True) -> Tuple[torch.Tensor, dict]:
    """Load Stocks dataset."""
    stocks_path = os.path.join(data_dir, "stocks", "stock_data.csv")
    if not os.path.exists(stocks_path):
        raise FileNotFoundError(f"Stocks data not found at: {stocks_path}")

    df = pd.read_csv(stocks_path)
    data, norm_stats = create_sliding_windows(df.values, seq_len=seq_len, stride=1, normalize=normalize_data)
    data = data.astype(np.float32)
    
    return torch.FloatTensor(data), norm_stats


def load_eeg_data(data_dir: str, seq_len: int = 24, normalize_data: bool = True) -> Tuple[torch.Tensor, dict]:
    """Load EEG Eye State dataset."""
    from scipy.io import arff
    
    eeg_path = os.path.join(data_dir, "EEG", "EEG_Eye_State.arff")
    if not os.path.exists(eeg_path):
        raise FileNotFoundError(f"EEG data not found at: {eeg_path}")
    
    # Load ARFF file
    eeg_data, eeg_meta = arff.loadarff(eeg_path)
    eeg_df = pd.DataFrame(eeg_data)

    # Use all columns except the last one (which is the eye state label)
    data, norm_stats = create_sliding_windows(eeg_df.values[:, :-1], seq_len=seq_len, stride=1, normalize=normalize_data)
    data = data.astype(np.float32)
    
    return torch.FloatTensor(data), norm_stats
