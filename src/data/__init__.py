"""
Data module for wavelet diffusion models.
"""

from .module import WaveletTimeSeriesDataModule
from .loaders import (
    load_ett_data, load_fmri_data, load_exchange_rate_data,
    load_stocks_data, load_eeg_data
)

__all__ = [
    'WaveletTimeSeriesDataModule',
    'load_ett_data', 'load_fmri_data', 'load_exchange_rate_data',
    'load_stocks_data', 'load_eeg_data'
]

