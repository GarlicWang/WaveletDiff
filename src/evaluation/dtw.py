"""
Dynamic Time Warping (DTW) utilities for time series analysis.

This module provides DTW algorithms for measuring similarity between time series
of different lengths, focusing on Euclidean distance and distribution-level metrics.
"""

import numpy as np
from typing import Tuple, Optional, Union, List
import warnings
from tqdm import tqdm
import multiprocessing as mp
from functools import partial


def dtw_distance(
    ts1: np.ndarray,
    ts2: np.ndarray,
    window: Optional[int] = None,
    normalize: bool = False
) -> float:
    """
    Compute the Dynamic Time Warping distance between two time series.
    
    Args:
        ts1: First time series (1D array)
        ts2: Second time series (1D array)
        window: Sakoe-Chiba band constraint (None for no constraint)
        normalize: Whether to normalize the DTW distance by path length
        
    Returns:
        DTW distance between the two time series
        
    Example:
        >>> ts1 = np.array([1, 2, 3, 4, 5])
        >>> ts2 = np.array([2, 3, 4, 5, 6])
        >>> distance = dtw_distance(ts1, ts2)
    """
    ts1 = np.asarray(ts1)
    ts2 = np.asarray(ts2)
    
    # Handle multi-feature time series
    if ts1.ndim == 2 and ts2.ndim == 2:
        if ts1.shape[1] != ts2.shape[1]:
            raise ValueError(f"Number of features must match: {ts1.shape[1]} vs {ts2.shape[1]}")
        
        # Compute DTW for each feature separately and take mean
        feature_distances = []
        for feature_idx in range(ts1.shape[1]):
            feature_dist = dtw_distance(ts1[:, feature_idx], ts2[:, feature_idx], 
                                      window, normalize)
            feature_distances.append(feature_dist)
        
        return np.mean(feature_distances)
    
    elif (ts1.ndim == 1 and ts2.ndim == 2) or (ts1.ndim == 2 and ts2.ndim == 1):
        raise ValueError("Both time series must have the same dimensionality (both 1D or both 2D)")
    elif ts1.ndim > 2 or ts2.ndim > 2:
        raise ValueError("Time series must be 1D (seq_len,) or 2D (seq_len, n_features) arrays")
    
    n, m = len(ts1), len(ts2)
    
    # Initialize DTW matrix with infinity
    dtw_matrix = np.full((n + 1, m + 1), np.inf)
    dtw_matrix[0, 0] = 0
    
    # Apply Sakoe-Chiba band constraint if specified
    if window is not None:
        if window < abs(n - m):
            warnings.warn(
                f"Window size {window} is smaller than length difference {abs(n-m)}. "
                "This may result in no valid path."
            )
    
    for i in range(1, n + 1):
        # Determine the range of j values based on window constraint
        if window is None:
            j_start, j_end = 1, m + 1
        else:
            j_start = max(1, i - window)
            j_end = min(m + 1, i + window + 1)
        
        for j in range(j_start, j_end):
            # Euclidean distance
            cost = abs(ts1[i-1] - ts2[j-1])
            dtw_matrix[i, j] = cost + min(
                dtw_matrix[i-1, j],      # insertion
                dtw_matrix[i, j-1],      # deletion
                dtw_matrix[i-1, j-1]     # match
            )
    
    distance = dtw_matrix[n, m]
    
    if distance == np.inf:
        raise ValueError("No valid DTW path found. Consider increasing the window size.")
    
    if normalize:
        # Normalize by the length of the optimal path
        path_length = _get_path_length(dtw_matrix, n, m)
        distance = distance / path_length
    
    return distance


def _get_path_length(dtw_matrix: np.ndarray, n: int, m: int) -> int:
    """Get the length of the optimal DTW path by backtracking."""
    path_length = 0
    i, j = n, m
    
    while i > 0 and j > 0:
        path_length += 1
        
        # Find the minimum of the three possible previous positions
        if i == 1:
            j -= 1
        elif j == 1:
            i -= 1
        else:
            min_val = min(dtw_matrix[i-1, j], dtw_matrix[i, j-1], dtw_matrix[i-1, j-1])
            if dtw_matrix[i-1, j-1] == min_val:
                i -= 1
                j -= 1
            elif dtw_matrix[i-1, j] == min_val:
                i -= 1
            else:
                j -= 1
    
    return path_length


def _compute_sample_distances(args):
    """Helper function for multiprocessing sample distance computation."""
    sample, reference_set, window, normalize = args
    sample = np.asarray(sample)
    sample_distances = []
    
    for ref_sample in reference_set:
        ref_sample = np.asarray(ref_sample)
        dist = dtw_distance(sample, ref_sample, window, normalize)
        sample_distances.append(dist)
    
    return np.mean(sample_distances)


def dtw_js_divergence_distance(
    real_samples: Union[list, np.ndarray],
    generated_samples: Union[list, np.ndarray],
    window: Optional[int] = None,
    normalize: bool = True,
    n_bins: int = 50,
    reference_samples: Optional[Union[list, np.ndarray]] = None,
    n_jobs: Optional[int] = None,
    n_samples: Optional[int] = None
) -> dict:
    """
    Measure DTW-based Jensen-Shannon divergence between two distributions.
    
    This approach compares the distributions of DTW distances rather than individual samples.
    It computes DTW distances from each sample to a reference set, then compares the
    resulting distance distributions using Jensen-Shannon divergence.
    
    Args:
        generated_samples: List of generated time series
        real_samples: List of real time series
        window: Sakoe-Chiba band constraint
        normalize: Whether to normalize DTW distances
        n_bins: Number of bins for histogram computation
        reference_samples: Reference set for DTW distance computation 
                          (if None, uses combined samples)
        n_jobs: Number of parallel jobs. If None, uses all available cores.
        n_samples: Number of samples to use from each distribution (for efficiency).
                  If None, uses all available samples.
        
    Returns:
        Dictionary containing:
        - 'js_divergence': Jensen-Shannon divergence between DTW distance distributions
        - 'kl_div_gen_to_real': KL divergence from generated to real distribution
        - 'kl_div_real_to_gen': KL divergence from real to generated distribution
        - 'generated_distances': DTW distances for generated samples
        - 'real_distances': DTW distances for real samples
        
    Example:
        >>> generated = [model.generate() for _ in range(100)]
        >>> real = [load_real_data() for _ in range(100)]
        >>> result = dtw_js_divergence_distance(generated, real, n_samples=50)
        >>> print(f"JS divergence: {result['js_divergence']:.4f}")
    """
    # Convert numpy arrays to lists if needed
    if isinstance(generated_samples, np.ndarray):
        if generated_samples.size == 0:
            raise ValueError("generated_samples array is empty")
        generated_samples = [generated_samples[i] for i in range(generated_samples.shape[0])]
    
    if isinstance(real_samples, np.ndarray):
        if real_samples.size == 0:
            raise ValueError("real_samples array is empty")
        real_samples = [real_samples[i] for i in range(real_samples.shape[0])]
    
    if isinstance(reference_samples, np.ndarray):
        reference_samples = [reference_samples[i] for i in range(reference_samples.shape[0])]
    
    # Check if lists are empty
    if len(generated_samples) == 0 or len(real_samples) == 0:
        raise ValueError("Both generated_samples and real_samples must be non-empty")
    
    # Sample from distributions if n_samples is specified and they're too large
    if n_samples is not None:
        if len(generated_samples) > n_samples:
            # Use different seed each time for variety
            random_seed = np.random.randint(0, 1000000)
            np.random.seed(random_seed)
            gen_indices = np.random.choice(len(generated_samples), size=n_samples, replace=False)
            generated_samples = [generated_samples[i] for i in gen_indices]
        
        if len(real_samples) > n_samples:
            # Use different seed each time for variety
            random_seed = np.random.randint(0, 1000000)
            np.random.seed(random_seed)
            real_indices = np.random.choice(len(real_samples), size=n_samples, replace=False)
            real_samples = [real_samples[i] for i in real_indices]
    
    # Create reference set for DTW distance computation
    if reference_samples is None:
        # Use a subset of both distributions as reference
        n_ref = min(100, len(generated_samples) + len(real_samples))
        all_samples = generated_samples + real_samples
        # Use different seed each time for variety
        random_seed = np.random.randint(0, 1000000)
        np.random.seed(random_seed)
        ref_indices = np.random.choice(len(all_samples), size=n_ref, replace=False)
        reference_samples = [all_samples[i] for i in ref_indices]
    
    # Set number of jobs for multiprocessing
    if n_jobs is None:
        n_jobs = mp.cpu_count()
    
    def compute_distance_distribution(samples, reference_set):
        """Compute DTW distances from samples to reference set using multiprocessing."""
        if n_jobs > 1 and len(samples) > 1:
            # Prepare arguments for multiprocessing
            args_list = [(sample, reference_set, window, normalize) for sample in samples]
            
            # Use multiprocessing
            with mp.Pool(processes=n_jobs) as pool:
                distances = list(tqdm(
                    pool.imap(_compute_sample_distances, args_list),
                    total=len(samples),
                    desc="Computing DTW distances (Parallel)"
                ))
            
            return np.array(distances)
        else:
            # Fallback to sequential processing
            distances = []
            for sample in tqdm(samples, desc="Computing DTW distances"):
                sample = np.asarray(sample)
                sample_distances = []
                
                for ref_sample in reference_set:
                    ref_sample = np.asarray(ref_sample)
                    dist = dtw_distance(sample, ref_sample, window, normalize)
                    sample_distances.append(dist)
                
                # Use mean distance to reference set as the representative distance
                distances.append(np.mean(sample_distances))
            
            return np.array(distances)
    
    # Compute distance distributions
    generated_distances = compute_distance_distribution(generated_samples, reference_samples)
    real_distances = compute_distance_distribution(real_samples, reference_samples)
    
    # Determine histogram bins based on combined data range
    all_distances = np.concatenate([generated_distances, real_distances])
    min_dist, max_dist = np.min(all_distances), np.max(all_distances)
    
    # Add small margin to avoid edge effects
    margin = (max_dist - min_dist) * 0.05
    bins = np.linspace(min_dist - margin, max_dist + margin, n_bins + 1)
    
    # Compute histograms
    generated_hist, _ = np.histogram(generated_distances, bins=bins, density=True)
    real_hist, _ = np.histogram(real_distances, bins=bins, density=True)
    
    # Normalize to create probability distributions
    generated_hist = generated_hist / np.sum(generated_hist)
    real_hist = real_hist / np.sum(real_hist)
    
    # Add small epsilon to avoid log(0) in KL divergence
    eps = 1e-10
    generated_hist = generated_hist + eps
    real_hist = real_hist + eps
    
    # Renormalize after adding epsilon
    generated_hist = generated_hist / np.sum(generated_hist)
    real_hist = real_hist / np.sum(real_hist)
    
    # Compute Jensen-Shannon divergence
    # JS(P,Q) = 0.5 * KL(P,M) + 0.5 * KL(Q,M), where M = 0.5*(P+Q)
    m_distribution = 0.5 * (generated_hist + real_hist)
    
    def kl_divergence(p, q):
        """Compute KL divergence KL(P||Q)."""
        return np.sum(p * np.log(p / q))
    
    kl_gen_to_m = kl_divergence(generated_hist, m_distribution)
    kl_real_to_m = kl_divergence(real_hist, m_distribution)
    js_divergence = 0.5 * kl_gen_to_m + 0.5 * kl_real_to_m
    
    # Also compute individual KL divergences for additional insight
    kl_gen_to_real = kl_divergence(generated_hist, real_hist)
    kl_real_to_gen = kl_divergence(real_hist, generated_hist)
    
    return {
        'js_divergence': js_divergence,
        'kl_div_gen_to_real': kl_gen_to_real,
        'kl_div_real_to_gen': kl_real_to_gen,
        'generated_distances': generated_distances,
        'real_distances': real_distances,
        'reference_set_size': len(reference_samples)
    }


