"""
Sample generation script for trained WaveletDiff models.
"""

import sys
import argparse
import torch
import numpy as np
from pathlib import Path

from models import WaveletDiffusionTransformer
from training import DiffusionTrainer
from data import WaveletTimeSeriesDataModule
from utils import ConfigManager


def main():
    """Generate samples from a trained model."""
    parser = argparse.ArgumentParser(description='Generate samples from WaveletDiff model')
    parser.add_argument('--experiment_name', type=str, required=True,
                       help='Experiment name (matches train.py experiment_name)')
    parser.add_argument('--dataset', type=str, default=None,
                       help='Dataset name (overrides config)')
    parser.add_argument('--num_samples', type=int, default=10000,
                       help='Number of samples to generate')
    parser.add_argument('--sampling_method', type=str, choices=['ddpm', 'ddim'], default=None,
                       help='Sampling method (overrides config)')
    
    args = parser.parse_args()
    
    # Load configuration
    config_manager = ConfigManager()
    dataset_name = args.dataset or 'etth1'
    config = config_manager.load(dataset_name=dataset_name)
    
    if args.dataset:
        config['dataset']['name'] = args.dataset
    if args.sampling_method:
        config['sampling']['method'] = args.sampling_method
    
    # Construct model path from experiment name
    experiment_dir = Path(config['paths']['output_dir']) / args.experiment_name
    model_path = experiment_dir / 'checkpoint.ckpt'
    
    if not model_path.exists():
        print(f"Error: Model checkpoint not found at {model_path}")
        print(f"Make sure you've trained a model with --experiment_name {args.experiment_name}")
        sys.exit(1)
    
    print(f"Generating Samples from WaveletDiff Model")
    print(f"Experiment: {args.experiment_name}")
    print(f"Dataset: {config['dataset']['name']}")
    print(f"Model: {model_path}")
    print(f"Samples: {args.num_samples}")
    print(f"Sampling Method: {config['sampling']['method'].upper()}")
    
    # Set up data module
    data_module = WaveletTimeSeriesDataModule(config=config)
    
    # Load model
    print("Loading model...")
    model = WaveletDiffusionTransformer.load_from_checkpoint(
        model_path,
        data_module=data_module,
        config=config,
    )
    model.eval()
    
    # Move to GPU if available
    if torch.cuda.is_available():
        model = model.cuda()
        print("Model moved to GPU")
    
    # Create trainer for evaluation
    trainer_util = DiffusionTrainer(model)
    
    # Generate samples
    print(f"Generating {args.num_samples} samples...")
    
    # Generate using specified method
    sampling_method = config['sampling']['method']
    use_ddim = (sampling_method == "ddim")
    
    print(f"Generating {sampling_method.upper()} samples...")
    samples = trainer_util.generate_samples(args.num_samples, use_ddim=use_ddim)
    
    # Convert to time series
    print("Converting to time series...")
    samples_ts = data_module.convert_wavelet_to_timeseries(samples)
    
    # Save results in the same experiment directory
    real_samples_path = experiment_dir / "real_samples.npy"
    output_file_path = experiment_dir / f"{sampling_method}_samples.npy"
    
    np.save(real_samples_path, data_module.raw_data_tensor.numpy())
    np.save(output_file_path, samples_ts.cpu().numpy())
    
    print(f"Real samples saved to {real_samples_path}")
    print(f"Generated samples saved to {output_file_path}")


if __name__ == "__main__":
    main()

