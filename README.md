# WaveletDiff: Multilevel Wavelet Diffusion for Time Series Generation
This repository contains code for the paper [WaveletDiff: Multilevel Wavelet Diffusion for Time Series Generation](https://arxiv.org/abs/2510.11839)

## Requirements

### Installation

1. Clone the repository:
```bash
git clone <repository_url>
cd WaveletDiff
```

2. Create and activate conda environment:
```bash
conda create -n waveletdiff python=3.13
conda activate waveletdiff
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Dependencies
- PyTorch == 2.7.1
- PyTorch Lightning == 2.5.2
- PyWavelets == 1.8.0
- NumPy == 2.3.1
- Pandas == 2.3.0
- SciPy == 1.16.0
- PyYAML == 6.0.2
- scikit-learn == 1.7.0
- tqdm == 4.67.1
- jupyter == 1.1.1
- matplotlib == 3.10.3
- seaborn == 0.13.2


## Training

Train WaveletDiff model with default settings:

```bash
cd src
python train.py --experiment_name default_experiment
```

**Note**: This code is compatible with HPC systems using SLURM for running training and sampling jobs on cluster environments.

### Configuration-based Training

You can override the default configuration file by specifying parameters directly when running the training script:

```bash
python train.py --experiment_name default_experiment --dataset etth1 --seq_len 24 --epochs 5000 --batch_size 512
```

### Configuration Files

The system automatically uses configuration files from the `configs/` directory:

- `configs/default.yaml`: Default hyperparameters for all settings
- `configs/datasets/`: Dataset-specific configurations (etth1.yaml, etth2.yaml, eeg.yaml, etc.)

The system automatically loads the default configuration and applies dataset-specific overrides when you specify a `--dataset` parameter. You can override individual parameters using command-line arguments or edit through the config files.

Key configuration sections:
- `model`: Architecture settings (embedding dimensions, number of layers)
- `training`: Training parameters (epochs, batch size, saving model)
- `attention`: Cross-level attention
- `energy`: Energy preservation weight
- `noise`: Noise schedule (cosine, linear, exponential)
- `wavelet`: Wavelet type and decomposition levels
- `data`: Data loading and preprocessing options
- `sampling`: Sampling method choice (DDPM or DDIM)
- `optimizer`: Learning rate, scheduler, and optimization settings

### Supported Datasets

The model supports the following 6 time series datasets:
- **ETTh1**: Electricity Transformer Temperature dataset 1
- **ETTh2**: Electricity Transformer Temperature dataset 2  
- **Stocks**: Stock market data
- **Exchange Rate**: Currency exchange rates
- **fMRI**: Functional magnetic resonance imaging time series
- **EEG**: Electroencephalogram data

## Sampling

### Sample Generation

Generate samples from a trained model:

```bash
cd src
python sample.py --experiment_name default_experiment --dataset etth1 --num_samples 10000 --sampling_method ddpm
```

Available sampling methods:
- `ddpm`: Standard DDPM sampling (stochastic)
- `ddim`: DDIM sampling (deterministic when eta=0)


## Evaluation

The evaluation is done in the notebook `src/evaluation/evaluation.ipynb`.

This notebook supports the following metrics:

- **Discriminative Score**: Measures how well a classifier can distinguish between real and generated data (lower is better)
- **Predictive Score**: Mean Absolute Error of one-step ahead prediction using an RNN (lower is better)
- **Context-FID Score**: Context-aware Fréchet Inception Distance using TS2Vec embeddings (lower is better)
- **Correlational Score**: Measures temporal correlation patterns between real and generated data (lower is better)
- **DTW-JS Distance**: Dynamic Time Warping with Jensen-Shannon divergence for temporal alignment (lower is better)

## Citation

```bibtex
@article{wang2025waveletdiff,
    title={WaveletDiff: Multilevel Wavelet Diffusion For Time Series Generation}, 
    author={Yu-Hsiang Wang and Olgica Milenkovic},
    journal={arXiv preprint arXiv:2510.11839},
    year={2025},
}
```