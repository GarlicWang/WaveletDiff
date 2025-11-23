# Dataset Directory

This directory should contain the time series datasets used for training and evaluation of the WaveletDiff model.

## Directory Structure

```
data/
├── README.md                    # This file
├── ETT-small/                   # Electricity Transformer Temperature datasets
│   ├── ETTh1.csv                # ETTh1 dataset
│   └── ETTh2.csv                # ETTh2 dataset
├── exchange_rate/               # Currency exchange rate data
│   └── exchange_rate.txt
├── stocks/                      # Stock market data
│   └── stock_data.csv
├── fMRI/                        # Functional MRI time series
│   └── sim4.mat
└── EEG/                         # Electroencephalogram data
    └── EEG_Eye_State.arff
```

## Dataset Sources

### Current Datasets

#### ETT (Electricity Transformer Temperature)
- **ETTh1 & ETTh2**: [Download from here](https://github.com/zhouhaoyi/ETDataset)
- **Description**: Hourly data from electricity transformers
- **Features**: 7 features (HUFL, HULL, MUFL, MULL, LUFL, LULL, OT)
- **Time Range**: 2016-07-01 to 2018-07-01 (ETTh1), 2016-07-01 to 2018-07-01 (ETTh2)

#### Exchange Rate
- **Source**: [Download from here](https://github.com/laiguokun/multivariate-time-series-data)
- **Description**: Daily exchange rates of 8 countries
- **Features**: 8 features (Australia, Canada, Switzerland, China, Japan, New Zealand, Singapore, UK)
- **Time Range**: 1990-01-01 to 2016-12-31

#### Stocks
- **Source**: [Download from here](https://github.com/Y-debug-sys/Diffusion-TS.git) (data file available via the Google Drive link provided by Diffusion-TS)
- **Description**: Stock market data
- **Features**: 5 features (different stock indices)
- **Time Range**: Varies by stock

#### fMRI
- **Source**: [Download from here](https://www.fmrib.ox.ac.uk/datasets/netsim/) (the original version is used rather than the slightly improved version)
- **Description**: Simulated fMRI time series
- **Features**: 50 features (brain regions)
- **Time Range**: 10,000 time points

#### EEG Eye State
- **Source**: [Download from here](https://archive.ics.uci.edu/dataset/264/eeg+eye+state)
- **Description**: EEG recordings during eye state classification
- **Features**: 14 features (EEG channels)
- **Time Range**: 14,980 time points

## Data Preprocessing

The `create_sliding_windows` function automatically:
- Creates sliding window samples from long time series
- Applies feature-wise standardization (if `normalize_data=True`)
- Handles zero-variance features
- Returns normalized statistics for reconstruction

## Adding New Datasets

### Step 1: Create Dataset Directory
```bash
mkdir data/your_dataset_name
```

### Step 2: Add Data File
Place your data file in the new directory:
```bash
data/your_dataset_name/your_data_file.csv
```

### Step 3: Update Data Loader
Add a new loader function in `src/data/loaders.py`:

```python
def load_your_dataset_data(data_dir: str, seq_len: int = 24, normalize_data: bool = True) -> Tuple[torch.Tensor, dict]:
    """Load Your Dataset."""
    dataset_path = os.path.join(data_dir, "your_dataset_name", "your_data_file.csv")
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Your dataset not found at: {dataset_path}")

    df = pd.read_csv(dataset_path)
    # Adjust column selection based on your data format
    data, norm_stats = create_sliding_windows(df.values, seq_len=seq_len, stride=1, normalize=normalize_data)
    data = data.astype(np.float32)
    
    return torch.FloatTensor(data), norm_stats
```

### Step 4: Update Data Module
Add the new loader to `src/data/__init__.py`:
```python
from .loaders import (
    load_ett_data, load_fmri_data, load_exchange_rate_data,
    load_stocks_data, load_eeg_data, load_your_dataset_data  # Add your loader
)

__all__ = [
    'WaveletTimeSeriesDataModule',
    'load_ett_data', 'load_fmri_data', 'load_exchange_rate_data',
    'load_stocks_data', 'load_eeg_data', 'load_your_dataset_data'  # Add your loader
]
```

### Step 5: Update Data Module Logic
Add your dataset to the `_load_dataset` method in `src/data/module.py`:
```python
elif dataset_name == "your_dataset_name":
    raw_data, norm_stats = load_your_dataset_data(self.data_dir, seq_len=seq_len, normalize_data=normalize_data)
```

### Step 6: Create Configuration
Create a configuration file `configs/datasets/your_dataset_name.yaml`:
```yaml
dataset:
  name: "your_dataset_name"
  seq_len: 24  # Adjust based on your data

training:
  # Add any dataset-specific training parameters

wavelet:
  # Add any dataset-specific wavelet parameters
```
