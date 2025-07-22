from __future__ import annotations

"""tfn.datasets.climate_loader
Climate and time series dataset loaders with Kaggle dataset support.
"""

from typing import List, Tuple, Dict
import random
from pathlib import Path
from tfn.utils.data_utils import split_indices

import torch
from torch.utils.data import TensorDataset

# Optional dependency ---------------------------------------------------------
try:
    from datasets import load_dataset  # type: ignore
    _HAVE_HF = True
except ImportError:
    _HAVE_HF = False

# ---------------------------------------------------------------------------
# Time series processing helpers
# ---------------------------------------------------------------------------

def _normalize_sequence(sequence: List[float]) -> List[float]:
    """Normalize a time series sequence to [0, 1] range."""
    if not sequence:
        return sequence
    
    min_val = min(sequence)
    max_val = max(sequence)
    
    if max_val == min_val:
        return [0.5] * len(sequence)
    
    return [(x - min_val) / (max_val - min_val) for x in sequence]


def _create_sequences(data: List[float], seq_len: int = 128, step: int = 1) -> Tuple[List[List[float]], List[float]]:
    """Create sliding window sequences for time series prediction."""
    sequences = []
    targets = []
    
    for i in range(0, len(data) - seq_len, step):
        seq = data[i:i + seq_len]
        target = data[i + seq_len] if i + seq_len < len(data) else data[-1]
        sequences.append(seq)
        targets.append(target)
    
    return sequences, targets


def _sequences_to_tensor(sequences: List[List[float]], targets: List[float]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert sequences and targets to tensors."""
    seq_tensor = torch.tensor(sequences, dtype=torch.float)
    target_tensor = torch.tensor(targets, dtype=torch.float)
    return seq_tensor, target_tensor

# ---------------------------------------------------------------------------
# Climate dataset loaders
# ---------------------------------------------------------------------------

def load_electricity_transformer(
    seq_len: int = 128,
    val_split: int = 1000,
    step: int = 1,
    shuffle_train: bool = False,
    shuffle_eval: bool = False,
):
    """Electricity Transformer Temperature prediction (regression).
    
    Predicts transformer temperature based on historical temperature data.
    """
    import pandas as pd
    
    # Try Kaggle path first
    kaggle_path = Path("/kaggle/input/electricity-transformer-temperature/electricity_transformer_temperature.csv")
    if kaggle_path.exists():
        df = pd.read_csv(kaggle_path)
        # Use temperature column for prediction
        if 'temperature' in df.columns:
            data = df['temperature'].dropna().tolist()
        elif 'temp' in df.columns:
            data = df['temp'].dropna().tolist()
        else:
            # Use first numeric column
            numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
            if len(numeric_cols) > 0:
                data = df[numeric_cols[0]].dropna().tolist()
            else:
                raise ValueError("No suitable temperature column found")
    elif _HAVE_HF:
        # Try to load from HuggingFace datasets
        try:
            dataset = load_dataset("mstz/electricity_transformer_temperature", split="train")
            data = [float(x) for x in dataset['temperature'] if x is not None]
        except:
            raise RuntimeError("Electricity Transformer Temperature loader requires either Kaggle dataset or `datasets` library.")
    else:
        raise RuntimeError("Electricity Transformer Temperature loader requires either Kaggle dataset or `datasets` library.")

    # Normalize data
    data = _normalize_sequence(data)
    
    # Create sequences
    sequences, targets = _create_sequences(data, seq_len, step)
    
    train_ratio = 1 - val_split / len(sequences)
    train_idx, val_idx = split_indices(len(sequences), train_ratio=train_ratio, seed=42)
    train_sequences = [sequences[i] for i in train_idx]
    train_targets = [targets[i] for i in train_idx]
    val_sequences = [sequences[i] for i in val_idx]
    val_targets = [targets[i] for i in val_idx]

    # Convert to tensors
    train_seq_tensor, train_target_tensor = _sequences_to_tensor(train_sequences, train_targets)
    val_seq_tensor, val_target_tensor = _sequences_to_tensor(val_sequences, val_targets)

    return (
        TensorDataset(train_seq_tensor, train_target_tensor),
        TensorDataset(val_seq_tensor, val_target_tensor),
        1,  # Single feature (temperature)
    )


def load_jena_climate(
    seq_len: int = 128,
    val_split: int = 1000,
    step: int = 1,
    feature_col: str = "T (degC)",  # Temperature by default
    shuffle_train: bool = False,
    shuffle_eval: bool = False,
):
    """Jena Climate Archive prediction (regression).
    
    Predicts climate variables based on historical climate data.
    Default feature is temperature, but can be changed.
    """
    import pandas as pd
    
    # Try Kaggle path first
    kaggle_path = Path("/kaggle/input/jena-climate-archive/jena_climate_2009_2016.csv")
    if kaggle_path.exists():
        df = pd.read_csv(kaggle_path)
        if feature_col in df.columns:
            data = df[feature_col].dropna().tolist()
        else:
            # Use first numeric column if specified column not found
            numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
            if len(numeric_cols) > 0:
                data = df[numeric_cols[0]].dropna().tolist()
                print(f"Warning: {feature_col} not found, using {numeric_cols[0]} instead")
            else:
                raise ValueError(f"Column {feature_col} not found and no numeric columns available")
    elif _HAVE_HF:
        # Try to load from HuggingFace datasets
        try:
            dataset = load_dataset("mstz/jena_climate", split="train")
            if feature_col in dataset.column_names:
                data = [float(x) for x in dataset[feature_col] if x is not None]
            else:
                # Use first available column
                data = [float(x) for x in dataset[dataset.column_names[0]] if x is not None]
                print(f"Warning: {feature_col} not found, using {dataset.column_names[0]} instead")
        except:
            raise RuntimeError("Jena Climate loader requires either Kaggle dataset or `datasets` library.")
    else:
        raise RuntimeError("Jena Climate loader requires either Kaggle dataset or `datasets` library.")

    # Normalize data
    data = _normalize_sequence(data)
    
    # Create sequences
    sequences, targets = _create_sequences(data, seq_len, step)
    
    train_ratio = 1 - val_split / len(sequences)
    train_idx, val_idx = split_indices(len(sequences), train_ratio=train_ratio, seed=42)
    train_sequences = [sequences[i] for i in train_idx]
    train_targets = [targets[i] for i in train_idx]
    val_sequences = [sequences[i] for i in val_idx]
    val_targets = [targets[i] for i in val_idx]

    # Convert to tensors
    train_seq_tensor, train_target_tensor = _sequences_to_tensor(train_sequences, train_targets)
    val_seq_tensor, val_target_tensor = _sequences_to_tensor(val_sequences, val_targets)

    return (
        TensorDataset(train_seq_tensor, train_target_tensor),
        TensorDataset(val_seq_tensor, val_target_tensor),
        1,  # Single feature
    )


def load_jena_climate_multi(
    seq_len: int = 128,
    val_split: int = 1000,
    step: int = 1,
    feature_cols: List[str] = None,  # Use all numeric columns if None
    shuffle_train: bool = False,
    shuffle_eval: bool = False,
):
    """Jena Climate Archive multi-variable prediction (regression).
    
    Predicts multiple climate variables based on historical climate data.
    """
    import pandas as pd
    
    # Try Kaggle path first
    kaggle_path = Path("/kaggle/input/jena-climate-archive/jena_climate_2009_2016.csv")
    if kaggle_path.exists():
        df = pd.read_csv(kaggle_path)
        if feature_cols is None:
            # Use all numeric columns
            feature_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        
        # Validate columns exist
        missing_cols = [col for col in feature_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Columns not found: {missing_cols}")
        
        # Extract data for all features
        data_dict = {}
        for col in feature_cols:
            data_dict[col] = df[col].dropna().tolist()
        
        # Find minimum length to align all features
        min_length = min(len(data) for data in data_dict.values())
        for col in data_dict:
            data_dict[col] = data_dict[col][:min_length]
            
    elif _HAVE_HF:
        # Try to load from HuggingFace datasets
        try:
            dataset = load_dataset("mstz/jena_climate", split="train")
            if feature_cols is None:
                feature_cols = dataset.column_names
            
            # Validate columns exist
            missing_cols = [col for col in feature_cols if col not in dataset.column_names]
            if missing_cols:
                raise ValueError(f"Columns not found: {missing_cols}")
            
            # Extract data for all features
            data_dict = {}
            for col in feature_cols:
                data_dict[col] = [float(x) for x in dataset[col] if x is not None]
            
            # Find minimum length to align all features
            min_length = min(len(data) for data in data_dict.values())
            for col in data_dict:
                data_dict[col] = data_dict[col][:min_length]
                
        except:
            raise RuntimeError("Jena Climate loader requires either Kaggle dataset or `datasets` library.")
    else:
        raise RuntimeError("Jena Climate loader requires either Kaggle dataset or `datasets` library.")

    # Normalize each feature
    for col in data_dict:
        data_dict[col] = _normalize_sequence(data_dict[col])
    
    # Create multi-variable sequences
    sequences = []
    targets = []
    
    for i in range(0, min_length - seq_len, step):
        # Create sequence with all features
        seq = []
        for j in range(seq_len):
            time_step = []
            for col in feature_cols:
                time_step.append(data_dict[col][i + j])
            seq.append(time_step)
        
        # Target is the next time step for all features
        target = []
        for col in feature_cols:
            target.append(data_dict[col][i + seq_len] if i + seq_len < min_length else data_dict[col][-1])
        
        sequences.append(seq)
        targets.append(target)
    
    # Split train/val
    train_ratio = 1 - val_split / len(sequences)
    train_idx, val_idx = split_indices(len(sequences), train_ratio=train_ratio, seed=42)
    train_sequences = [sequences[i] for i in train_idx]
    train_targets = [targets[i] for i in train_idx]
    val_sequences = [sequences[i] for i in val_idx]
    val_targets = [targets[i] for i in val_idx]

    # Convert to tensors
    train_seq_tensor = torch.tensor(train_sequences, dtype=torch.float)
    train_target_tensor = torch.tensor(train_targets, dtype=torch.float)
    val_seq_tensor = torch.tensor(val_sequences, dtype=torch.float)
    val_target_tensor = torch.tensor(val_targets, dtype=torch.float)

    return (
        TensorDataset(train_seq_tensor, train_target_tensor),
        TensorDataset(val_seq_tensor, val_target_tensor),
        len(feature_cols),  # Number of features
    ) 