"""
Data utilities for TFN models.

Utilities for loading, preprocessing, and batching data for Token Field Networks.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Optional, Dict, Any
import numpy as np


class TFNDataset(Dataset):
    """Base dataset class for TFN models."""
    
    def __init__(self, data: List[Any], targets: List[Any], 
                 max_seq_len: int = 512, pad_token: int = 0):
        self.data = data
        self.targets = targets
        self.max_seq_len = max_seq_len
        self.pad_token = pad_token
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        raise NotImplementedError("Subclasses must implement __getitem__")


class TextClassificationDataset(TFNDataset):
    """Dataset for text classification tasks."""
    
    def __init__(self, texts: List[str], labels: List[int], 
                 vocab: Dict[str, int], max_seq_len: int = 512):
        super().__init__(texts, labels, max_seq_len)
        self.vocab = vocab
    
    def __getitem__(self, idx):
        text = self.data[idx]
        label = self.targets[idx]
        
        # Tokenize text
        tokens = text.split()[:self.max_seq_len]
        token_ids = [self.vocab.get(token, self.vocab.get('<UNK>', 1)) for token in tokens]
        
        # Pad sequence
        if len(token_ids) < self.max_seq_len:
            token_ids += [0] * (self.max_seq_len - len(token_ids))
        else:
            token_ids = token_ids[:self.max_seq_len]
        
        return {
            'input_ids': torch.tensor(token_ids, dtype=torch.long),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class SequenceRegressionDataset(TFNDataset):
    """Dataset for sequence regression tasks."""
    
    def __init__(self, sequences: List[np.ndarray], targets: List[np.ndarray], 
                 max_seq_len: int = 512):
        super().__init__(sequences, targets, max_seq_len)
    
    def __getitem__(self, idx):
        sequence = self.data[idx]
        target = self.targets[idx]
        
        # Pad or truncate sequence
        if len(sequence) < self.max_seq_len:
            # Pad with zeros
            padded = np.zeros((self.max_seq_len, sequence.shape[1]))
            padded[:len(sequence)] = sequence
        else:
            # Truncate
            padded = sequence[:self.max_seq_len]
        
        return {
            'features': torch.tensor(padded, dtype=torch.float32),
            'targets': torch.tensor(target, dtype=torch.float32)
        }


def create_vocab(texts: List[str], min_freq: int = 2) -> Dict[str, int]:
    """Create vocabulary from text data."""
    word_counts = {}
    
    # Count word frequencies
    for text in texts:
        for word in text.split():
            word_counts[word] = word_counts.get(word, 0) + 1
    
    # Filter by minimum frequency
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for word, count in word_counts.items():
        if count >= min_freq:
            vocab[word] = len(vocab)
    
    return vocab


def collate_fn_classification(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Collate function for classification tasks."""
    input_ids = torch.stack([item['input_ids'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])
    
    return {
        'input_ids': input_ids,
        'labels': labels
    }


def collate_fn_regression(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Collate function for regression tasks."""
    features = torch.stack([item['features'] for item in batch])
    targets = torch.stack([item['targets'] for item in batch])
    
    return {
        'features': features,
        'targets': targets
    }


def create_dataloader(dataset: Dataset, batch_size: int = 32, 
                     shuffle: bool = True, num_workers: int = 0,
                     collate_fn=None) -> DataLoader:
    """Create a DataLoader with appropriate collate function."""
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn
    )


def generate_positions(seq_len: int, batch_size: int = 1, 
                      device: torch.device = None) -> torch.Tensor:
    """Generate position embeddings for sequences."""
    positions = torch.linspace(0.1, 0.9, seq_len, device=device)
    positions = positions.unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, -1)
    return positions


def create_synthetic_data(num_samples: int = 1000, seq_len: int = 50, 
                         vocab_size: int = 1000, num_classes: int = 3) -> Tuple[List[str], List[int]]:
    """Create synthetic text classification data."""
    import random
    
    # Generate random "words"
    words = [f"word_{i}" for i in range(vocab_size)]
    
    texts = []
    labels = []
    
    for _ in range(num_samples):
        # Generate random sequence
        sentence_len = random.randint(10, seq_len)
        sentence = " ".join(random.choices(words, k=sentence_len))
        texts.append(sentence)
        
        # Generate random label
        label = random.randint(0, num_classes - 1)
        labels.append(label)
    
    return texts, labels


def create_synthetic_regression_data(num_samples: int = 1000, seq_len: int = 50, 
                                   feature_dim: int = 32, output_dim: int = 8) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Create synthetic regression data."""
    sequences = []
    targets = []
    
    for _ in range(num_samples):
        # Generate random sequence
        sequence = np.random.randn(seq_len, feature_dim)
        sequences.append(sequence)
        
        # Generate random target
        target = np.random.randn(seq_len, output_dim)
        targets.append(target)
    
    return sequences, targets
