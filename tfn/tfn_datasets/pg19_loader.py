from __future__ import annotations

"""tfn.datasets.pg19_loader
PG-19 language modeling dataset loader for long-sequence testing.
Supports streaming chunks and efficient tokenization for sequences up to 8K tokens.
"""

from typing import List, Tuple, Dict, Optional, Iterator
import random
from pathlib import Path
import logging

# Suppress verbose download bars from HuggingFace datasets
logging.getLogger("datasets").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)

# Suppress tqdm progress bars
import os
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import IterableDataset

# Optional dependency ---------------------------------------------------------
try:
    from datasets import load_dataset, Dataset as HFDataset  # type: ignore
    _HAVE_HF = True
except ImportError:
    _HAVE_HF = False

# ---------------------------------------------------------------------------
# Tokenization helpers for PG-19
# ---------------------------------------------------------------------------

def _build_pg19_vocab(texts: List[str], vocab_size: int = 50000) -> Dict[str, int]:
    """Build vocabulary from PG-19 texts with larger vocab for long sequences."""
    from collections import Counter
    import re
    
    # Simple word-level tokenization for PG-19
    def tokenize(text: str) -> List[str]:
        # Split on whitespace and punctuation
        return re.findall(r"\b\w+\b", text.lower())
    
    counter = Counter()
    for text in texts:
        counter.update(tokenize(text))
    
    # Build vocab with special tokens
    vocab = {
        "<PAD>": 0, 
        "<UNK>": 1, 
        "<BOS>": 2, 
        "<EOS>": 3,
        "<MASK>": 4
    }
    
    # Add most common words
    for word, _ in counter.most_common(vocab_size - len(vocab)):
        vocab[word] = len(vocab)
    
    return vocab


def _tokenize_pg19_text(
    text: str, 
    word2idx: Dict[str, int], 
    seq_len: int = 4096,
    add_special_tokens: bool = True
) -> List[int]:
    """Tokenize PG-19 text with optional special tokens."""
    import re
    
    # Simple word-level tokenization
    tokens = re.findall(r"\b\w+\b", text.lower())
    
    # Convert to indices
    ids = [word2idx.get(tok, 1) for tok in tokens]  # 1 = <UNK>
    
    # Add special tokens if requested
    if add_special_tokens:
        ids = [word2idx["<BOS>"]] + ids + [word2idx["<EOS>"]]
    
    # Truncate or pad to seq_len
    if len(ids) > seq_len:
        ids = ids[:seq_len]
    else:
        ids += [word2idx["<PAD>"]] * (seq_len - len(ids))
    
    return ids


class PG19ChunkDataset(Dataset):
    """Dataset for PG-19 with fixed-length chunks."""
    
    def __init__(
        self,
        texts: List[str],
        word2idx: Dict[str, int],
        seq_len: int = 4096,
        chunk_overlap: int = 0,
        add_special_tokens: bool = True,
        shuffle_chunks: bool = False
    ):
        self.word2idx = word2idx
        self.seq_len = seq_len
        self.chunk_overlap = chunk_overlap
        self.add_special_tokens = add_special_tokens
        self.shuffle_chunks = shuffle_chunks
        
        # Create chunks from texts
        self.chunks = self._create_chunks(texts)
        
        if shuffle_chunks:
            random.shuffle(self.chunks)
    
    def _create_chunks(self, texts: List[str]) -> List[List[int]]:
        """Create overlapping chunks from texts."""
        chunks = []
        
        for text in texts:
            # Tokenize full text
            tokens = _tokenize_pg19_text(
                text, self.word2idx, 
                seq_len=len(text),  # Don't truncate here
                add_special_tokens=False
            )
            
            # Create overlapping chunks
            step = self.seq_len - self.chunk_overlap
            for i in range(0, len(tokens) - self.seq_len + 1, step):
                chunk = tokens[i:i + self.seq_len]
                if self.add_special_tokens:
                    chunk = [self.word2idx["<BOS>"]] + chunk[:-1] + [self.word2idx["<EOS>"]]
                chunks.append(chunk)
        
        return chunks
    
    def __len__(self) -> int:
        return len(self.chunks)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        return torch.tensor(self.chunks[idx], dtype=torch.long)


class PG19StreamingDataset(IterableDataset):
    """Streaming dataset for PG-19 that yields chunks on-the-fly."""
    
    def __init__(
        self,
        word2idx: Dict[str, int],
        dataset_name: str = "pg19",
        split: str = "train",
        seq_len: int = 4096,
        chunk_overlap: int = 0,
        add_special_tokens: bool = True,
        max_chunks_per_epoch: Optional[int] = None
    ):
        if not _HAVE_HF:
            raise RuntimeError("PG19StreamingDataset requires `datasets` library.")
        
        self.dataset_name = dataset_name
        self.split = split
        self.word2idx = word2idx
        self.seq_len = seq_len
        self.chunk_overlap = chunk_overlap
        self.add_special_tokens = add_special_tokens
        self.max_chunks_per_epoch = max_chunks_per_epoch
    
    def __iter__(self) -> Iterator[torch.Tensor]:
        """Stream chunks from PG-19 dataset."""
        # Load dataset in streaming mode
        dataset = load_dataset(
            self.dataset_name, 
            split=self.split, 
            streaming=True,
            trust_remote_code=True
        )
        
        chunk_count = 0
        
        for example in dataset:
            text = example["text"]
            
            # Tokenize text
            tokens = _tokenize_pg19_text(
                text, self.word2idx,
                seq_len=len(text),  # Don't truncate here
                add_special_tokens=False
            )
            
            # Create overlapping chunks
            step = self.seq_len - self.chunk_overlap
            for i in range(0, len(tokens) - self.seq_len + 1, step):
                chunk = tokens[i:i + self.seq_len]
                if self.add_special_tokens:
                    chunk = [self.word2idx["<BOS>"]] + chunk[:-1] + [self.word2idx["<EOS>"]]
                
                yield torch.tensor(chunk, dtype=torch.long)
                chunk_count += 1
                
                if self.max_chunks_per_epoch and chunk_count >= self.max_chunks_per_epoch:
                    return


# ---------------------------------------------------------------------------
# Public loaders
# ---------------------------------------------------------------------------

def load_pg19_chunked(
    seq_len: int = 4096,
    vocab_size: int = 50000,
    chunk_overlap: int = 0,
    max_train_chunks: Optional[int] = None,
    max_val_chunks: Optional[int] = None,
    val_split_ratio: float = 0.1,
    add_special_tokens: bool = True,
    shuffle_chunks: bool = True,
    streaming: bool = False
) -> Tuple[Dataset, Dataset, int]:
    """Load PG-19 dataset with chunking for long-sequence language modeling.
    
    Args:
        seq_len: Length of each sequence chunk
        vocab_size: Size of vocabulary
        chunk_overlap: Overlap between consecutive chunks
        max_train_chunks: Maximum number of training chunks (None = all)
        max_val_chunks: Maximum number of validation chunks (None = all)
        val_split_ratio: Ratio of data to use for validation
        add_special_tokens: Whether to add BOS/EOS tokens
        shuffle_chunks: Whether to shuffle chunks
        streaming: Whether to use streaming dataset
    
    Returns:
        (train_dataset, val_dataset, vocab_size)
    """
    
    if not _HAVE_HF:
        raise RuntimeError("PG-19 loader requires `datasets` library.")
    
    # Load PG-19 dataset
    dataset = load_dataset("pg19", split="train", trust_remote_code=True)
    
    # Extract texts
    texts = [ex["text"] for ex in dataset]
    
    # Build vocabulary from training texts
    train_size = int(len(texts) * (1 - val_split_ratio))
    train_texts = texts[:train_size]
    val_texts = texts[train_size:]
    
    word2idx = _build_pg19_vocab(train_texts, vocab_size)
    
    if streaming:
        # Use streaming datasets
        train_dataset = PG19StreamingDataset(
            dataset_name="pg19",
            split="train",
            word2idx=word2idx,
            seq_len=seq_len,
            chunk_overlap=chunk_overlap,
            add_special_tokens=add_special_tokens,
            max_chunks_per_epoch=max_train_chunks
        )
        
        val_dataset = PG19StreamingDataset(
            dataset_name="pg19",
            split="train",  # We'll filter in the iterator
            word2idx=word2idx,
            seq_len=seq_len,
            chunk_overlap=chunk_overlap,
            add_special_tokens=add_special_tokens,
            max_chunks_per_epoch=max_val_chunks
        )
    else:
        # Use chunked datasets
        train_dataset = PG19ChunkDataset(
            texts=train_texts,
            word2idx=word2idx,
            seq_len=seq_len,
            chunk_overlap=chunk_overlap,
            add_special_tokens=add_special_tokens,
            shuffle_chunks=shuffle_chunks
        )
        
        val_dataset = PG19ChunkDataset(
            texts=val_texts,
            word2idx=word2idx,
            seq_len=seq_len,
            chunk_overlap=chunk_overlap,
            add_special_tokens=add_special_tokens,
            shuffle_chunks=False  # Don't shuffle validation
        )
    
    return train_dataset, val_dataset, len(word2idx)


def create_pg19_dataloader(
    seq_len: int = 4096,
    batch_size: int = 4,
    vocab_size: int = 50000,
    chunk_overlap: int = 0,
    max_train_chunks: Optional[int] = None,
    max_val_chunks: Optional[int] = None,
    val_split_ratio: float = 0.1,
    add_special_tokens: bool = True,
    shuffle_chunks: bool = True,
    streaming: bool = False,
    num_workers: int = 4,
    pin_memory: bool = True
) -> Tuple[DataLoader, DataLoader, int]:
    """Create PG-19 dataloaders for training.
    
    Returns:
        (train_loader, val_loader, vocab_size)
    """
    
    train_dataset, val_dataset, vocab_size = load_pg19_chunked(
        seq_len=seq_len,
        vocab_size=vocab_size,
        chunk_overlap=chunk_overlap,
        max_train_chunks=max_train_chunks,
        max_val_chunks=max_val_chunks,
        val_split_ratio=val_split_ratio,
        add_special_tokens=add_special_tokens,
        shuffle_chunks=shuffle_chunks,
        streaming=streaming
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=not streaming,  # Don't shuffle streaming datasets
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )
    
    return train_loader, val_loader, vocab_size


# ---------------------------------------------------------------------------
# Utility functions for language modeling
# ---------------------------------------------------------------------------

def compute_perplexity(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device
) -> float:
    """Compute perplexity on a dataset."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    
    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch, torch.Tensor):
                input_ids = batch.to(device)
            else:
                input_ids = batch[0].to(device)
            
            # Shift for language modeling
            input_ids = input_ids[:, :-1]
            target_ids = batch[:, 1:]
            
            # Forward pass
            outputs = model(input_ids)
            
            # Compute loss
            loss = torch.nn.functional.cross_entropy(
                outputs.view(-1, outputs.size(-1)),
                target_ids.view(-1),
                ignore_index=0  # Ignore padding
            )
            
            # Count non-padding tokens
            non_pad_mask = target_ids != 0
            num_tokens = non_pad_mask.sum().item()
            
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens
    
    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    
    return perplexity


def measure_memory_usage(
    model: torch.nn.Module,
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    device: torch.device
) -> Dict[str, float]:
    """Measure memory usage of model."""
    import psutil
    import os
    
    # Get initial memory
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    # Create dummy input
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len)).to(device)
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids)
    
    # Get peak memory
    peak_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    # Get GPU memory if available
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB
    else:
        gpu_memory = 0.0
    
    return {
        "cpu_memory_mb": peak_memory - initial_memory,
        "gpu_memory_mb": gpu_memory,
        "total_memory_mb": peak_memory
    }


if __name__ == "__main__":
    # Test the PG-19 loader
    try:
        train_loader, val_loader, vocab_size = create_pg19_dataloader(
            seq_len=1024,
            batch_size=2,
            max_train_chunks=10,
            max_val_chunks=5
        )
        
        print(f"Vocabulary size: {vocab_size}")
        print(f"Train batches: {len(train_loader)}")
        print(f"Val batches: {len(val_loader)}")
        
        # Test a batch
        for batch in train_loader:
            print(f"Batch shape: {batch.shape}")
            print(f"Batch dtype: {batch.dtype}")
            break
            
    except Exception as e:
        print(f"Error testing PG-19 loader: {e}") 