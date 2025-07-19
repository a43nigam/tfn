from __future__ import annotations

"""tfn.datasets.long_text_loader
Flexible long-text dataset loader supporting multiple sources.
"""

import logging
import random
import re
from typing import List, Tuple, Dict, Optional, Iterator
from pathlib import Path

# Suppress verbose download bars
logging.getLogger("datasets").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
import os
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset

# Optional HuggingFace datasets
try:
    from datasets import load_dataset
    _HAVE_HF = True
except ImportError:
    _HAVE_HF = False


def generate_synthetic_long_text(num_samples: int = 100, avg_length: int = 5000) -> List[str]:
    """Generate synthetic long-form text for testing.
    
    Creates realistic-looking text with paragraphs, sentences, and vocabulary.
    """
    # Vocabulary for synthetic text
    words = [
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by",
        "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "do", "does", "did",
        "will", "would", "could", "should", "may", "might", "can", "must", "shall",
        "this", "that", "these", "those", "it", "its", "they", "them", "their", "we", "us", "our",
        "you", "your", "he", "him", "his", "she", "her", "hers", "I", "me", "my", "mine",
        "research", "study", "analysis", "method", "approach", "technique", "system", "process",
        "development", "implementation", "evaluation", "assessment", "comparison", "investigation",
        "examination", "exploration", "discovery", "innovation", "advancement", "improvement",
        "enhancement", "optimization", "refinement", "modification", "adaptation", "integration",
        "coordination", "collaboration", "communication", "interaction", "relationship", "connection",
        "structure", "framework", "architecture", "design", "pattern", "model", "theory", "concept",
        "principle", "foundation", "basis", "core", "central", "fundamental", "essential", "critical",
        "important", "significant", "relevant", "appropriate", "suitable", "effective", "efficient",
        "successful", "productive", "valuable", "useful", "helpful", "beneficial", "advantageous",
        "favorable", "positive", "constructive", "creative", "innovative", "original", "unique",
        "distinctive", "characteristic", "typical", "common", "usual", "normal", "standard",
        "conventional", "traditional", "established", "proven", "reliable", "dependable", "stable",
        "consistent", "uniform", "regular", "systematic", "organized", "structured", "planned",
        "prepared", "arranged", "configured", "set", "established", "created", "developed",
        "produced", "generated", "formed", "constructed", "built", "assembled", "composed",
        "organized", "arranged", "structured", "designed", "planned", "prepared", "configured"
    ]
    
    # Sentence patterns
    sentence_patterns = [
        "The {noun} {verb} {adverb} in the {location}.",
        "Research shows that {noun} {verb} {adverb}.",
        "This {noun} demonstrates {adjective} {noun} capabilities.",
        "The {adjective} {noun} provides {adjective} {noun}.",
        "Analysis reveals {adjective} {noun} patterns.",
        "Studies indicate {adjective} {noun} relationships.",
        "The {noun} approach enables {adjective} {noun}.",
        "Results show {adjective} {noun} improvements.",
        "The {noun} system supports {adjective} {noun}.",
        "Evaluation confirms {adjective} {noun} effectiveness."
    ]
    
    # Additional vocabulary for variety
    nouns = ["method", "approach", "technique", "system", "process", "analysis", "study", "research", "model", "framework"]
    verbs = ["demonstrates", "shows", "indicates", "reveals", "confirms", "supports", "enables", "provides", "achieves", "facilitates"]
    adjectives = ["effective", "efficient", "successful", "reliable", "robust", "flexible", "scalable", "innovative", "comprehensive", "advanced"]
    adverbs = ["effectively", "efficiently", "successfully", "reliably", "robustly", "flexibly", "scalably", "innovatively", "comprehensively", "advancedly"]
    locations = ["laboratory", "environment", "system", "framework", "context", "setting", "domain", "field", "area", "space"]
    
    def generate_paragraph():
        """Generate a realistic paragraph."""
        sentences = []
        for _ in range(random.randint(3, 8)):
            pattern = random.choice(sentence_patterns)
            sentence = pattern.format(
                noun=random.choice(nouns),
                verb=random.choice(verbs),
                adjective=random.choice(adjectives),
                adverb=random.choice(adverbs),
                location=random.choice(locations)
            )
            sentences.append(sentence)
        return " ".join(sentences)
    
    texts = []
    for _ in range(num_samples):
        # Generate multiple paragraphs
        paragraphs = []
        target_length = random.randint(avg_length // 2, avg_length * 2)
        current_length = 0
        
        while current_length < target_length:
            paragraph = generate_paragraph()
            paragraphs.append(paragraph)
            current_length += len(paragraph)
        
        text = "\n\n".join(paragraphs)
        texts.append(text)
    
    return texts


class SyntheticLongTextDataset(Dataset):
    """Dataset for synthetic long-form text."""
    
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
            # Simple word-level tokenization
            words = re.findall(r'\b\w+\b', text.lower())
            tokens = [self.word2idx.get(word, 1) for word in words]  # 1 = <UNK>
            
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


def create_long_text_dataloader(
    dataset_type: str = "synthetic",
    seq_len: int = 4096,
    batch_size: int = 4,
    vocab_size: int = 10000,
    chunk_overlap: int = 0,
    max_train_chunks: Optional[int] = None,
    max_val_chunks: Optional[int] = None,
    val_split_ratio: float = 0.1,
    add_special_tokens: bool = True,
    shuffle_chunks: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True
) -> Tuple[DataLoader, DataLoader, int]:
    """Create long-text dataloaders for testing.
    
    Args:
        dataset_type: "synthetic", "wikitext", "bookcorpus", or "openwebtext"
        seq_len: Length of each sequence chunk
        batch_size: Batch size for training
        vocab_size: Size of vocabulary
        chunk_overlap: Overlap between consecutive chunks
        max_train_chunks: Maximum number of training chunks
        max_val_chunks: Maximum number of validation chunks
        val_split_ratio: Ratio of data to use for validation
        add_special_tokens: Whether to add BOS/EOS tokens
        shuffle_chunks: Whether to shuffle chunks
        num_workers: Number of workers for data loading
        pin_memory: Whether to pin memory
    
    Returns:
        (train_loader, val_loader, vocab_size)
    """
    
    if dataset_type == "synthetic":
        # Generate synthetic text
        print("Generating synthetic long-text dataset...")
        texts = generate_synthetic_long_text(num_samples=50, avg_length=8000)
        
        # Build vocabulary
        word2idx = {
            "<PAD>": 0, 
            "<UNK>": 1, 
            "<BOS>": 2, 
            "<EOS>": 3,
            "<MASK>": 4
        }
        
        # Add words from texts
        all_words = []
        for text in texts:
            words = re.findall(r'\b\w+\b', text.lower())
            all_words.extend(words)
        
        # Add most common words to vocab
        from collections import Counter
        word_counts = Counter(all_words)
        for word, _ in word_counts.most_common(vocab_size - len(word2idx)):
            word2idx[word] = len(word2idx)
        
        vocab_size = len(word2idx)
        
        # Create datasets
        train_size = int(len(texts) * (1 - val_split_ratio))
        train_texts = texts[:train_size]
        val_texts = texts[train_size:]
        
        train_dataset = SyntheticLongTextDataset(
            texts=train_texts,
            word2idx=word2idx,
            seq_len=seq_len,
            chunk_overlap=chunk_overlap,
            add_special_tokens=add_special_tokens,
            shuffle_chunks=shuffle_chunks
        )
        
        val_dataset = SyntheticLongTextDataset(
            texts=val_texts,
            word2idx=word2idx,
            seq_len=seq_len,
            chunk_overlap=chunk_overlap,
            add_special_tokens=add_special_tokens,
            shuffle_chunks=False
        )
        
    elif dataset_type in ["wikitext", "bookcorpus", "openwebtext"] and _HAVE_HF:
        # Use HuggingFace datasets
        print(f"Loading {dataset_type} dataset...")
        
        if dataset_type == "wikitext":
            dataset = load_dataset("wikitext", "wikitext-103-v1", split="train")
        elif dataset_type == "bookcorpus":
            dataset = load_dataset("bookcorpus", split="train")
        else:  # openwebtext
            dataset = load_dataset("openwebtext", split="train")
        
        # Extract texts
        texts = [ex["text"] for ex in dataset]
        
        # Build vocabulary (simplified)
        word2idx = {
            "<PAD>": 0, 
            "<UNK>": 1, 
            "<BOS>": 2, 
            "<EOS>": 3,
            "<MASK>": 4
        }
        
        # Add words (simplified vocabulary building)
        all_words = []
        for text in texts[:100]:  # Use subset for vocab building
            words = re.findall(r'\b\w+\b', text.lower())
            all_words.extend(words)
        
        from collections import Counter
        word_counts = Counter(all_words)
        for word, _ in word_counts.most_common(vocab_size - len(word2idx)):
            word2idx[word] = len(word2idx)
        
        vocab_size = len(word2idx)
        
        # Create datasets
        train_size = int(len(texts) * (1 - val_split_ratio))
        train_texts = texts[:train_size]
        val_texts = texts[train_size:]
        
        train_dataset = SyntheticLongTextDataset(
            texts=train_texts,
            word2idx=word2idx,
            seq_len=seq_len,
            chunk_overlap=chunk_overlap,
            add_special_tokens=add_special_tokens,
            shuffle_chunks=shuffle_chunks
        )
        
        val_dataset = SyntheticLongTextDataset(
            texts=val_texts,
            word2idx=word2idx,
            seq_len=seq_len,
            chunk_overlap=chunk_overlap,
            add_special_tokens=add_special_tokens,
            shuffle_chunks=False
        )
        
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
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


if __name__ == "__main__":
    # Test the loader
    train_loader, val_loader, vocab_size = create_long_text_dataloader(
        dataset_type="synthetic",
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