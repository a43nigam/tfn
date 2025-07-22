#!/usr/bin/env python3
"""
PG-19 Long-Sequence Efficiency Benchmark (Phase-2)

Runs TFN-1D and Performer baselines at 4 096 / 8 192 tokens and
ablates grid size for TFN. Measures:
    • Wall-time throughput (tokens/s)
    • Peak VRAM / CPU RAM
    • Perplexity after one quick pass (no full training)

Invocation example:
    python test_pg19_efficiency_cli.py \
        --seq-lens 4096 8192 \
        --grid-sizes 32 64 128 256 512 \
        --batch-size 2 --model-types tfn performer

Outputs JSON summary to pg19_efficiency_results.json by default so it
slots neatly into the existing test-result workflow.
"""
from __future__ import annotations

import argparse
import json
import time
import sys
import os
import logging
from pathlib import Path
from typing import Dict, List

# Suppress verbose download bars from HuggingFace datasets
logging.getLogger("datasets").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Setup paths for Kaggle compatibility
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

import torch
import torch.nn as nn

# Local imports -------------------------------------------------------------
try:
    from tfn.tfn_datasets.pg19_loader import (
        create_pg19_dataloader,
        compute_perplexity,
        measure_memory_usage,
    )
    from tfn.model.tfn_base import TrainableTFNLayer
    from tfn.model.seq_baselines import (
        SimplePerformerSeqModel,
    )
    from tfn.core.grid_utils import compute_auto_grid_size
except ImportError as e:
    print(f"Import error: {e}")
    print("Current sys.path:", sys.path)
    print("Current directory:", os.getcwd())
    raise

# ---------------------------------------------------------------------------
# Helper: create tiny language model shells (TFN or Performer)
# ---------------------------------------------------------------------------

def build_model(
    model_type: str,
    vocab_size: int,
    seq_len: int,
    embed_dim: int,
    grid_size: int | None,
    num_layers: int,
    device: torch.device,
) -> nn.Module:
    if model_type == "tfn":
        if grid_size is None:
            grid_size, _ = compute_auto_grid_size(seq_len, embed_dim)
        model = nn.Sequential(
            nn.Embedding(vocab_size, embed_dim),
            TrainableTFNLayer(
                embed_dim=embed_dim,
                grid_size=grid_size,
                kernel_type="rbf",
                evolution_type="cnn",
                time_steps=3,
                max_seq_len=seq_len,
            ),
            nn.Linear(embed_dim, vocab_size),
        )
    elif model_type == "performer":
        model = SimplePerformerSeqModel(
            vocab_size=vocab_size,
            seq_len=seq_len,
            embed_dim=embed_dim,
            num_layers=num_layers,
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    return model.to(device)


# ---------------------------------------------------------------------------
# Benchmark procedure
# ---------------------------------------------------------------------------

def run_benchmark(
    model_type: str,
    seq_len: int,
    batch_size: int,
    embed_dim: int,
    grid_size: int | None,
    num_layers: int,
    device: torch.device,
    val_chunks: int,
) -> Dict[str, float]:
    # Dataloader (small subset just for perplexity estimation)
    train_loader, val_loader, vocab_size = create_pg19_dataloader(
        seq_len=seq_len,
        batch_size=batch_size,
        max_train_chunks=val_chunks,
        max_val_chunks=val_chunks,
    )

    model = build_model(
        model_type,
        vocab_size,
        seq_len,
        embed_dim,
        grid_size,
        num_layers,
        device,
    )

    # Performance & memory --------------------------------------------------
    perf_metrics = measure_memory_usage(
        model, batch_size, seq_len, vocab_size, device
    )

    # Wall-time throughput (10 warm-up, 20 timed)
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len)).to(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None

    # Warm-up
    model.eval()
    with torch.no_grad():
        for _ in range(10):
            _ = model(input_ids)

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start = time.time()
    with torch.no_grad():
        for _ in range(20):
            _ = model(input_ids)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    wall_time = (time.time() - start) / 20
    tokens_per_sec = (batch_size * seq_len) / wall_time

    # Perplexity (quick estimate on val set)
    ppl = compute_perplexity(model, val_loader, device)

    num_params = sum(p.numel() for p in model.parameters())

    return {
        "tokens_per_second": tokens_per_sec,
        "avg_wall_time_per_batch": wall_time,
        "perplexity": ppl,
        "num_params": num_params,
        **perf_metrics,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="PG-19 Efficiency Benchmark")
    parser.add_argument("--model-types", nargs="*", default=["tfn", "performer"],
                        choices=["tfn", "performer"], help="Models to test")
    parser.add_argument("--seq-lens", nargs="*", type=int, default=[4096, 8192],
                        help="Sequence lengths to benchmark")
    parser.add_argument("--grid-sizes", nargs="*", type=int, default=[32, 64, 128, 256, 512],
                        help="Grid sizes (TFN only; use 0 or -1 for auto)")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--val-chunks", type=int, default=50,
                        help="Number of validation chunks for quick PPL calc")
    parser.add_argument("--output", type=str, default="pg19_efficiency_results.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    results: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = {}

    for seq_len in args.seq_lens:
        results[str(seq_len)] = {}
        for model_type in args.model_types:
            grid_iter: List[int | None]
            if model_type == "tfn":
                grid_iter = [None if g in (0, -1) else g for g in args.grid_sizes]
            else:
                grid_iter = [None]  # performer ignores grid size

            for gs in grid_iter:
                tag = f"{model_type}-g{gs if gs is not None else 'auto'}"
                print(f"\n>>> Benchmarking {tag} @ seq_len {seq_len}")
                res = run_benchmark(
                    model_type,
                    seq_len,
                    args.batch_size,
                    args.embed_dim,
                    gs,
                    args.num_layers,
                    device,
                    args.val_chunks,
                )
                results[str(seq_len)][tag] = res
                print(f"  tokens/s: {res['tokens_per_second']:.0f}, PPL: {res['perplexity']:.2f}, "
                      f"VRAM: {res['gpu_memory_mb']:.1f} MB, CPU Δ: {res['cpu_memory_mb']:.1f} MB")

    # Save results ----------------------------------------------------------
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main() 