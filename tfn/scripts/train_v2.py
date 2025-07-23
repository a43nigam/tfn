"""tfn.scripts.train_v2
A new, simplified training script for all 1-D TFN and baseline models.

This script serves as a replacement for the original `train.py`, featuring a
more robust and maintainable command-line interface and data loading pipeline.

Key Design Improvements:
-------------------------
1.  **Simplified CLI**: Instead of dynamically building the CLI by introspecting
    function signatures, this script uses a fixed set of core arguments.
    Model-specific and dataset-specific hyperparameters are passed via
    structured JSON strings (`--model_kwargs` and `--dataset_kwargs`). This
    is more robust, explicit, and easier to debug.

2.  **Decoupled Data Logic**: The script relies on the `tfn.tfn_datasets.registry`
    to return a standardized `(train_ds, val_ds, meta_dict)` tuple. The
    `meta_dict` (containing info like `vocab_size`, `num_classes`) is then
    programmatically passed to the model, decoupling the data loading from
    the model instantiation.

3.  **Clear Configuration Flow**: The final model configuration is built by
    layering defaults, dataset metadata, and user-provided CLI arguments,
    making the process transparent and reproducible.

Usage Example (CLI):
--------------------
# Train a TFN classifier on the SST-2 dataset, overriding default layers and kernel.
python -m tfn.scripts.train_v2 \\
    --task classification \\
    --dataset sst2 \\
    --model tfn_classifier \\
    --epochs 5 \\
    --model_kwargs '{"num_layers": 4, "kernel_type": "compact"}'

# Train a Transformer baseline, overriding the dataset's sequence length.
python -m tfn.scripts.train_v2 \\
    --task classification \\
    --dataset sst2 \\
    --model transformer_classifier \\
    --epochs 5 \\
    --dataset_kwargs '{"seq_len": 256}'
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn as nn

# TFN Project Imports
from tfn.model.registry import get_model_config, validate_model_task_compatibility
from tfn.tfn_datasets.registry import get_dataset
from tfn.utils.metrics import accuracy


def _build_parser() -> argparse.ArgumentParser:
    """Builds the argument parser with a simplified, explicit interface."""
    parser = argparse.ArgumentParser(
        description="TFN Training Script (v2)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Core arguments
    parser.add_argument("--task", type=str, required=True,
                        choices=["classification", "regression", "time_series", "language_modeling", "ner"],
                        help="The training task.")
    parser.add_argument("--dataset", type=str, required=True, help="The dataset to use (e.g., 'sst2', 'agnews').")
    parser.add_argument("--model", type=str, required=True, help="The model to train (e.g., 'tfn_classifier').")

    # Training loop arguments
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training and validation.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for the optimizer.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for the AdamW optimizer.")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping value.")
    parser.add_argument("--warmup_ratio", type=float, default=0.1,
                        help="Ratio of total steps for learning rate warmup.")

    # Hardware and reproducibility
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use for training.")
    parser.add_argument("--num_workers", type=int, default=2, help="Number of workers for DataLoader.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")

    # Hyperparameters (passed as JSON strings)
    parser.add_argument("--model_kwargs", type=str, default="{}",
                        help='JSON string of keyword arguments for the model constructor (e.g., \'{"num_layers": 4}\').')
    parser.add_argument("--dataset_kwargs", type=str, default="{}",
                        help='JSON string of keyword arguments for the dataset loader (e.g., \'{"seq_len": 256}\').')

    # Output and logging
    parser.add_argument("--output_dir", type=str, default="outputs", help="Directory to save training artifacts.")
    parser.add_argument("--run_name", type=str, default=None,
                        help="A specific name for this run. If None, it's auto-generated.")

    return parser


def train_and_evaluate(
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        args: argparse.Namespace,
) -> Dict[str, Any]:
    """
    Trains and evaluates the given model, returning a history of metrics.
    Uses AdamW optimizer and a linear warmup with cosine decay LR schedule.
    """
    model.to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Learning rate scheduler
    total_steps = args.epochs * len(train_loader)
    warmup_steps = int(args.warmup_ratio * total_steps)

    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Loss function
    criterion = nn.CrossEntropyLoss() if args.task == "classification" else nn.MSELoss()

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_loss = float("inf")
    best_model_state = None

    print("\n--- Starting Training ---")
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss, total_correct, total_samples = 0.0, 0.0, 0
        start_time = time.time()

        for batch in train_loader:
            optimizer.zero_grad()
            inputs, targets = batch[0].to(args.device), batch[1].to(args.device)

            logits = model(inputs)
            loss = criterion(logits, targets)
            loss.backward()

            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            optimizer.step()
            scheduler.step()

            train_loss += loss.item()
            if args.task == "classification":
                total_correct += accuracy(logits, targets) * targets.size(0)
                total_samples += targets.size(0)

        epoch_train_loss = train_loss / len(train_loader)
        epoch_train_acc = total_correct / total_samples if total_samples > 0 else 0.0
        history["train_loss"].append(epoch_train_loss)
        history["train_acc"].append(epoch_train_acc)

        # Validation loop
        model.eval()
        val_loss, total_val_correct, total_val_samples = 0.0, 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                inputs, targets = batch[0].to(args.device), batch[1].to(args.device)
                logits = model(inputs)
                val_loss += criterion(logits, targets).item()
                if args.task == "classification":
                    total_val_correct += accuracy(logits, targets) * targets.size(0)
                    total_val_samples += targets.size(0)

        epoch_val_loss = val_loss / len(val_loader)
        epoch_val_acc = total_val_correct / total_val_samples if total_val_samples > 0 else 0.0
        history["val_loss"].append(epoch_val_loss)
        history["val_acc"].append(epoch_val_acc)

        epoch_time = time.time() - start_time
        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"Time: {epoch_time:.2f}s | "
            f"Train Loss: {epoch_train_loss:.4f} | "
            f"Val Loss: {epoch_val_loss:.4f} | "
            f"Val Acc: {epoch_val_acc:.4f}"
        )

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_model_state = model.state_dict()

    print("--- Training Finished ---")
    return {"history": history, "best_model_state": best_model_state}


def main():
    """Main entry point for the training script."""
    parser = _build_parser()
    args = parser.parse_args()

    # --- 1. Setup and Configuration ---
    torch.manual_seed(args.seed)
    run_name = args.run_name or f"{args.dataset}_{args.model}_{int(time.time())}"
    output_dir = Path(args.output_dir) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"--- Configuration for run: {run_name} ---")
    print(f"Task: {args.task}, Model: {args.model}, Dataset: {args.dataset}")
    print(f"Device: {args.device}, Epochs: {args.epochs}, Batch Size: {args.batch_size}")
    print(f"Output directory: {output_dir}")

    # --- 2. Load Dataset ---
    print("\n--- Loading Dataset ---")
    try:
        dataset_kwargs = json.loads(args.dataset_kwargs)
        train_ds, val_ds, meta_dict = get_dataset(args.dataset, **dataset_kwargs)
        print(f"Dataset '{args.dataset}' loaded successfully.")
        print(f"  Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")
        print(f"  Dataset metadata: {meta_dict}")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.num_workers)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=args.batch_size, num_workers=args.num_workers)

    # --- 3. Instantiate Model ---
    print("\n--- Instantiating Model ---")
    try:
        model_config = get_model_config(args.model)

        # Build model kwargs: defaults < dataset_meta < cli_kwargs
        model_kwargs = model_config.get("defaults", {}).copy()
        model_kwargs.update(meta_dict)
        cli_model_kwargs = json.loads(args.model_kwargs)
        model_kwargs.update(cli_model_kwargs)

        # Ensure the 'task' argument is passed to models that need it (like UnifiedTFN)
        if "task" in model_config.get("required_params", []) or "task" in model_config.get("optional_params", []):
            model_kwargs["task"] = args.task

        # Remove any None values that might have been added
        model_kwargs = {k: v for k, v in model_kwargs.items() if v is not None}

        model = model_config
        "class"
        print(f"Model '{args.model}' instantiated successfully.")
        print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"  Final model kwargs: {model_kwargs}")

    except Exception as e:
        print(f"Error instantiating model: {e}")
        return

    # --- 4. Train and Evaluate ---
    results = train_and_evaluate(model, train_loader, val_loader, args)

    # --- 5. Save Artifacts ---
    print("\n--- Saving Artifacts ---")
    # Save final configuration
    final_config = {
        "args": vars(args),
        "model_kwargs": model_kwargs,
        "dataset_kwargs": dataset_kwargs,
        "meta_dict": meta_dict,
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(final_config, f, indent=2)
    print(f"  Saved final configuration to {output_dir / 'config.json'}")

    # Save training history
    with open(output_dir / "history.json", "w") as f:
        json.dump(results["history"], f, indent=2)
    print(f"  Saved training history to {output_dir / 'history.json'}")

    # Save best model state
    if results["best_model_state"]:
        torch.save(results["best_model_state"], output_dir / "best_model.pt")
        print(f"  Saved best model state to {output_dir / 'best_model.pt'}")

    print("\n--- Run Finished ---")


if __name__ == "__main__":
    main()
