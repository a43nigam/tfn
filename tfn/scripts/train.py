"""tfn.scripts.train
Unified entry-point for **all** 1-D TFN and baseline model training.

Usage (CLI)
------------
$ python -m tfn.scripts.train --task classification --dataset agnews --model tfn \
       --epochs 10 --batch_size 32

Programmatic API
----------------
from tfn.scripts.train import train_and_evaluate
metrics = train_and_evaluate(model, task, train_loader, val_loader, epochs=5)
"""
from __future__ import annotations

import sys
import argparse
from types import SimpleNamespace
from pathlib import Path
from typing import Dict, Any, Union
import math

import torch
import inspect
from tfn.utils.metrics import accuracy

# Import dynamic registries -----------------------------------------------------
from tfn.model.registry import (
    get_model_config,
    validate_model_task_compatibility,
    get_required_params,
    get_optional_params,
)
from tfn.tfn_datasets.registry import (
    list_datasets as _list_datasets,
    get_dataset as _get_dataset,
)

# Simple helper to check dataset key exists
def _dataset_available(name: str) -> bool:
    return name in _list_datasets()

# Placeholder compat – remove once legacy utilities are cleaned
def validate_dataset_task_compatibility(dataset: str, task: str) -> bool:
    # Currently no strict mapping – assume compatible
    return _dataset_available(dataset)

# -----------------------------------------------------------------------------
# Enhanced training util -------------------------------------------------------
# -----------------------------------------------------------------------------


def train_and_evaluate(
    model: torch.nn.Module,
    task: str,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    *,
    epochs: int = 5,
    device: str | torch.device = "cpu",
    lr: float = 1e-3,
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.1,
    grad_clip: float = 1.0,
) -> Dict[str, Any]:
    """Train *model* and return a metrics dictionary.

    Key improvements over the earlier simplistic loop:

    1. **AdamW** optimiser for proper decoupled weight-decay.
    2. **Warm-up + Cosine decay** learning-rate schedule (`WarmCosineScheduler`).
    3. **Gradient clipping** to avoid exploding gradients.

    All hyper-parameters retain sensible defaults so existing code paths/CLI
    invocations will continue to work unchanged.
    """

    model.to(device)

    # AdamW optimiser ---------------------------------------------------------
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # ---------------------------------------------------------------------
    # Warm Cosine LR Scheduler (linear warm-up → cosine decay)
    # ---------------------------------------------------------------------

    total_steps = epochs * max(1, len(train_loader))
    warmup_steps = int(warmup_ratio * total_steps)

    def _lr_lambda(current_step: int):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))  # cosine decay  → 0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)

    if task in {"classification", "ner"}:
        criterion = torch.nn.CrossEntropyLoss()
    else:
        criterion = torch.nn.MSELoss()

    history: Dict[str, list[float]] = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(1, epochs + 1):
        # Print current learning rate at the start of the epoch
        current_lr = optimizer.param_groups[0]['lr']
        print(f"--- Epoch {epoch}/{epochs} | Learning rate: {current_lr:.6g} ---")
        # ------------------------- train ----------------------------------
        model.train()
        total = 0.0
        correct = 0.0
        total_samples = 0
        for batch in train_loader:
            optimizer.zero_grad()

            if task == "classification":
                x, y = batch
                logits = model(x.to(device))
                loss = criterion(logits, y.to(device))
                acc = accuracy(logits, y.to(device))
                correct += acc * y.size(0)
                total_samples += y.size(0)
            elif task in {"regression", "time_series"}:
                x, y = batch
                preds = model(x.to(device))
                loss = criterion(preds, y.to(device))
                acc = None  # Not applicable
            elif task == "language_modeling":
                x, targets = batch
                logits = model(x.to(device))
                loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1).to(device))
                acc = None  # Not applicable
            elif task == "ner":
                x, tags = batch
                logits = model(x.to(device))
                loss = criterion(logits.view(-1, logits.size(-1)), tags.view(-1).to(device))
                # Flatten for token-level accuracy
                acc = accuracy(logits.view(-1, logits.size(-1)), tags.view(-1).to(device))
                correct += acc * tags.numel()
                total_samples += tags.numel()
            else:
                raise ValueError(f"Unsupported task: {task}")

            loss.backward()
            # Gradient clipping for stability
            if grad_clip is not None and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            optimizer.step()
            scheduler.step()
            total += loss.item()
        train_loss = total / max(1, len(train_loader))
        if task in {"classification", "ner"}:
            train_acc = correct / max(1, total_samples)
        else:
            train_acc = None
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)

        # ------------------------- validation -----------------------------
        model.eval()
        total_val = 0.0
        correct_val = 0.0
        total_val_samples = 0
        with torch.no_grad():
            for batch in val_loader:
                if task == "classification":
                    x, y = batch
                    logits = model(x.to(device))
                    loss = criterion(logits, y.to(device))
                    acc = accuracy(logits, y.to(device))
                    correct_val += acc * y.size(0)
                    total_val_samples += y.size(0)
                elif task in {"regression", "time_series"}:
                    x, y = batch
                    preds = model(x.to(device))
                    loss = criterion(preds, y.to(device))
                    acc = None
                elif task == "language_modeling":
                    x, targets = batch
                    logits = model(x.to(device))
                    loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1).to(device))
                    acc = None
                elif task == "ner":
                    x, tags = batch
                    logits = model(x.to(device))
                    loss = criterion(logits.view(-1, logits.size(-1)), tags.view(-1).to(device))
                    acc = accuracy(logits.view(-1, logits.size(-1)), tags.view(-1).to(device))
                    correct_val += acc * tags.numel()
                    total_val_samples += tags.numel()
                else:
                    raise ValueError(f"Unsupported task: {task}")
                total_val += loss.item()
        val_loss = total_val / max(1, len(val_loader))
        if task in {"classification", "ner"}:
            val_acc = correct_val / max(1, total_val_samples)
        else:
            val_acc = None
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"Epoch {epoch:02d}/{epochs} | train {train_loss:.4f} | val {val_loss:.4f} | train_acc {train_acc if train_acc is not None else 'NA':.4f} | val_acc {val_acc if val_acc is not None else 'NA':.4f}")

    return {
        "history": history,
        "final_train_loss": history["train_loss"][-1],
        "final_val_loss": history["val_loss"][-1],
        "final_train_acc": history["train_acc"][-1],
        "final_val_acc": history["val_acc"][-1],
    }


# ------------------------ Legacy helpers removed -----------------------------
# The old dynamic-parser implementation has been fully removed to avoid
# confusion and maintenance overhead. Users should migrate to the simplified
# JSON-based CLI exposed via `_build_simple_parser` and the `main` defined
# below. Any attempt to import the deprecated helpers will raise `ImportError`.

# -----------------------------------------------------------------------------
# Utility helpers --------------------------------------------------------------
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# CLI -------------------------------------------------------------------------
# -----------------------------------------------------------------------------


def _build_simple_parser() -> argparse.ArgumentParser:
    """Return an ArgumentParser with **explicit** arguments only.

    Compared to the legacy dynamic `_build_parser`, this version:
    1. Accepts JSON strings for `--model_kwargs` and `--dataset_kwargs` so that
       experimental hyper-parameters can be provided without brittle
       auto-introspection.
    2. Drops automatic signature inspection, greatly simplifying maintenance.
    3. Keeps the original core hyper-parameters for backwards compatibility.
    """
    p = argparse.ArgumentParser(description="Unified TFN Trainer (Simplified CLI)")

    # Core identifiers -----------------------------------------------------
    p.add_argument("--task", required=True, choices=[
        "classification", "regression", "time_series", "language_modeling", "ner",
    ])
    p.add_argument("--dataset", required=True)
    p.add_argument("--model", required=True)

    # Optim / training -----------------------------------------------------
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--dry_run", action="store_true")

    # New explicit kwargs hooks -------------------------------------------
    p.add_argument(
        "--model_kwargs",
        type=str,
        default="{}",
        help="JSON string with model-specific hyper-parameters",
    )
    p.add_argument(
        "--dataset_kwargs",
        type=str,
        default="{}",
        help="JSON string with dataset-loader arguments",
    )

    return p


# ------------------------------ helpers -------------------------------------

def _merge_kwargs(defaults: dict, overrides: dict) -> dict:
    """Return a new dict = defaults ∪ overrides (overrides win)."""
    out = defaults.copy()
    out.update(overrides)
    return out


# ---------------------------------------------------------------------------
# New main implementation (overrides legacy version defined above)
# ---------------------------------------------------------------------------

def main(argv: Optional[list[str]] = None):  # noqa: C901 – complexity fine here
    parser = _build_simple_parser()
    args = parser.parse_args(argv)

    # ------------------------------------------------------------------
    # Parse JSON kwargs (robust-ish – descriptive error if malformed)
    # ------------------------------------------------------------------
    try:
        user_model_kwargs: dict = json.loads(args.model_kwargs)
        user_dataset_kwargs: dict = json.loads(args.dataset_kwargs)
    except json.JSONDecodeError as e:
        parser.error(f"Malformed JSON in --model_kwargs / --dataset_kwargs: {e}")

    # ------------------------------------------------------------------
    # Compatibility checks (reuse registry utilities)
    # ------------------------------------------------------------------
    if not validate_model_task_compatibility(args.model, args.task):
        parser.error(f"Model '{args.model}' is incompatible with task '{args.task}'.")

    if not validate_dataset_task_compatibility(args.dataset, args.task):
        parser.error(f"Dataset '{args.dataset}' is incompatible with task '{args.task}'.")

    # ------------------------------------------------------------------
    # Resolve model + dataset configurations from registries
    # ------------------------------------------------------------------
    model_cfg = get_model_config(args.model)
    dataset_loader_fn = _list_datasets()[args.dataset]

    default_model_kwargs: dict = model_cfg.get("defaults", {})
    default_dataset_kwargs: dict = getattr(dataset_loader_fn, "default_params", {})

    model_kwargs = _merge_kwargs(default_model_kwargs, user_model_kwargs)
    dataset_kwargs = _merge_kwargs(default_dataset_kwargs, user_dataset_kwargs)

    # ------------------------------------------------------------------
    # Instantiate dataset (expecting new-style return (train, val, meta_dict))
    # ------------------------------------------------------------------
    train_ds, val_ds, meta = dataset_loader_fn(**dataset_kwargs)

    # ------------------------------------------------------------------
    # Legacy loader compatibility
    # ------------------------------------------------------------------
    if isinstance(meta, int):
        # Assume int == vocab_size for text datasets
        meta = {"vocab_size": meta}
    elif not isinstance(meta, dict):
        # Fallback – wrap under key 'info'
        meta = {"info": meta}

    # Allow dataset metadata to fill in missing model kwargs (unless user override)
    for k, v in meta.items():
        model_kwargs.setdefault(k, v)

    # Convenience mappings ------------------------------------------------
    if "input_dim" in get_required_params(args.model) and "input_dim" not in model_kwargs:
        if "num_features" in meta:
            model_kwargs["input_dim"] = meta["num_features"]

    # ------------------------------------------------------------------
    # Instantiate DataLoaders
    # ------------------------------------------------------------------
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    # ------------------------------------------------------------------
    # Build the model
    # ------------------------------------------------------------------
    model_cls = model_cfg["class"]

    # Pass task hint if model expects it
    if "task" in model_cls.__init__.__code__.co_varnames and "task" not in model_kwargs:
        model_kwargs["task"] = args.task

    model = model_cls(**model_kwargs)

    # Pretty print configuration ---------------------------------------
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[TFN] {args.model} with {total_params:,} parameters")

    if args.dry_run:
        print("[DRY-RUN] Successful initialisation; exiting.")
        return

    # ------------------------------------------------------------------
    # Train & evaluate
    # ------------------------------------------------------------------
    metrics = train_and_evaluate(
        model,
        args.task,
        train_loader,
        val_loader,
        epochs=args.epochs,
        device=args.device,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        grad_clip=args.grad_clip,
    )

    # Persist artefacts --------------------------------------------------
    out_dir = Path("outputs") / f"{args.dataset}_{args.model}"
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "history.json", "w") as fp:
        json.dump(metrics["history"], fp)

    torch.save(model.state_dict(), out_dir / "best_model.pt")

    print(
        f"✔ Training complete. Final val loss = {metrics['final_val_loss']:.4f} | "
        f"Final val acc = {metrics['final_val_acc'] if metrics['final_val_acc'] is not None else 'NA':.4f}"
    )


# NOTE: The legacy `_build_parser` and `main` remain above for reference but
# are overshadowed by the simplified versions defined here. This keeps the diff
# minimal and allows roll-back if needed. 