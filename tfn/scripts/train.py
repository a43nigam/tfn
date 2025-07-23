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
from tfn.datasets.registry import (
    DATASET_REGISTRY,
    get_dataset_config,
    validate_dataset_task_compatibility,
)

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


# -----------------------------------------------------------------------------
# Utility helpers --------------------------------------------------------------
# -----------------------------------------------------------------------------


def _infer_arg_type(param_name: str, model_cls: type) -> type:
    """Infer CLI argument type for *param_name* based on *model_cls* signature.

    Falls back to ``str`` when type cannot be determined reliably.
    """
    try:
        # Resolve type hints (handles PEP 563 "string" annotations)
        from typing import get_type_hints

        type_hints = get_type_hints(model_cls.__init__, localns=model_cls.__dict__)

        ann = type_hints.get(param_name)
        if ann is None:  # Fallback to raw signature annotation
            sig = inspect.signature(model_cls.__init__)
            if param_name in sig.parameters:
                ann = sig.parameters[param_name].annotation

        # ------------------------------------------------------------------
        # Basic primitive types
        # ------------------------------------------------------------------
        if ann in {int, float, bool, str}:
            return ann
        # Handle Optional annotations like Optional[int]
        from typing import get_origin, get_args

        origin = get_origin(ann)
        if origin is None:
            return str
        if origin is Union:  # type: ignore[name-defined]
            # Return first non-None arg if it's simple
            for arg in get_args(ann):
                if arg is not type(None) and arg in {int, float, bool, str}:
                    return arg
    except Exception:
        pass  # Robust fallback – treat as str on any failure
    return str

# -----------------------------------------------------------------------------
# CLI -------------------------------------------------------------------------
# -----------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Unified TFN Trainer")
    p.add_argument("--task", required=True, choices=[
        "classification", "regression", "time_series", "language_modeling", "ner"
    ])
    p.add_argument("--dataset", required=True)
    p.add_argument("--model", required=True)

    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--dry_run", action="store_true")
    return p


# --------------------- arg handling helpers (simplified) ----------------------


def main(argv: list[str] | None = None):
    argv = argv or sys.argv[1:]
    base_parser = _build_parser()
    base_args, _ = base_parser.parse_known_args(argv)

    # Validate compatibility ------------------------------------------------
    if not validate_model_task_compatibility(base_args.model, base_args.task):
        sys.exit(f"Model {base_args.model} is incompatible with task {base_args.task}.")
    if not validate_dataset_task_compatibility(base_args.dataset, base_args.task):
        sys.exit(f"Dataset {base_args.dataset} is incompatible with task {base_args.task}.")

    # Fetch registry configs -----------------------------------------------
    m_cfg = get_model_config(base_args.model)
    d_cfg = get_dataset_config(base_args.dataset)

    # Dynamically add model-specific arguments
    model_required = get_required_params(base_args.model)
    model_optional = get_optional_params(base_args.model)

    # Dynamically add dataset-specific arguments by inspecting loader signature
    loader_fn = d_cfg["loader_function"]
    if isinstance(loader_fn, str):
        from importlib import import_module
        parts = loader_fn.split(".")
        mod = import_module(".".join(["tfn", "tfn_datasets"] + parts[:-1]))
        loader_fn = getattr(mod, parts[-1])
    dataset_params = []
    dataset_defaults = {}
    sig = inspect.signature(loader_fn)
    for name, param in sig.parameters.items():
        if name == "self":
            continue
        if param.default is inspect.Parameter.empty:
            dataset_params.append(name)
        else:
            dataset_defaults[name] = param.default

    # Build a new parser with all dynamic args
    parser = _build_parser()
    # Track already-registered option strings to avoid argparse conflicts
    _existing_opts = set(parser._option_string_actions)
    for param in model_required:
        if param in {"task", "dataset", "model", "epochs", "batch_size", "lr", "device", "num_workers", "dry_run"}:
            continue

        opt = f"--{param}"
        if opt in _existing_opts:
            continue

        # Determine if we can supply a default automatically (from model defaults or dataset meta)
        default_val = None
        if param in m_cfg.get("defaults", {}):
            default_val = m_cfg["defaults"][param]
        elif param in d_cfg:
            default_val = d_cfg[param]
        elif param in d_cfg.get("default_params", {}):
            default_val = d_cfg["default_params"][param]

        if default_val is not None:
            parser.add_argument(opt, default=default_val, type=type(default_val))
        else:
            # Infer type from model signature; default to str if unknown
            inferred_type = _infer_arg_type(param, m_cfg["class"])
            parser.add_argument(opt, required=True, type=inferred_type)

        _existing_opts.add(opt)
    for param in model_optional:
        if param not in {"task", "dataset", "model", "epochs", "batch_size", "lr", "device", "num_workers", "dry_run"}:
            opt = f"--{param}"
            if opt not in _existing_opts:
                parser.add_argument(opt, default=m_cfg.get("defaults", {}).get(param))
                _existing_opts.add(opt)
    for param in dataset_params:
        if param not in {"task", "dataset", "model", "epochs", "batch_size", "lr", "device", "num_workers", "dry_run"}:
            opt = f"--{param}"
            if opt not in _existing_opts:
                # Infer type for dataset-required params (best-effort) – default to str
                inferred_type = _infer_arg_type(param, m_cfg["class"])
                parser.add_argument(opt, required=True, type=inferred_type)
                _existing_opts.add(opt)
    for param, default in dataset_defaults.items():
        if param not in {"task", "dataset", "model", "epochs", "batch_size", "lr", "device", "num_workers", "dry_run"}:
            opt = f"--{param}"
            if opt not in _existing_opts:
                parser.add_argument(opt, type=type(default), default=default)
                _existing_opts.add(opt)

    args = parser.parse_args(argv)

    # Build kwargs from parsed args
    model_kwargs = m_cfg.get("defaults", {}).copy()
    dataset_kwargs = d_cfg.get("default_params", {}).copy()
    for param in model_required + model_optional:
        if hasattr(args, param):
            model_kwargs[param] = getattr(args, param)
    for param in dataset_params + list(dataset_defaults.keys()):
        if hasattr(args, param):
            dataset_kwargs[param] = getattr(args, param)

    # Instantiate dataset ---------------------------------------------------
    train_ds, val_ds, *extra = loader_fn(**dataset_kwargs)

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

    # Add extra dataset info (e.g., vocab_size) to model_kwargs -------------
    for k, v in zip(["vocab_size", "num_classes", "output_dim", "num_tags"], extra):
        if k not in model_kwargs and v is not None:
            model_kwargs[k] = v

    # Fallback: use static dataset registry values if still missing
    for k in ["vocab_size", "num_classes", "output_dim", "num_tags", "num_features"]:
        if k not in model_kwargs and d_cfg.get(k) is not None:
            model_kwargs[k] = d_cfg[k]

    # Map generic dataset fields to model-required names
    if "input_dim" in model_required and "input_dim" not in model_kwargs:
        if "num_features" in model_kwargs:
            model_kwargs["input_dim"] = model_kwargs["num_features"]
        elif "output_dim" in model_kwargs:  # fallback single feature
            model_kwargs["input_dim"] = model_kwargs["output_dim"]
        else:
            model_kwargs["input_dim"] = 1

    if "output_len" in model_required and "output_len" not in model_kwargs:
        if "output_dim" in model_kwargs:
            model_kwargs["output_len"] = model_kwargs["output_dim"]
        else:
            model_kwargs["output_len"] = 1

    model_cls = m_cfg["class"]

    # Ensure 'task' argument is passed if model expects it (Unified TFN)
    if "task" in model_cls.__init__.__code__.co_varnames and "task" not in model_kwargs:
        model_kwargs["task"] = args.task

    model = model_cls(**model_kwargs)

    # ------------------- PRINT MODEL PARAMETER COUNT ----------------------
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total model parameters:     {total_params:,}")
    print(f"Trainable parameters:       {trainable_params:,}\n")
    # ----------------------------------------------------------------------

    # ------------------- PRINT TRAINING CONFIGURATION ----------------------
    print("\n================ TFN TRAINING CONFIGURATION ================")
    print(f"Model:        {args.model}")
    print(f"Task:         {args.task}")
    print(f"Dataset:      {args.dataset}")
    print(f"Train size:   {len(train_ds)} samples")
    print(f"Val size:     {len(val_ds)} samples")
    print(f"Batch size:   {args.batch_size}")
    print(f"Epochs:       {args.epochs}")
    print(f"Learning rate:{args.lr}")
    print(f"Device:       {args.device}")
    # Print extra dataset info if available
    for k in ["vocab_size", "num_classes", "output_dim", "num_tags", "num_features"]:
        if k in model_kwargs:
            print(f"{k.replace('_', ' ').capitalize()}: {model_kwargs[k]}")
    # Print all model hyperparameters
    print("Model hyperparameters:")
    for k, v in model_kwargs.items():
        print(f"  {k}: {v}")
    print("===========================================================\n")
    # ----------------------------------------------------------------------

    if args.dry_run:
        print("[DRY-RUN] Model & dataset instantiated successfully – exiting.")
        return

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

    # Save metrics to outputs/ directory -----------------------------------
    out_dir = Path("outputs") / f"{args.dataset}_{args.model}"
    out_dir.mkdir(parents=True, exist_ok=True)
    import json

    with open(out_dir / "history.json", "w") as fp:
        json.dump(metrics["history"], fp)

    torch.save(model.state_dict(), out_dir / "best_model.pt")

    print(f"✔ Training complete. Final val loss = {metrics['final_val_loss']:.4f} | Final val acc = {metrics['final_val_acc'] if metrics['final_val_acc'] is not None else 'NA':.4f}")


if __name__ == "__main__":
    main() 