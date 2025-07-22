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
from typing import Dict, Any

import torch
import inspect

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
# Helper: generic training loop ------------------------------------------------
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
) -> Dict[str, Any]:
    """Train *model* and return a metrics dictionary."""

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    if task in {"classification", "ner"}:
        criterion = torch.nn.CrossEntropyLoss()
    else:
        criterion = torch.nn.MSELoss()

    history: Dict[str, list[float]] = {"train_loss": [], "val_loss": []}

    for epoch in range(1, epochs + 1):
        # ------------------------- train ----------------------------------
        model.train()
        total = 0.0
        for batch in train_loader:
            optimizer.zero_grad()

            if task == "classification":
                x, y = batch
                logits = model(x.to(device))
                loss = criterion(logits, y.to(device))
            elif task in {"regression", "time_series"}:
                x, y = batch
                preds = model(x.to(device))
                loss = criterion(preds, y.to(device))
            elif task == "language_modeling":
                x, targets = batch
                logits = model(x.to(device))
                loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1).to(device))
            elif task == "ner":
                x, tags = batch
                logits = model(x.to(device))
                loss = criterion(logits.view(-1, logits.size(-1)), tags.view(-1).to(device))
            else:
                raise ValueError(f"Unsupported task: {task}")

            loss.backward()
            optimizer.step()
            total += loss.item()
        train_loss = total / max(1, len(train_loader))
        history["train_loss"].append(train_loss)

        # ------------------------- validation -----------------------------
        model.eval()
        total_val = 0.0
        with torch.no_grad():
            for batch in val_loader:
                if task == "classification":
                    x, y = batch
                    logits = model(x.to(device))
                    loss = criterion(logits, y.to(device))
                elif task in {"regression", "time_series"}:
                    x, y = batch
                    preds = model(x.to(device))
                    loss = criterion(preds, y.to(device))
                elif task == "language_modeling":
                    x, targets = batch
                    logits = model(x.to(device))
                    loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1).to(device))
                elif task == "ner":
                    x, tags = batch
                    logits = model(x.to(device))
                    loss = criterion(logits.view(-1, logits.size(-1)), tags.view(-1).to(device))
                else:
                    raise ValueError(f"Unsupported task: {task}")
                total_val += loss.item()
        val_loss = total_val / max(1, len(val_loader))
        history["val_loss"].append(val_loss)

        print(f"Epoch {epoch:02d}/{epochs} | train {train_loss:.4f} | val {val_loss:.4f}")

    return {
        "history": history,
        "final_train_loss": history["train_loss"][-1],
        "final_val_loss": history["val_loss"][-1],
    }


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
    for param in model_required:
        if param not in {"task", "dataset", "model", "epochs", "batch_size", "lr", "device", "num_workers", "dry_run"}:
            parser.add_argument(f"--{param}", required=True)
    for param in model_optional:
        if param not in {"task", "dataset", "model", "epochs", "batch_size", "lr", "device", "num_workers", "dry_run"}:
            parser.add_argument(f"--{param}", default=m_cfg.get("defaults", {}).get(param))
    for param in dataset_params:
        if param not in {"task", "dataset", "model", "epochs", "batch_size", "lr", "device", "num_workers", "dry_run"}:
            parser.add_argument(f"--{param}", required=True)
    for param, default in dataset_defaults.items():
        if param not in {"task", "dataset", "model", "epochs", "batch_size", "lr", "device", "num_workers", "dry_run"}:
            parser.add_argument(f"--{param}", type=type(default), default=default)

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

    model_cls = m_cfg["class"]
    model = model_cls(**model_kwargs)

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
    )

    # Save metrics to outputs/ directory -----------------------------------
    out_dir = Path("outputs") / f"{args.dataset}_{args.model}"
    out_dir.mkdir(parents=True, exist_ok=True)
    import json

    with open(out_dir / "history.json", "w") as fp:
        json.dump(metrics["history"], fp)

    torch.save(model.state_dict(), out_dir / "best_model.pt")

    print(f"✔ Training complete. Final val loss = {metrics['final_val_loss']:.4f}")


if __name__ == "__main__":
    main() 