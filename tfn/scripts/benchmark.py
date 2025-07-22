"""tfn.scripts.benchmark
Programmatic benchmark runner – no `subprocess` or stdout parsing.

Example
-------
$ python -m tfn.scripts.benchmark --preset quick
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Dict, Any, List

import torch

from tfn.scripts.train import train_and_evaluate
from tfn.model.registry import get_model_config
from tfn.datasets.registry import get_dataset_config

# -----------------------------------------------------------------------------
# Pre-defined sweep presets ----------------------------------------------------
# -----------------------------------------------------------------------------

PRESETS: Dict[str, Dict[str, Any]] = {
    "quick": {
        "datasets": ["agnews", "electricity"],
        "models": ["tfn", "transformer"],
        "epochs": 1,
        "batch_size": 8,
    },
    "nlp_small": {
        "datasets": ["agnews", "sst2", "yelp_full"],
        "models": ["tfn", "transformer", "lstm"],
        "epochs": 5,
        "batch_size": 32,
    },
    "nlp_full": {
        "datasets": [
            "agnews", "imdb", "yelp_full",  # generic text
            "sst2", "mrpc", "qqp", "qnli", "rte", "cola", "stsb", "wnli",  # GLUE
        ],
        "models": ["tfn", "transformer", "performer", "lstm", "cnn"],
        "epochs": 5,
        "batch_size": 32,
    },
    "time_series": {
        "datasets": ["electricity", "jena", "jena_multi"],
        "models": ["tfn", "transformer", "lstm"],
        "epochs": 5,
        "batch_size": 64,
    },
}


# -----------------------------------------------------------------------------
# Helpers ----------------------------------------------------------------------
# -----------------------------------------------------------------------------

def _instantiate_dataset(name: str):
    d_cfg = get_dataset_config(name)
    loader_fn = d_cfg["loader_function"]
    if isinstance(loader_fn, str):
        from importlib import import_module

        parts = loader_fn.split(".")
        mod = import_module(".".join(["tfn", "tfn_datasets"] + parts[:-1]))
        loader_fn = getattr(mod, parts[-1])
    train_ds, val_ds, *extra = loader_fn(**d_cfg.get("default_params", {}))
    return train_ds, val_ds, extra


def _instantiate_model(name: str, extra_info: List[Any]):
    m_cfg = get_model_config(name)
    kwargs = m_cfg.get("defaults", {}).copy()
    for k, v in zip(["vocab_size", "num_classes", "output_dim", "num_tags"], extra_info):
        if v is not None:
            kwargs.setdefault(k, v)
    return m_cfg["class"](**kwargs)


# -----------------------------------------------------------------------------
# Main benchmark driver --------------------------------------------------------
# -----------------------------------------------------------------------------

def run_benchmark(preset: str = "quick"):
    if preset not in PRESETS:
        raise ValueError(f"Unknown preset '{preset}'. Options: {list(PRESETS)}")
    cfg = PRESETS[preset]

    results = []

    # Simple heuristic mapping from dataset → task-type
    _TASK_MAP = {
        "electricity": "time_series",
        "jena": "time_series",
        "jena_multi": "time_series",
        # Everything else defaults to classification
    }

    for dataset_name, model_name in itertools.product(cfg["datasets"], cfg["models"]):
        task_type = _TASK_MAP.get(dataset_name, "classification")

        print(f"📊 Running {model_name} on {dataset_name} (task: {task_type})…")
        train_ds, val_ds, extra = _instantiate_dataset(dataset_name)
        model = _instantiate_model(model_name, extra)

        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False)

        metrics = train_and_evaluate(
            model,
            task_type,
            train_loader,
            val_loader,
            epochs=cfg["epochs"],
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

        results.append({
            "dataset": dataset_name,
            "task": task_type,
            "model": model_name,
            "val_loss": metrics["final_val_loss"],
        })

    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    with open(out_dir / f"benchmark_{preset}.json", "w") as fp:
        json.dump(results, fp, indent=2)

    print("✔ Benchmark complete ∙ results saved to", out_dir)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--preset", default="quick", help=f"One of: {list(PRESETS)}")
    args = p.parse_args()

    run_benchmark(args.preset) 