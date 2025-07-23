"""tfn.tfn_datasets.registry
Centralised mapping from dataset-name strings → loader callables so that all
training scripts can obtain `(train_ds, val_ds, meta)` with a single function.

Each loader must accept **kwargs to keep the API flexible. For datasets that
require heavy external resources (e.g. ImageNet-32) we expose a thin wrapper
that raises a helpful error if the data is missing.
"""
from __future__ import annotations

from typing import Callable, Dict, Any, Tuple

# Import individual loader modules -------------------------------------------------
from . import (
    glue_loader,
    long_text_loader,
    pg19_loader,
    arxiv_loader,
    climate_loader,
    physics_loader,
    ner_loader,
    dataset_loaders,
    vision_loader,
)
from tfn.utils.synthetic_sequence_tasks import get_synthetic_sequence_dataloaders

__all__ = [
    "get_dataset",
    "list_datasets",
]

# ---------------------------------------------------------------------------
# Internal mapping {name: callable}
# The callable must return at minimum (train_ds, val_ds, <any meta info>)
# ---------------------------------------------------------------------------

_DATASET_REGISTRY: Dict[str, Callable[..., Any]] = {
    # ----------------------- GLUE ------------------------------------------------
    "sst2": glue_loader.load_sst2,
    "mrpc": glue_loader.load_mrpc,
    "qqp": glue_loader.load_qqp,
    "qnli": glue_loader.load_qnli,
    "rte": glue_loader.load_rte,
    "cola": glue_loader.load_cola,
    "stsb": glue_loader.load_stsb,
    "wnli": glue_loader.load_wnli,
    # ----------------------- generic text ---------------------------------------
    "agnews": dataset_loaders.load_agnews,
    "yelp_full": dataset_loaders.load_yelp_full,
    "imdb": dataset_loaders.load_imdb,
    "arxiv": arxiv_loader.load_arxiv if hasattr(arxiv_loader, "load_arxiv") else None,
    # ----------------------- long text / language modelling ---------------------
    "pg19": pg19_loader.create_pg19_dataloader if hasattr(pg19_loader, "create_pg19_dataloader") else None,
    "long_text_synth": long_text_loader.create_long_text_dataloader,
    # ----------------------- time-series ----------------------------------------
    "electricity": climate_loader.load_electricity if hasattr(climate_loader, "load_electricity") else None,
    "jena": climate_loader.load_jena_single if hasattr(climate_loader, "load_jena_single") else None,
    "jena_multi": climate_loader.load_jena_multi if hasattr(climate_loader, "load_jena_multi") else None,
    # ----------------------- physics / PDE --------------------------------------
    "burgers": lambda **kw: physics_loader.load_physics_dataset(dataset_type="burgers", **kw),
    "wave": lambda **kw: physics_loader.load_physics_dataset(dataset_type="wave", **kw),
    "heat": lambda **kw: physics_loader.load_physics_dataset(dataset_type="heat", **kw),
    # ----------------------- NER -------------------------------------------------
    "conll2003": ner_loader.load_conll if hasattr(ner_loader, "load_conll") else None,
    # ----------------------- synthetic seq tasks --------------------------------
    "synthetic_copy": lambda **kw: get_synthetic_sequence_dataloaders(task="copy", **kw),
    "synthetic_reverse": lambda **kw: get_synthetic_sequence_dataloaders(task="reverse", **kw),
}

# Vision datasets via torchvision / ImageFolder
_DATASET_REGISTRY.update({
    "cifar10": vision_loader.load_cifar10,
    "cifar100": vision_loader.load_cifar100,
    "imagenet32": vision_loader.load_imagenet32,
})


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def get_dataset(name: str, **kwargs: Any):
    """Return (train_ds, val_ds, meta) for *name*.

    Parameters
    ----------
    name : str
        Canonical dataset key (see `list_datasets()`).
    **kwargs : Any
        Passed verbatim to the underlying loader.
    """
    if name not in _DATASET_REGISTRY or _DATASET_REGISTRY[name] is None:
        raise ValueError(f"Dataset '{name}' not available. Known: {list(_DATASET_REGISTRY.keys())}")
    loader = _DATASET_REGISTRY[name]
    return loader(**kwargs)


def list_datasets() -> Dict[str, Callable[..., Any]]:
    """Return the internal registry mapping (name → loader)."""
    return dict(_DATASET_REGISTRY) 