from __future__ import annotations

"""tfn.datasets
Standard dataset loaders used across TFN examples and scripts.

Currently contains:
    • load_agnews
    • load_yelp_full
    • load_imdb
    • GLUE tasks (sst2, mrpc, qqp, qnli, rte, cola, stsb, wnli)
    • load_arxiv
    • Climate datasets (electricity, jena, jena_multi)

All loaders return `(train_ds, val_ds, vocab_size)` where the datasets are
`torch.utils.data.TensorDataset` objects with tensors `(input_ids, labels)`.
"""

from importlib import import_module as _imp

__all__ = [
    "load_agnews",
    "load_yelp_full", 
    "load_imdb",
    # GLUE tasks
    "load_sst2",
    "load_mrpc", 
    "load_qqp",
    "load_qnli",
    "load_rte",
    "load_cola",
    "load_stsb",
    "load_wnli",
    # Arxiv
    "load_arxiv",
    # Climate datasets
    "load_electricity_transformer",
    "load_jena_climate",
    "load_jena_climate_multi",
]

# Use relative import instead of absolute
try:
    _text_mod = _imp(".text_classification", package="tfn_datasets")
    _glue_mod = _imp(".glue_loader", package="tfn_datasets")
    _arxiv_mod = _imp(".arxiv_loader", package="tfn_datasets")
    _climate_mod = _imp(".climate_loader", package="tfn_datasets")
    
    # Text classification loaders
    for name in ["load_agnews", "load_yelp_full", "load_imdb"]:
        globals()[name] = getattr(_text_mod, name)
    
    # GLUE loaders
    for name in ["load_sst2", "load_mrpc", "load_qqp", "load_qnli", "load_rte", "load_cola", "load_stsb", "load_wnli"]:
        globals()[name] = getattr(_glue_mod, name)
    
    # Arxiv loader
    globals()["load_arxiv"] = getattr(_arxiv_mod, "load_arxiv")
    
    # Climate loaders
    for name in ["load_electricity_transformer", "load_jena_climate", "load_jena_climate_multi"]:
        globals()[name] = getattr(_climate_mod, name)
        
except ImportError:
    # Fallback for when running from different contexts
    pass 