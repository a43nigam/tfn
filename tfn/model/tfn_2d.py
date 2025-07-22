"""Stub module for deprecated 2-D TFN implementation.

The previous *tfn_2d.py* (full 2-D TFN) has been removed during model
consolidation.  Importing from this module is now deprecated.  Use
`tfn.model.tfn_pytorch.ImageTFN` for image tasks instead.
"""

from __future__ import annotations

import warnings
import torch.nn as nn
from .tfn_pytorch import ImageTFN

warnings.warn(
    "tfn.model.tfn_2d is deprecated and will be removed in a future release. "
    "Please switch to tfn.model.tfn_pytorch.ImageTFN.",
    DeprecationWarning,
    stacklevel=2,
)

# ---------------------------------------------------------------------------
# Minimal alias – keeps old names alive but maps to ImageTFN where feasible
# ---------------------------------------------------------------------------

class TrainableTFNLayer2D(nn.Module):
    """Removed component.  Exists only to avoid import errors."""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "TrainableTFNLayer2D has been removed. Use ImageTFN or rewrite your "
            "model to operate on images directly."
        )


class TFNClassifier2D(ImageTFN):
    """Alias to the new ImageTFN class (2-D image model)."""

    def __init__(self, *args, **kwargs):
        # Adapt num_classes argument if passed positionally/keyword
        num_classes = kwargs.get("num_classes", 10)
        super().__init__(in_ch=3, num_classes=num_classes)


def create_tfn2d_variants(num_classes: int = 10):  # type: ignore
    """Return dict with a single basic ImageTFN variant (back-compat)."""
    return {"image_basic": TFNClassifier2D(num_classes=num_classes)} 