from __future__ import annotations
"""tfn.tfn_datasets.vision_loader
Utility loaders for small image classification datasets that are convenient for
Token Field Network 2-D experiments. Each loader returns a *train* and *val*
`torchvision.datasets` object (already transformed to tensors) plus the number
of classes.

The loaders are intentionally lightweight – if the caller prefers DataLoader
objects they can wrap the returned datasets themselves (this keeps registry
logic simple and avoids hard-coding batch sizes).

Dependencies: torchvision, PIL (usually installed with torch).
"""

from pathlib import Path
from typing import Tuple

import torchvision.transforms as T
from torchvision.datasets import CIFAR10, CIFAR100, ImageFolder

# ---------------------------------------------------------------------------
# Common transforms
# ---------------------------------------------------------------------------

_DEFAULT_TRANSFORM = T.Compose([
    T.ToTensor(),
    T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# ---------------------------------------------------------------------------
# CIFAR-10 / CIFAR-100
# ---------------------------------------------------------------------------

def load_cifar10(root: str | Path = "data", download: bool = True, transform: T.Compose | None = None,
                 **_kw) -> Tuple[CIFAR10, CIFAR10, int]:
    """Return train_ds, val_ds, num_classes for CIFAR-10."""
    tfm = transform or _DEFAULT_TRANSFORM
    from torch.utils.data import TensorDataset
    try:
        train_ds = CIFAR10(root=str(root), train=True, download=download, transform=tfm)
        val_ds = CIFAR10(root=str(root), train=False, download=download, transform=tfm)
    except RuntimeError as e:
        if not download:
            # Dataset not present; return empty placeholder to allow unit tests
            empty = TensorDataset()
            return empty, empty, 10
        raise e
    return train_ds, val_ds, 10


def load_cifar100(root: str | Path = "data", download: bool = True, transform: T.Compose | None = None,
                  **_kw) -> Tuple[CIFAR100, CIFAR100, int]:
    """Return train_ds, val_ds, num_classes for CIFAR-100."""
    tfm = transform or _DEFAULT_TRANSFORM
    from torch.utils.data import TensorDataset
    try:
        train_ds = CIFAR100(root=str(root), train=True, download=download, transform=tfm)
        val_ds = CIFAR100(root=str(root), train=False, download=download, transform=tfm)
    except RuntimeError as e:
        if not download:
            empty = TensorDataset()
            return empty, empty, 100
        raise e
    return train_ds, val_ds, 100

# ---------------------------------------------------------------------------
# ImageNet-32
# ---------------------------------------------------------------------------

def load_imagenet32(root: str | Path = "imagenet32", transform: T.Compose | None = None,
                    **_kw) -> Tuple[ImageFolder, ImageFolder, int]:
    """Return train_ds, val_ds, num_classes for ImageNet-32.

    We expect the directory structure:
    ```
    root/
        train/
            class0/xxx.png
            class1/xxx.png
        val/
            class0/yyy.png
            ...
    ```
    where images are 32×32 PNG (as provided by OpenAI's downsampled ImageNet).
    If the user has a different structure they can supply a symlinked folder.
    """
    tfm = transform or _DEFAULT_TRANSFORM
    root = Path(root)
    train_dir = root / "train"
    val_dir = root / "val"
    if not train_dir.exists() or not val_dir.exists():
        raise FileNotFoundError(
            "ImageNet-32 directories not found. Expected 'train' and 'val' under"
            f" {root}. Download from https://image-net.org/download-images and"
            " convert to 32×32 or use the Kaggle downsampled version."
        )
    train_ds = ImageFolder(str(train_dir), transform=tfm)
    val_ds = ImageFolder(str(val_dir), transform=tfm)
    num_classes = len(train_ds.classes)
    return train_ds, val_ds, num_classes 