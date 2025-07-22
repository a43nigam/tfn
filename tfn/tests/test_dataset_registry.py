import pytest

from tfn.tfn_datasets.registry import get_dataset, list_datasets


def test_list_datasets_contains_key() -> None:
    names = list_datasets().keys()
    assert "synthetic_copy" in names
    assert "sst2" in names


@pytest.mark.parametrize("name", ["synthetic_copy", "synthetic_reverse"])
def test_get_dataset_synthetic(name: str) -> None:
    # Should return train_ds, val_ds, vocab/meta without errors
    train, val, meta = get_dataset(name, seq_len=8, vocab_size=32, train_samples=16, val_samples=8, batch_size=4)
    # quick sanity checks
    assert len(train.dataset) == 16
    assert len(val.dataset) == 8
    assert meta == 32


def test_get_dataset_invalid() -> None:
    with pytest.raises(ValueError):
        _ = get_dataset("nonexistent_dataset")


def test_get_dataset_cifar10_meta_only() -> None:
    """vision loader should return datasets and num_classes."""
    train_ds, val_ds, num_classes = get_dataset("cifar10", root="data", download=False)
    assert num_classes == 10 