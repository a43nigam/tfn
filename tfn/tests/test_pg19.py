import os
import pytest

# Heavy test guard ---------------------------------------------------------
if os.getenv("TFN_HEAVY_TESTS", "0") != "1":
    pytest.skip(
        "Skipping PG-19 dataset test; set TFN_HEAVY_TESTS=1 to run.",
        allow_module_level=True,
    )

# Optional external dependency
pytest.importorskip("datasets", reason="`datasets` library is required for PG-19 loader")

import torch

from tfn.tfn_datasets.pg19_loader import create_pg19_dataloader


@pytest.mark.parametrize("seq_len", [256])  # Keep tiny for CI
def test_pg19_dataloader_basic(seq_len: int):
    """Ensure PG-19 dataloader yields batches of correct shape.

    This test is marked as *heavy* and is executed only when the environment
    variable `TFN_HEAVY_TESTS=1` is present. The sequence length and number of
    chunks are deliberately small to minimise resource usage while still
    covering the code path.
    """
    batch_size = 1
    train_loader, val_loader, vocab_size = create_pg19_dataloader(
        seq_len=seq_len,
        batch_size=batch_size,
        max_train_chunks=2,
        max_val_chunks=1,
        vocab_size=500,  # small synthetic vocab to speed up tokenisation
        streaming=True,  # avoid full download when possible
    )

    # Fetch a single batch and sanity-check shapes / dtypes
    batch = next(iter(train_loader))
    assert batch.shape == (batch_size, seq_len), "Batch shape mismatch"
    assert batch.dtype == torch.long, "Batch dtype should be torch.long"
    assert vocab_size >= 10, "Vocab size unexpectedly small" 