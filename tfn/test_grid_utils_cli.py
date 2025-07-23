"""Lightweight unit tests for `tfn.core.grid_utils`.

These replace the earlier CLI-style script that expected command-line
arguments and iterated over thousands of combinations.  Here we simply verify
that the core helper functions succeed on a few small inputs.
"""

import pytest


from tfn.core.grid_utils import (
    compute_auto_grid_size,
    estimate_memory_usage,
    estimate_flops,
)


@pytest.mark.parametrize(
    "seq_len, embed_dim, heuristic",
    [
        (32, 64, "sqrt"),
        (128, 128, "linear"),
        (256, 64, "adaptive"),
    ],
)
def test_compute_auto_grid_size_basic(seq_len: int, embed_dim: int, heuristic: str) -> None:
    """`compute_auto_grid_size` should return a positive integer and scale with inputs."""
    size = compute_auto_grid_size(seq_len=seq_len, embed_dim=embed_dim, heuristic=heuristic)
    assert isinstance(size, int) and size > 0

    # Sanity-check ancillary helpers on the same config (lightweight)
    mem_info = estimate_memory_usage(batch_size=1, seq_len=seq_len, grid_size=size, embed_dim=embed_dim)
    flops_info = estimate_flops(seq_len=seq_len, grid_size=size, embed_dim=embed_dim)

    assert mem_info["total_memory_mb"] > 0
    assert flops_info["flops_per_token"] > 0 