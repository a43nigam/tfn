import pytest

from tfn.model.registry import validate_kernel_evolution


@pytest.mark.parametrize(
    "kernel,evolution",
    [
        ("rbf", "cnn"),
        ("compact", "cnn"),
        ("rbf", "pde"),
    ],
)
def test_valid_kernel_evolution_pairs(kernel: str, evolution: str) -> None:
    validate_kernel_evolution(kernel, evolution)  # should NOT raise


@pytest.mark.parametrize(
    "kernel,evolution",
    [
        ("fourier", "cnn"),  # disallowed
        ("compact", "pde"),
    ],
)
def test_invalid_kernel_evolution_pairs(kernel: str, evolution: str) -> None:
    with pytest.raises(ValueError):
        validate_kernel_evolution(kernel, evolution) 