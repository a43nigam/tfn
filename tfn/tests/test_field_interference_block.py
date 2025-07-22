import pytest
import torch

from tfn.core.field_interference_block import ImageFieldInterference


def test_image_field_interference_token_mode() -> None:
    """Token-mode input should keep shape and be differentiable."""
    torch.manual_seed(0)
    batch, seq_len, channels = 2, 6, 32
    x = torch.randn(batch, seq_len, channels, requires_grad=True)

    module = ImageFieldInterference(num_heads=8)
    out = module(x)

    assert out.shape == x.shape
    out.sum().backward()
    assert x.grad is not None


def test_image_field_interference_image_mode() -> None:
    """Image-mode 4-D input should be processed and have same shape."""
    torch.manual_seed(0)
    batch, channels, height, width = 2, 32, 4, 4
    x = torch.randn(batch, channels, height, width, requires_grad=True)

    module = ImageFieldInterference(num_heads=8)
    out = module(x)

    assert out.shape == x.shape
    out.mean().backward()
    assert x.grad is not None


def test_image_field_interference_invalid_shape() -> None:
    """Passing an unsupported tensor rank should raise ValueError."""
    x = torch.randn(3, 5)  # 2-D tensor is invalid
    module = ImageFieldInterference(num_heads=4)
    with pytest.raises(ValueError):
        _ = module(x) 