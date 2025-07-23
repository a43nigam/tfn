import pytest
import torch

from tfn.model.tfn_base import TrainableTFNLayer
from tfn.model.seq_baselines import (
    SimpleTransformerSeqModel,
    SimplePerformerSeqModel,
)


@pytest.mark.parametrize("seq_len", [256, 512, 1024])
@pytest.mark.parametrize("model_type", ["tfn", "transformer", "performer"])
def test_model_forward_pass(seq_len: int, model_type: str):
    """Ensure that each model variant can perform a single forward pass.

    The test intentionally keeps batch/embedding dimensions small to remain
    lightweight yet meaningful. If a GPU is available, it will be utilised
    automatically.
    """
    batch_size, embed_dim, vocab_size = 2, 128, 1000
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------
    # Model construction
    # ------------------------------------------------------------------
    if model_type == "tfn":
        model = torch.nn.Sequential(
            torch.nn.Embedding(vocab_size, embed_dim),
            TrainableTFNLayer(
                embed_dim=embed_dim,
                kernel_type="rbf",
                evolution_type="cnn",
                grid_size=32,
                time_steps=3,
                max_seq_len=seq_len,
            ),
            torch.nn.Linear(embed_dim, vocab_size),
        )
    elif model_type == "transformer":
        model = SimpleTransformerSeqModel(
            vocab_size=vocab_size,
            seq_len=seq_len,
            embed_dim=embed_dim,
            num_layers=2,
        )
    else:  # performer
        model = SimplePerformerSeqModel(
            vocab_size=vocab_size,
            seq_len=seq_len,
            embed_dim=embed_dim,
            num_layers=2,
        )

    model.to(device)
    model.eval()

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

    with torch.no_grad():
        if model_type == "tfn":
            # 1-D positional coordinates in \[0,1]. Shape: [B, L, 1]
            positions = (
                torch.linspace(0, 1, seq_len, device=device)
                .view(1, seq_len, 1)
                .expand(batch_size, -1, -1)
            )
            output = model[1](model[0](input_ids), positions)
        else:
            output = model(input_ids)

    # ------------------------------------------------------------------
    # Assertions
    # ------------------------------------------------------------------
    assert output is not None, "Model returned None"
    assert output.shape[0] == batch_size, "Batch dimension mismatch" 