import os
import pytest
import torch

# Skip heavy PDE test unless explicitly enabled
if os.getenv("TFN_HEAVY_TESTS", "0") != "1":
    pytest.skip("Skipping physics-PDE tests (set TFN_HEAVY_TESTS=1 to run).", allow_module_level=True)

from tfn.tfn_datasets.physics_loader import create_physics_dataloader, compute_pde_metrics
from tfn.model.tfn_base import TrainableTFNLayer


@pytest.mark.parametrize("pde_type", ["burgers", "wave", "heat"])
def test_tfn_pde_forward(pde_type: str):
    """Ensure TFN layer can process small PDE batches and produce finite outputs."""
    batch_size, grid_points, embed_dim = 2, 32, 64

    train_loader, _ = create_physics_dataloader(
        dataset_type=pde_type,
        batch_size=batch_size,
        num_samples=20,
        grid_points=grid_points,
        time_steps=20,
        input_steps=5,
        output_steps=10,
        num_workers=0,
    )

    model = TrainableTFNLayer(
        embed_dim=embed_dim,
        kernel_type="rbf",
        evolution_type="cnn",
        grid_size=grid_points,
        time_steps=3,
        max_seq_len=grid_points,
    )

    # grab a single batch
    input_seq, target_seq = next(iter(train_loader))
    positions = torch.linspace(0, 1, grid_points).view(1, 1, grid_points).expand(batch_size, 5, -1)
    with torch.no_grad():
        preds = model(input_seq, positions)

    assert preds.shape == target_seq.shape
    metrics = compute_pde_metrics(preds, target_seq)
    assert metrics["mse"] < 10.0  # Sanity: finite and not huge 