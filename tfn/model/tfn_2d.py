#!/usr/bin/env python3
# pyright: reportMissingImports=false, reportGeneralTypeIssues=false
"""
Token Field Network – 2-D Complete Implementation
================================================
This module provides a full 2-D Token Field Network implementation that mirrors
the 1-D base but operates on an (H×W) lattice.

Public API
----------
• TrainableTFNLayer2D – projects tokens to a Gaussian field, evolves it, samples back.
• TFNClassifier2D      – stacks multiple 2-D layers + classifier head.
• create_tfn2d_variants – convenience factory with common presets.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------------------------------------------------------
# 2-D TFN Layer
# -----------------------------------------------------------------------------


class TrainableTFNLayer2D(nn.Module):
    """A minimal but expressive 2-D extension of the 1-D TrainableTFNLayer.

    Each token emits a Gaussian patch onto an (H×W) lattice. The aggregated field
    is evolved *k* times with a depth-wise convolution and finally re-sampled to
    update the token embeddings. Numerous knobs (multiscale, kernel mixing,
    global context) are provided for research flexibility.
    """

    def __init__(
        self,
        embed_dim: int,
        field_dim: int = 64,
        grid_height: int = 32,
        grid_width: int = 32,
        num_evolution_steps: int = 5,
        kernel_sigma: float = 4.0,
        dropout: float = 0.1,
        channel_mixing: bool = True,
        kernel_size: int = 3,
        learnable_sigma: bool = True,
        learnable_out_sigma: bool = False,
        out_sigma_scale: float = 2.0,
        field_dropout: float = 0.0,
        use_global_context: bool = False,
        use_gate: bool = True,
        multiscale: bool = False,
        kernel_mix: bool = False,
        kernel_mix_scale: float = 2.0,
    ) -> None:
        super().__init__()
        self.E = embed_dim
        self.D = field_dim
        self.H = grid_height
        self.W = grid_width
        self.steps = num_evolution_steps

        # ------------------------------------------------------------------
        # Linear projections token ⇌ field
        # ------------------------------------------------------------------
        self.to_field = nn.Linear(embed_dim, field_dim, bias=False)
        self.from_field = nn.Linear(field_dim, embed_dim, bias=False)

        # ------------------------------------------------------------------
        # Evolution operator (depth-wise conv or channel-mix variant)
        # ------------------------------------------------------------------
        if channel_mixing:
            self.evolve = nn.Sequential(
                nn.Conv2d(field_dim, field_dim * 2, kernel_size=3, padding=1, groups=1),
                nn.ReLU(),
                nn.Conv2d(field_dim * 2, field_dim, kernel_size=1),
            )
        else:
            pad = kernel_size // 2
            self.evolve = nn.Conv2d(
                field_dim, field_dim, kernel_size=kernel_size, padding=pad, groups=field_dim
            )

        # ------------------------------------------------------------------
        # Gaussian width parameters
        # ------------------------------------------------------------------
        self.learnable_sigma = learnable_sigma
        if learnable_sigma:
            self._log_sigma = nn.Parameter(torch.log(torch.tensor(kernel_sigma)))
        else:
            self.register_buffer("sigma_const", torch.tensor(kernel_sigma))

        self.learnable_out_sigma = learnable_out_sigma
        if learnable_out_sigma:
            self._log_sigma_out = nn.Parameter(
                torch.log(torch.tensor(kernel_sigma * out_sigma_scale))
            )
        else:
            self.out_sigma_scale = out_sigma_scale

        # ------------------------------------------------------------------
        # Misc options
        # ------------------------------------------------------------------
        self.field_dropout = field_dropout
        self.dropout = nn.Dropout(dropout)
        self.use_global_context = use_global_context
        self.use_gate = use_global_context and use_gate

        # Multiscale branch
        self.multiscale = multiscale
        if multiscale:
            self.ms_conv = nn.Conv2d(field_dim, field_dim, kernel_size=3, padding=1, groups=field_dim)

        # Two-component kernel mixing
        self.kernel_mix = kernel_mix
        if kernel_mix:
            self.mix_logits = nn.Parameter(torch.zeros(2))
            self.kernel_mix_scale = kernel_mix_scale

        if self.use_gate:
            self.gate_proj = nn.Linear(embed_dim, 1)

        # Pre-compute lattice coordinates: (1,1,H,1) & (1,1,1,W)
        y = torch.arange(self.H).view(1, 1, self.H, 1)
        x = torch.arange(self.W).view(1, 1, 1, self.W)
        self.register_buffer("grid_y", y.float())
        self.register_buffer("grid_x", x.float())

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------
    def forward(
        self,
        tokens: torch.Tensor,  # (B,L,E)
        positions: torch.Tensor,  # (B,L,2) – lattice coords (y,x)
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        B, L, E = tokens.shape
        assert E == self.E, "embed_dim mismatch"
        assert positions.shape == (B, L, 2), "positions must be (B,L,2)"

        sigma_in = (
            F.softplus(self._log_sigma) + 1e-6 if self.learnable_sigma else self.sigma_const
        )
        sigma_out = (
            F.softplus(self._log_sigma_out) + 1e-6
            if self.learnable_out_sigma
            else sigma_in * self.out_sigma_scale
        )

        # ---------------- Projection: tokens ➔ field ----------------
        proj = self.to_field(tokens)
        if self.training and self.field_dropout > 0.0:
            mask = torch.rand(B, L, 1, device=tokens.device) > self.field_dropout
            proj = proj * mask

        pos_y = positions[..., 0].unsqueeze(-1).unsqueeze(-1)  # (B,L,1,1)
        pos_x = positions[..., 1].unsqueeze(-1).unsqueeze(-1)  # (B,L,1,1)
        dy2 = (self.grid_y - pos_y).pow(2)
        dx2 = (self.grid_x - pos_x).pow(2)

        gauss1 = torch.exp(-(dy2 + dx2) / (2 * sigma_in**2))
        if self.kernel_mix:
            gauss2 = torch.exp(-(dy2 + dx2) / (2 * (sigma_in * self.kernel_mix_scale) ** 2))
            w_comp = torch.stack([gauss1, gauss2], dim=0)  # (2,B,L,H,W)
            mix_w = torch.softmax(self.mix_logits, dim=0).view(2, 1, 1, 1, 1)
            weights_in = (mix_w * w_comp).sum(0)
        else:
            weights_in = gauss1

        weights_in = weights_in / (weights_in.sum(dim=(-2, -1), keepdim=True) + 1e-6)
        field = torch.einsum("blc,blhw->bchw", proj, weights_in)  # (B,D,H,W)

        # ---------------- Evolution ----------------
        for _ in range(self.steps):
            field = field + self.evolve(field)
            if self.multiscale:
                pooled = F.avg_pool2d(field, 2)
                up = F.interpolate(self.ms_conv(pooled), scale_factor=2, mode="nearest")
                field = field + up

        # ---------------- Sampling: field ➔ tokens ----------------
        weights_out = torch.exp(-(dy2 + dx2) / (2 * sigma_out**2))
        weights_out = weights_out / (weights_out.sum(dim=(-2, -1), keepdim=True) + 1e-6)
        sampled = torch.einsum("bchw,blhw->blc", field, weights_out)

        if self.use_global_context:
            ctx = field.mean(dim=(-2, -1)).unsqueeze(1).expand(B, L, self.D)
            if self.use_gate:
                gate = torch.sigmoid(self.gate_proj(tokens))
                ctx = ctx * gate
            sampled = sampled + ctx

        tokens_out = self.from_field(sampled)
        tokens_out = self.dropout(tokens_out)
        return tokens_out, {"field": field.detach(), "sigma": float(sigma_in)}


# -----------------------------------------------------------------------------
# 2-D TFN Classifier
# -----------------------------------------------------------------------------


class TFNClassifier2D(nn.Module):
    """A stack of 2-D TFN layers followed by pooling + MLP head."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 256,
        num_classes: int = 4,
        num_layers: int = 4,
        num_evolution_steps: int = 5,
        field_dim: int = 64,
        grid_height: int = 32,
        grid_width: int = 32,
        use_dynamic_positions: bool = False,
        learnable_sigma: bool = True,
        learnable_out_sigma: bool = False,
        out_sigma_scale: float = 2.0,
        field_dropout: float = 0.0,
        use_global_context: bool = False,
        dropout: float = 0.1,
        multiscale: bool = False,
        kernel_mix: bool = False,
        kernel_mix_scale: float = 2.0,
        snake: bool = False,
    ) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        # Positional embeddings (separate for x and y)
        self.pos_embed_x = nn.Embedding(grid_width, embed_dim)
        self.pos_embed_y = nn.Embedding(grid_height, embed_dim)

        self.use_dyn = use_dynamic_positions
        if self.use_dyn:
            self.delta_pred = nn.Sequential(
                nn.LayerNorm(embed_dim),
                nn.Linear(embed_dim, embed_dim),
                nn.ReLU(),
                nn.Linear(embed_dim, 2),
                nn.Tanh(),
            )
        else:
            self.delta_pred = None

        self.layers = nn.ModuleList(
            [
                TrainableTFNLayer2D(
                    embed_dim=embed_dim,
                    field_dim=field_dim,
                    grid_height=grid_height,
                    grid_width=grid_width,
                    channel_mixing=True,
                    num_evolution_steps=num_evolution_steps,
                    learnable_sigma=learnable_sigma,
                    learnable_out_sigma=learnable_out_sigma,
                    out_sigma_scale=out_sigma_scale,
                    field_dropout=field_dropout,
                    use_global_context=use_global_context,
                    multiscale=multiscale,
                    kernel_mix=kernel_mix,
                    kernel_mix_scale=kernel_mix_scale,
                )
                for _ in range(num_layers)
            ]
        )

        self.snake = snake
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, num_classes),
        )
        self.dropout = nn.Dropout(dropout)
        self.H, self.W = grid_height, grid_width

    # ------------------------------------------------------------
    def forward(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Forward pass accepting either integer token indices or pre-embedded inputs."""
        if input_ids.dim() == 3:
            x = input_ids
            B, L, _ = x.shape
            device = x.device
        else:
            B, L = input_ids.shape
            device = input_ids.device
            x = self.embed(input_ids)

        idx = torch.arange(L, device=device)
        row = idx // self.W
        col = idx % self.W
        if self.snake:
            col = torch.where(row % 2 == 0, col, (self.W - 1 - col))
        pos_y = row.float()
        pos_x = col.float()
        pos_y = pos_y.unsqueeze(0).expand(B, -1)
        pos_x = pos_x.unsqueeze(0).expand(B, -1)

        pos_x_idx = pos_x.long(); pos_y_idx = pos_y.long()
        x = x + self.pos_embed_x(pos_x_idx) + self.pos_embed_y(pos_y_idx)
        x = self.dropout(x)

        positions = torch.stack([pos_y, pos_x], dim=-1)

        if self.use_dyn and self.delta_pred is not None:
            delta = self.delta_pred(x)
            pos_y_dyn = (delta[..., 0] + 1) * 0.5 * (self.H - 1)
            pos_x_dyn = (delta[..., 1] + 1) * 0.5 * (self.W - 1)
            positions = torch.stack([pos_y_dyn, pos_x_dyn], dim=-1)
        else:
            positions = torch.stack(
                [positions[..., 0].clamp(0, self.H - 1), positions[..., 1].clamp(0, self.W - 1)],
                dim=-1,
            )

        aux_layers: List[Dict[str, Any]] = []
        for layer in self.layers:
            x, aux = layer(x, positions)
            aux_layers.append(aux)

        pooled = x.mean(dim=1)
        logits = self.classifier(pooled)
        return logits, {"layers": aux_layers}


# -----------------------------------------------------------------------------
# Convenience factory
# -----------------------------------------------------------------------------

def create_tfn2d_variants(
    vocab_size: int,
    num_classes: int,
    embed_dim: int = 256,
    evo_steps: int = 5,
    grid_height: int = 32,
    grid_width: int = 32,
    learnable_sigma: bool = True,
    learnable_out_sigma: bool = False,
    out_sigma_scale: float = 2.0,
    field_dropout: float = 0.0,
    global_context: bool = False,
    multiscale: bool = False,
    kernel_mix: bool = False,
    kernel_mix_scale: float = 2.0,
    snake: bool = False,
) -> Dict[str, nn.Module]:
    """Return a dict of common 2-D TFN variants ready for experimentation."""
    base_args = dict(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        num_classes=num_classes,
        num_layers=4,
        field_dim=64,
        grid_height=grid_height,
        grid_width=grid_width,
        num_evolution_steps=evo_steps,
        learnable_sigma=learnable_sigma,
        learnable_out_sigma=learnable_out_sigma,
        out_sigma_scale=out_sigma_scale,
        field_dropout=field_dropout,
        use_global_context=global_context,
        dropout=0.1,
        multiscale=multiscale,
        kernel_mix=kernel_mix,
        kernel_mix_scale=kernel_mix_scale,
        snake=snake,
    )

    return {
        "tfn2d_static": TFNClassifier2D(**base_args, use_dynamic_positions=False),
        "tfn2d_dynamic": TFNClassifier2D(**base_args, use_dynamic_positions=True),
    }


# -----------------------------------------------------------------------------
__all__ = [
    "TrainableTFNLayer2D",
    "TFNClassifier2D",
    "create_tfn2d_variants",
] 