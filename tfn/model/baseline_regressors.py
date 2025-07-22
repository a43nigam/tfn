from __future__ import annotations

"""baseline_regressors.py
Baseline regressor models for comparison with TFN.

Includes:
1. TransformerRegressor - Standard Transformer encoder for regression
2. PerformerRegressor - Linear attention approximation for regression
3. LSTMRegressor - LSTM-based regressor
4. CNNRegressor - CNN-based regressor
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Shared components -----------------------------------------------------------
from .shared_layers import LearnedPositionalEmbeddings, LinearAttention


class TransformerRegressor(nn.Module):
    """Standard Transformer encoder for regression tasks."""

    def __init__(
        self,
        input_dim: int,
        embed_dim: int = 128,
        output_dim: int = 1,
        seq_len: int = 512,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.input_proj = nn.Linear(input_dim, embed_dim)
        self.pos = LearnedPositionalEmbeddings(seq_len, embed_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Regression head
        self.regressor = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: [B, L, input_dim]
        h = self.input_proj(x) + self.pos(x.size(1))
        h = self.transformer(h)
        
        # Use first token ([CLS] position) for regression
        pooled = h[:, 0, :]  # [B, embed_dim]
        
        # Regression
        output = self.regressor(pooled)  # [B, output_dim]
        return output


class PerformerRegressor(nn.Module):
    """Performer-style linear attention regressor."""

    def __init__(
        self,
        input_dim: int,
        embed_dim: int = 128,
        output_dim: int = 1,
        seq_len: int = 512,
        num_layers: int = 2,
        proj_dim: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_dim, embed_dim)
        self.pos = LearnedPositionalEmbeddings(seq_len, embed_dim)
        
        self.layers = nn.ModuleList(
            [
                nn.Sequential(
                    LinearAttention(embed_dim, proj_dim=proj_dim),
                    nn.LayerNorm(embed_dim),
                    nn.Linear(embed_dim, embed_dim * 4),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(embed_dim * 4, embed_dim),
                    nn.Dropout(dropout),
                )
                for _ in range(num_layers)
            ]
        )
        
        # Regression head
        self.regressor = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: [B, L, input_dim]
        h = self.input_proj(x) + self.pos(x.size(1))
        for layer in self.layers:
            residual = h
            h = layer(h)
            h = h + residual
        
        # Use first token ([CLS] position) for regression
        pooled = h[:, 0, :]  # [B, embed_dim]
        
        # Regression
        output = self.regressor(pooled)  # [B, output_dim]
        return output


class LSTMRegressor(nn.Module):
    """LSTM-based regressor."""

    def __init__(
        self,
        input_dim: int,
        embed_dim: int = 128,
        output_dim: int = 1,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        bidirectional: bool = True,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_dim, embed_dim)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True,
        )
        
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        
        # Regression head
        self.regressor = nn.Sequential(
            nn.Linear(lstm_output_dim, lstm_output_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_output_dim // 2, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: [B, L, input_dim]
        h = self.input_proj(x)
        lstm_out, (hidden, cell) = self.lstm(h)
        
        # Use last hidden state for regression
        if self.lstm.bidirectional:
            # Concatenate forward and backward hidden states
            last_hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        else:
            last_hidden = hidden[-1]
        
        # Regression
        output = self.regressor(last_hidden)  # [B, output_dim]
        return output


class CNNRegressor(nn.Module):
    """CNN-based regressor."""

    def __init__(
        self,
        input_dim: int,
        embed_dim: int = 128,
        output_dim: int = 1,
        num_filters: int = 128,
        filter_sizes: list = [3, 4, 5],
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_dim, embed_dim)
        
        # Convolutional layers
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, num_filters, kernel_size=k)
            for k in filter_sizes
        ])
        
        # Regression head
        total_filters = num_filters * len(filter_sizes)
        self.regressor = nn.Sequential(
            nn.Linear(total_filters, total_filters // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(total_filters // 2, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: [B, L, input_dim]
        h = self.input_proj(x)  # [B, L, embed_dim]
        h = h.transpose(1, 2)  # [B, embed_dim, L]
        
        # Apply convolutions
        conv_outputs = []
        for conv in self.convs:
            conv_out = F.relu(conv(h))  # [B, num_filters, L-k+1]
            # Global max pooling
            pooled = F.max_pool1d(conv_out, conv_out.size(2))  # [B, num_filters, 1]
            conv_outputs.append(pooled.squeeze(2))  # [B, num_filters]
        
        # Concatenate all conv outputs
        concatenated = torch.cat(conv_outputs, dim=1)  # [B, total_filters]
        
        # Regression
        output = self.regressor(concatenated)  # [B, output_dim]
        return output 