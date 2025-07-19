"""
TFN Regressor Models for Time Series Forecasting

Specialized regressor models for time series forecasting tasks like ETT and Jena Climate.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from .tfn_base import TrainableTFNLayer


class TFNTimeSeriesRegressor(nn.Module):
    """
    TFN Regressor specifically designed for time series forecasting.
    
    This model is optimized for forecasting tasks where we need to predict
    future values based on historical sequences.
    """
    
    def __init__(self, 
                 input_dim: int,
                 embed_dim: int = 128,
                 output_len: int = 1,
                 num_layers: int = 2,
                 kernel_type: str = "rbf",
                 evolution_type: str = "cnn",
                 grid_size: int = 64,
                 time_steps: int = 3,
                 dropout: float = 0.1,
                 use_global_pooling: bool = True):
        """
        Initialize TFN Time Series Regressor.
        
        Args:
            input_dim: Dimension of input features
            embed_dim: Embedding dimension for TFN layers
            output_len: Number of future time steps to predict
            num_layers: Number of TFN layers
            kernel_type: Type of kernel for field projection
            evolution_type: Type of field evolution
            grid_size: Size of the spatial grid
            time_steps: Number of evolution time steps
            dropout: Dropout rate
            use_global_pooling: Whether to use global pooling or sequence output
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.output_len = output_len
        self.use_global_pooling = use_global_pooling
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, embed_dim)
        
        # TFN layers
        self.tfn_layers = nn.ModuleList([
            TrainableTFNLayer(
                embed_dim=embed_dim,
                kernel_type=kernel_type,
                evolution_type=evolution_type,
                grid_size=grid_size,
                time_steps=time_steps,
                dropout=dropout
            ) for _ in range(num_layers)
        ])
        
        # Output projection for forecasting
        if use_global_pooling:
            # Global pooling approach: pool sequence and predict all future steps
            self.output_proj = nn.Sequential(
                nn.Linear(embed_dim, embed_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(embed_dim // 2, output_len)
            )
        else:
            # Sequence approach: predict each future step from sequence
            self.output_proj = nn.Sequential(
                nn.Linear(embed_dim, embed_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(embed_dim // 2, output_len)
            )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        nn.init.normal_(self.input_proj.weight, 0, 0.1)
        nn.init.zeros_(self.input_proj.bias)
        
        for module in self.output_proj:
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.1)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor, positions: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for time series forecasting.
        
        Args:
            x: Input sequence of shape [B, seq_len, input_dim]
            positions: Token positions of shape [B, seq_len, 1] (optional)
        
        Returns:
            predictions: Forecasted values of shape [B, output_len]
        """
        B, seq_len, input_dim = x.shape
        
        # Project to embedding dimension
        embeddings = self.input_proj(x)  # [B, seq_len, embed_dim]
        
        # Generate positions if not provided
        if positions is None:
            positions = torch.linspace(0.1, 0.9, seq_len, device=x.device)
            positions = positions.unsqueeze(0).unsqueeze(-1).expand(B, -1, -1)
        
        # Apply TFN layers
        for tfn_layer in self.tfn_layers:
            embeddings = tfn_layer(embeddings, positions)
        
        if self.use_global_pooling:
            # Global pooling approach: average across sequence
            pooled = embeddings.mean(dim=1)  # [B, embed_dim]
            predictions = self.output_proj(pooled)  # [B, output_len]
        else:
            # Sequence approach: use last token for prediction
            last_token = embeddings[:, -1, :]  # [B, embed_dim]
            predictions = self.output_proj(last_token)  # [B, output_len]
        
        return predictions


class TFNMultiStepRegressor(nn.Module):
    """
    TFN Regressor for multi-step forecasting with sequence output.
    
    This model predicts the entire future sequence rather than just
    a single value, useful for longer-term forecasting.
    """
    
    def __init__(self,
                 input_dim: int,
                 embed_dim: int = 128,
                 output_len: int = 1,
                 num_layers: int = 2,
                 kernel_type: str = "rbf",
                 evolution_type: str = "cnn",
                 grid_size: int = 64,
                 time_steps: int = 3,
                 dropout: float = 0.1):
        """
        Initialize TFN Multi-Step Regressor.
        
        Args:
            input_dim: Dimension of input features
            embed_dim: Embedding dimension for TFN layers
            output_len: Number of future time steps to predict
            num_layers: Number of TFN layers
            kernel_type: Type of kernel for field projection
            evolution_type: Type of field evolution
            grid_size: Size of the spatial grid
            time_steps: Number of evolution time steps
            dropout: Dropout rate
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.output_len = output_len
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, embed_dim)
        
        # TFN layers
        self.tfn_layers = nn.ModuleList([
            TrainableTFNLayer(
                embed_dim=embed_dim,
                kernel_type=kernel_type,
                evolution_type=evolution_type,
                grid_size=grid_size,
                time_steps=time_steps,
                dropout=dropout
            ) for _ in range(num_layers)
        ])
        
        # Output projection for multi-step forecasting
        self.output_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, output_len)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        nn.init.normal_(self.input_proj.weight, 0, 0.1)
        nn.init.zeros_(self.input_proj.bias)
        
        for module in self.output_proj:
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.1)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor, positions: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for multi-step forecasting.
        
        Args:
            x: Input sequence of shape [B, seq_len, input_dim]
            positions: Token positions of shape [B, seq_len, 1] (optional)
        
        Returns:
            predictions: Forecasted sequence of shape [B, output_len]
        """
        B, seq_len, input_dim = x.shape
        
        # Project to embedding dimension
        embeddings = self.input_proj(x)  # [B, seq_len, embed_dim]
        
        # Generate positions if not provided
        if positions is None:
            positions = torch.linspace(0.1, 0.9, seq_len, device=x.device)
            positions = positions.unsqueeze(0).unsqueeze(-1).expand(B, -1, -1)
        
        # Apply TFN layers
        for tfn_layer in self.tfn_layers:
            embeddings = tfn_layer(embeddings, positions)
        
        # Use the last token for multi-step prediction
        last_token = embeddings[:, -1, :]  # [B, embed_dim]
        predictions = self.output_proj(last_token)  # [B, output_len]
        
        return predictions


class TFNSequenceRegressor(nn.Module):
    """
    TFN Regressor that outputs predictions for each input position.
    
    This model is useful for tasks where we need to predict values
    at each time step of the input sequence.
    """
    
    def __init__(self,
                 input_dim: int,
                 embed_dim: int = 128,
                 output_dim: int = 1,
                 num_layers: int = 2,
                 kernel_type: str = "rbf",
                 evolution_type: str = "cnn",
                 grid_size: int = 64,
                 time_steps: int = 3,
                 dropout: float = 0.1):
        """
        Initialize TFN Sequence Regressor.
        
        Args:
            input_dim: Dimension of input features
            embed_dim: Embedding dimension for TFN layers
            output_dim: Dimension of output features
            num_layers: Number of TFN layers
            kernel_type: Type of kernel for field projection
            evolution_type: Type of field evolution
            grid_size: Size of the spatial grid
            time_steps: Number of evolution time steps
            dropout: Dropout rate
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.output_dim = output_dim
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, embed_dim)
        
        # TFN layers
        self.tfn_layers = nn.ModuleList([
            TrainableTFNLayer(
                embed_dim=embed_dim,
                kernel_type=kernel_type,
                evolution_type=evolution_type,
                grid_size=grid_size,
                time_steps=time_steps,
                dropout=dropout
            ) for _ in range(num_layers)
        ])
        
        # Output projection for sequence prediction
        self.output_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, output_dim)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        nn.init.normal_(self.input_proj.weight, 0, 0.1)
        nn.init.zeros_(self.input_proj.bias)
        
        for module in self.output_proj:
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.1)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor, positions: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for sequence prediction.
        
        Args:
            x: Input sequence of shape [B, seq_len, input_dim]
            positions: Token positions of shape [B, seq_len, 1] (optional)
        
        Returns:
            predictions: Predictions for each position of shape [B, seq_len, output_dim]
        """
        B, seq_len, input_dim = x.shape
        
        # Project to embedding dimension
        embeddings = self.input_proj(x)  # [B, seq_len, embed_dim]
        
        # Generate positions if not provided
        if positions is None:
            positions = torch.linspace(0.1, 0.9, seq_len, device=x.device)
            positions = positions.unsqueeze(0).unsqueeze(-1).expand(B, -1, -1)
        
        # Apply TFN layers
        for tfn_layer in self.tfn_layers:
            embeddings = tfn_layer(embeddings, positions)
        
        # Predict for each position
        predictions = self.output_proj(embeddings)  # [B, seq_len, output_dim]
        
        return predictions 