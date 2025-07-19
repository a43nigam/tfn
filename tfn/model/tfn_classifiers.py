"""
TFN Classifier and Regressor Models

Clean, reusable model classes for sequence classification and regression tasks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .tfn_base import TrainableTFNLayer


class TFNClassifier(nn.Module):
    """Simple classifier using TFN layers."""
    
    def __init__(self, vocab_size: int, embed_dim: int, num_classes: int, 
                 num_layers: int = 2, kernel_type: str = "rbf", evolution_type: str = "cnn",
                 grid_size: int = 100, time_steps: int = 3, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        
        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        
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
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, num_classes)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        nn.init.normal_(self.token_embedding.weight, 0, 0.1)
        
        for module in self.classifier:
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.1)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass of TFN classifier.
        
        Args:
            input_ids: [B, N] token indices
            positions: [B, N, 1] token positions (optional, auto-generated if None)
        
        Returns:
            logits: [B, num_classes] classification logits
        """
        B, N = input_ids.shape
        
        # Get token embeddings
        embeddings = self.token_embedding(input_ids)  # [B, N, embed_dim]
        
        # Generate positions if not provided
        if positions is None:
            positions = torch.linspace(0.1, 0.9, N, device=input_ids.device)
            positions = positions.unsqueeze(0).unsqueeze(-1).expand(B, -1, -1)
        
        # Apply TFN layers
        x = embeddings
        for tfn_layer in self.tfn_layers:
            x = tfn_layer(x, positions)
        
        # Global average pooling
        pooled = x.mean(dim=1)  # [B, embed_dim]
        
        # Classification
        logits = self.classifier(pooled)  # [B, num_classes]
        
        return logits


class TFNRegressor(nn.Module):
    """Simple regressor using TFN layers."""
    
    def __init__(self, input_dim: int, embed_dim: int, output_dim: int,
                 num_layers: int = 2, kernel_type: str = "rbf", evolution_type: str = "cnn",
                 grid_size: int = 100, time_steps: int = 3, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        
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
        
        # Output projection
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
    
    def forward(self, x: torch.Tensor, positions: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass of TFN regressor.
        
        Args:
            x: [B, N, input_dim] input features
            positions: [B, N, 1] positions (optional, auto-generated if None)
        
        Returns:
            output: [B, N, output_dim] or [B, output_dim] depending on use
        """
        B, N, input_dim = x.shape
        
        # Project to embedding dimension
        embeddings = self.input_proj(x)  # [B, N, embed_dim]
        
        # Generate positions if not provided
        if positions is None:
            positions = torch.linspace(0.1, 0.9, N, device=x.device)
            positions = positions.unsqueeze(0).unsqueeze(-1).expand(B, -1, -1)
        
        # Apply TFN layers
        for tfn_layer in self.tfn_layers:
            embeddings = tfn_layer(embeddings, positions)
        
        # Output projection
        output = self.output_proj(embeddings)  # [B, N, output_dim]
        
        return output 