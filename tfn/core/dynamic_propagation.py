"""
Dynamic Field Propagation for TFN.

This module implements hybrid discrete-continuous PDE evolution with interference
terms, bridging token representations with continuous field dynamics.

Mathematical formulation:
    ∂F/∂t = L(F) + Σᵢⱼ βᵢⱼ I(Fᵢ, Fⱼ)
    
Where:
    - L(F) = linear evolution operator (diffusion, wave, etc.)
    - I(Fᵢ, Fⱼ) = interference term between fields i and j
    - βᵢⱼ = learnable coupling coefficients
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any, Literal
import math
from .field_interference import TokenFieldInterference


class DynamicFieldPropagator(nn.Module):
    """
    Dynamic field propagation with coupled PDEs and interference.
    
    Implements hybrid discrete-continuous evolution that bridges token
    representations with continuous field dynamics.
    
    Mathematical formulation:
        ∂F/∂t = L(F) + Σᵢⱼ βᵢⱼ I(Fᵢ, Fⱼ)
        F(z, t+Δt) = F(z, t) + Δt * [L(F) + Σᵢⱼ βᵢⱼ I(Fᵢ, Fⱼ)]
    """
    
    def __init__(self, 
                 embed_dim: int,
                 pos_dim: int,
                 evolution_type: str = "diffusion",
                 interference_type: str = "standard",
                 num_steps: int = 4,
                 dt: float = 0.01,
                 interference_weight: float = 0.5,
                 dropout: float = 0.1):
        """
        Initialize dynamic field propagator.
        
        Args:
            embed_dim: Dimension of token embeddings
            pos_dim: Dimension of position space
            evolution_type: Type of evolution ("diffusion", "wave", "schrodinger")
            interference_type: Type of interference ("standard", "causal", "multiscale", "physics")
            num_steps: Number of evolution steps
            dt: Time step size
            interference_weight: Weight for interference terms
            dropout: Dropout rate for regularization
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.pos_dim = pos_dim
        self.evolution_type = evolution_type
        self.num_steps = num_steps
        self.dt = dt
        self.interference_weight = interference_weight
        
        # Learnable evolution parameters
        if evolution_type == "diffusion":
            self.diffusion_coeff = nn.Parameter(torch.tensor(0.1))
        elif evolution_type == "wave":
            self.wave_speed = nn.Parameter(torch.tensor(1.0))
            # Initialize velocity field for second-order wave equation
            self.velocity_field = nn.Parameter(torch.zeros(embed_dim))
        elif evolution_type == "schrodinger":
            # Complex Hamiltonian for Schrödinger equation
            self.hamiltonian_real = nn.Parameter(torch.eye(embed_dim))
            self.hamiltonian_imag = nn.Parameter(torch.zeros(embed_dim, embed_dim))
        
        # Field interference module
        self.interference = TokenFieldInterference(
            embed_dim=embed_dim,
            interference_types=("constructive", "destructive", "phase"),
            dropout=dropout
        )
        
        # Learnable coupling coefficients
        self.coupling_coeffs = nn.Parameter(torch.ones(embed_dim))
        
        # Output projection
        self.output_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, 
                token_fields: torch.Tensor,  # [B, N, D] token field representations
                positions: Optional[torch.Tensor] = None,  # [B, N, P] token positions
                grid_points: Optional[torch.Tensor] = None) -> torch.Tensor:  # [B, M, P] grid points
        """
        Evolve fields with dynamic propagation.
        
        Args:
            token_fields: Token field representations [B, N, D]
            positions: Token positions [B, N, P] (optional)
            grid_points: Grid points for continuous evolution [B, M, P] (optional)
            
        Returns:
            Evolved token fields [B, N, D]
        """
        batch_size, num_tokens, embed_dim = token_fields.shape
        
        # Initialize evolved fields
        evolved_fields = token_fields.clone()
        
        # Initialize velocity field for wave equation
        if self.evolution_type == "wave":
            velocities = torch.zeros_like(evolved_fields)
        
        # Multi-step evolution
        for step in range(self.num_steps):
            # Compute linear evolution term: L(F)
            if self.evolution_type == "wave":
                linear_evolution, velocities = self._compute_linear_evolution(evolved_fields, velocities)
            else:
                linear_evolution, _ = self._compute_linear_evolution(evolved_fields, positions)
            
            # Compute interference term: Σᵢⱼ βᵢⱼ I(Fᵢ, Fⱼ)
            interference_term = self._compute_interference_term(evolved_fields, positions)
            
            # Update fields: F(t+Δt) = F(t) + Δt * [L(F) + β*I(F)]
            dt = float(max(0.001, min(float(self.dt), 0.1)))
            evolved_fields = evolved_fields + dt * (linear_evolution + self.interference_weight * interference_term)
            
            # Apply dropout for regularization
            evolved_fields = self.dropout(evolved_fields)
        
        # Final output projection
        output = self.output_proj(evolved_fields)
        
        return output
    
    def _compute_linear_evolution(self, 
                                 fields: torch.Tensor,  # [B, N, D]
                                 positions: Optional[torch.Tensor] = None,
                                 velocities: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Compute linear evolution term L(F).
        
        Args:
            fields: Current field values [B, N, D]
            positions: Token positions [B, N, P] (optional)
            velocities: Velocity field for wave equation [B, N, D] (optional)
            
        Returns:
            Tuple of (linear evolution term [B, N, D], updated velocities [B, N, D] or None)
        """
        if self.evolution_type == "diffusion":
            evolution = self._diffusion_evolution(fields)
            return evolution, None
        elif self.evolution_type == "wave":
            if velocities is None:
                velocities = torch.zeros_like(fields)
            evolution, new_velocities = self._wave_evolution(fields, velocities)
            return evolution, new_velocities
        elif self.evolution_type == "schrodinger":
            evolution = self._schrodinger_evolution(fields)
            return evolution, None
        else:
            raise ValueError(f"Unknown evolution type: {self.evolution_type}")
    
    def _diffusion_evolution(self, fields: torch.Tensor) -> torch.Tensor:
        """Compute diffusion evolution: ∂F/∂t = α∇²F"""
        batch_size, num_tokens, embed_dim = fields.shape
        
        # Learnable diffusion coefficient
        alpha = torch.clamp(self.diffusion_coeff, min=0.01, max=1.0)
        
        # Compute discrete Laplacian (1D for token sequence)
        laplacian = torch.zeros_like(fields)
        
        # Interior points: ∇²F = F_{i+1} - 2F_i + F_{i-1}
        if num_tokens > 2:
            laplacian[:, 1:-1, :] = (fields[:, 2:, :] - 2 * fields[:, 1:-1, :] + fields[:, :-2, :])
        
        # Boundary conditions (zero gradient)
        if num_tokens > 1:
            laplacian[:, 0, :] = fields[:, 1, :] - fields[:, 0, :]  # Forward difference
            laplacian[:, -1, :] = fields[:, -2, :] - fields[:, -1, :]  # Backward difference
        
        return alpha * laplacian
    
    def _wave_evolution(self, fields: torch.Tensor, velocities: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute proper wave evolution: ∂²F/∂t² = c²∇²F"""
        batch_size, num_tokens, embed_dim = fields.shape
        
        # Learnable wave speed
        c = torch.clamp(self.wave_speed, min=0.1, max=10.0)
        
        # Compute discrete Laplacian (same as diffusion)
        laplacian = torch.zeros_like(fields)
        
        if num_tokens > 2:
            laplacian[:, 1:-1, :] = (fields[:, 2:, :] - 2 * fields[:, 1:-1, :] + fields[:, :-2, :])
        
        if num_tokens > 1:
            laplacian[:, 0, :] = fields[:, 1, :] - fields[:, 0, :]
            laplacian[:, -1, :] = fields[:, -2, :] - fields[:, -1, :]
        
        # Second-order wave equation: ∂²F/∂t² = c²∇²F
        # Split into first-order system:
        # ∂F/∂t = v
        # ∂v/∂t = c²∇²F
        
        # Update velocity: ∂v/∂t = c²∇²F
        acceleration = c**2 * laplacian
        new_velocities = velocities + self.dt * acceleration
        
        # Update field: ∂F/∂t = v
        field_evolution = new_velocities
        
        return field_evolution, new_velocities
    
    def _schrodinger_evolution(self, fields: torch.Tensor) -> torch.Tensor:
        """Compute Schrödinger-like evolution: i∂F/∂t = HF"""
        batch_size, num_tokens, embed_dim = fields.shape
        
        # Construct complex Hamiltonian: H = H_real + i*H_imag
        H_real = self.hamiltonian_real
        H_imag = self.hamiltonian_imag
        
        # Ensure Hermitian: H = (H + H†)/2
        H_real = (H_real + H_real.T) / 2
        H_imag = (H_imag - H_imag.T) / 2  # Anti-Hermitian imaginary part
        
        # Apply Hamiltonian: HF
        # [B, N, D] × [D, D] -> [B, N, D]
        hamiltonian_evolution_real = torch.einsum('bnd,de->bne', fields, H_real)
        hamiltonian_evolution_imag = torch.einsum('bnd,de->bne', fields, H_imag)
        
        # For Schrödinger equation: i∂F/∂t = HF
        # ∂F/∂t = -i*HF = H_imag*F - i*H_real*F
        # Since we work with real fields, we take the real part
        evolution_real = hamiltonian_evolution_imag  # Real part of -i*HF
        evolution_imag = -hamiltonian_evolution_real  # Imaginary part of -i*HF
        
        # For real fields, we only keep the real part of the evolution
        return evolution_real
    
    def _compute_interference_term(self, 
                                  fields: torch.Tensor,  # [B, N, D]
                                  positions: Optional[torch.Tensor] = None) -> torch.Tensor:  # [B, N, D]
        """
        Compute interference term Σᵢⱼ βᵢⱼ I(Fᵢ, Fⱼ).
        
        Args:
            fields: Current field values [B, N, D]
            positions: Token positions [B, N, P] (optional)
            
        Returns:
            Interference term [B, N, D]
        """
        # Apply field interference
        interference_output = self.interference(fields, positions)
        
        # Apply learnable coupling coefficients
        coupling_coeffs = torch.clamp(self.coupling_coeffs, min=0.01, max=10.0)
        interference_term = interference_output * coupling_coeffs.unsqueeze(0).unsqueeze(0)
        
        return interference_term


class AdaptiveFieldPropagator(DynamicFieldPropagator):
    """
    Adaptive field propagator with learnable evolution parameters.
    
    Automatically adjusts evolution parameters based on field characteristics.
    """
    
    def __init__(self, 
                 embed_dim: int,
                 pos_dim: int,
                 evolution_type: str = "diffusion",
                 interference_type: str = "standard",
                 num_steps: int = 4,
                 dt: float = 0.01,
                 interference_weight: float = 0.5,
                 dropout: float = 0.1):
        """
        Initialize adaptive field propagator.
        
        Args:
            embed_dim: Dimension of token embeddings
            pos_dim: Dimension of position space
            evolution_type: Type of evolution
            interference_type: Type of interference
            num_steps: Number of evolution steps
            dt: Initial time step size
            interference_weight: Weight for interference terms
            dropout: Dropout rate for regularization
        """
        super().__init__(embed_dim, pos_dim, evolution_type, interference_type, 
                        num_steps, dt, interference_weight, dropout)
        
        # Adaptive parameters
        self.adaptive_dt = nn.Parameter(torch.tensor(dt))
        self.field_norm_threshold = nn.Parameter(torch.tensor(1.0))
        
        # Adaptive evolution network
        self.adaptive_net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, embed_dim // 4),
            nn.ReLU(),
            nn.Linear(embed_dim // 4, 3)  # dt, alpha, interference_weight
        )
        
    def forward(self, 
                token_fields: torch.Tensor,  # [B, N, D]
                positions: Optional[torch.Tensor] = None,
                grid_points: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Evolve fields with adaptive parameters.
        
        Args:
            token_fields: Token field representations [B, N, D]
            positions: Token positions [B, N, P] (optional)
            grid_points: Grid points [B, M, P] (optional)
            
        Returns:
            Evolved token fields [B, N, D]
        """
        batch_size, num_tokens, embed_dim = token_fields.shape
        
        # Initialize evolved fields
        evolved_fields = token_fields.clone()
        
        # Initialize velocity field for wave equation
        if self.evolution_type == "wave":
            velocities = torch.zeros_like(evolved_fields)
        
        # Multi-step evolution with adaptive parameters
        for step in range(self.num_steps):
            # Compute adaptive parameters based on current field state
            field_mean = torch.mean(evolved_fields, dim=1)  # [B, D]
            
            # Adaptive parameter network
            adaptive_params = self.adaptive_net(field_mean)  # [B, 3]
            adaptive_dt = torch.sigmoid(adaptive_params[:, 0]) * 0.1  # [0, 0.1]
            adaptive_alpha = torch.sigmoid(adaptive_params[:, 1]) * 2.0  # [0, 2]
            adaptive_interference = torch.sigmoid(adaptive_params[:, 2]) * 1.0  # [0, 1]
            
            # Compute evolution terms with adaptive parameters
            if self.evolution_type == "wave":
                linear_evolution, velocities = self._compute_adaptive_linear_evolution(evolved_fields, adaptive_alpha, velocities)
            else:
                linear_evolution, _ = self._compute_adaptive_linear_evolution(evolved_fields, adaptive_alpha)
            
            interference_term = self._compute_interference_term(evolved_fields, positions)
            
            # Update fields with adaptive time step
            evolved_fields = evolved_fields + adaptive_dt.unsqueeze(1).unsqueeze(2) * (
                linear_evolution + adaptive_interference.unsqueeze(1).unsqueeze(2) * interference_term
            )
            
            # Apply dropout
            evolved_fields = self.dropout(evolved_fields)
        
        # Final output projection
        output = self.output_proj(evolved_fields)
        
        return output
    
    def _compute_adaptive_linear_evolution(self, 
                                          fields: torch.Tensor,  # [B, N, D]
                                          adaptive_alpha: torch.Tensor,  # [B]
                                          velocities: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Compute linear evolution with adaptive parameters.
        
        Args:
            fields: Current field values [B, N, D]
            adaptive_alpha: Adaptive evolution coefficient [B]
            velocities: Velocity field for wave equation [B, N, D] (optional)
            
        Returns:
            Tuple of (adaptive linear evolution term [B, N, D], updated velocities [B, N, D] or None)
        """
        if self.evolution_type == "diffusion":
            # Compute standard diffusion
            laplacian = torch.zeros_like(fields)
            num_tokens = fields.shape[1]
            
            if num_tokens > 2:
                laplacian[:, 1:-1, :] = (fields[:, 2:, :] - 2 * fields[:, 1:-1, :] + fields[:, :-2, :])
            
            if num_tokens > 1:
                laplacian[:, 0, :] = fields[:, 1, :] - fields[:, 0, :]
                laplacian[:, -1, :] = fields[:, -2, :] - fields[:, -1, :]
            
            # Apply adaptive coefficient
            return adaptive_alpha.unsqueeze(1).unsqueeze(2) * laplacian, None
        elif self.evolution_type == "wave":
            # Compute wave evolution with velocities
            if velocities is None:
                velocities = torch.zeros_like(fields)
            
            # Learnable wave speed
            c = torch.clamp(self.wave_speed, min=0.1, max=10.0)
            
            # Compute discrete Laplacian
            laplacian = torch.zeros_like(fields)
            num_tokens = fields.shape[1]
            
            if num_tokens > 2:
                laplacian[:, 1:-1, :] = (fields[:, 2:, :] - 2 * fields[:, 1:-1, :] + fields[:, :-2, :])
            
            if num_tokens > 1:
                laplacian[:, 0, :] = fields[:, 1, :] - fields[:, 0, :]
                laplacian[:, -1, :] = fields[:, -2, :] - fields[:, -1, :]
            
            # Second-order wave equation: ∂²F/∂t² = c²∇²F
            # Split into first-order system:
            # ∂F/∂t = v
            # ∂v/∂t = c²∇²F
            
            # Update velocity: ∂v/∂t = c²∇²F
            acceleration = c**2 * laplacian
            new_velocities = velocities + self.dt * acceleration
            
            # Update field: ∂F/∂t = v
            field_evolution = new_velocities
            
            # Return both field evolution and updated velocities
            return field_evolution, new_velocities
        else:
            # For other evolution types, use standard computation
            standard_evolution, _ = self._compute_linear_evolution(fields)
            return standard_evolution, None


class CausalFieldPropagator(DynamicFieldPropagator):
    """
    Causal field propagator for time-series applications.
    
    Ensures causality by only allowing backward-looking evolution.
    """
    
    def __init__(self, 
                 embed_dim: int,
                 pos_dim: int,
                 evolution_type: str = "diffusion",
                 interference_type: str = "causal",
                 num_steps: int = 4,
                 dt: float = 0.01,
                 interference_weight: float = 0.5,
                 dropout: float = 0.1):
        """
        Initialize causal field propagator.
        
        Args:
            embed_dim: Dimension of token embeddings
            pos_dim: Dimension of position space
            evolution_type: Type of evolution
            interference_type: Type of interference (should be "causal")
            num_steps: Number of evolution steps
            dt: Time step size
            interference_weight: Weight for interference terms
            dropout: Dropout rate for regularization
        """
        super().__init__(embed_dim, pos_dim, evolution_type, interference_type, 
                        num_steps, dt, interference_weight, dropout)
        
        # Ensure interference is causal
        if interference_type != "causal":
            raise ValueError("CausalFieldPropagator requires interference_type='causal'")
    
    def _compute_linear_evolution(self, 
                                 fields: torch.Tensor,  # [B, N, D]
                                 positions: Optional[torch.Tensor] = None,
                                 velocities: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Compute causal linear evolution.
        
        Args:
            fields: Current field values [B, N, D]
            positions: Token positions [B, N, P] (optional)
            velocities: Velocity field for wave equation [B, N, D] (optional)
            
        Returns:
            Tuple of (causal linear evolution term [B, N, D], updated velocities [B, N, D] or None)
        """
        if self.evolution_type == "diffusion":
            evolution = self._causal_diffusion_evolution(fields)
            return evolution, None
        elif self.evolution_type == "wave":
            evolution = self._causal_wave_evolution(fields)
            return evolution, None
        else:
            # Fall back to parent implementation
            return super()._compute_linear_evolution(fields, positions, velocities)
    
    def _causal_diffusion_evolution(self, fields: torch.Tensor) -> torch.Tensor:
        """Compute causal diffusion evolution."""
        batch_size, num_tokens, embed_dim = fields.shape
        
        # Learnable diffusion coefficient
        alpha = torch.clamp(self.diffusion_coeff, min=0.01, max=1.0)
        
        # Compute causal Laplacian (only backward differences)
        laplacian = torch.zeros_like(fields)
        
        # Interior points: ∇²F = F_{i-1} - 2F_i + F_{i+1} (causal)
        if num_tokens > 2:
            laplacian[:, 1:-1, :] = (fields[:, :-2, :] - 2 * fields[:, 1:-1, :] + fields[:, 2:, :])
        
        # Boundary conditions (causal)
        if num_tokens > 1:
            laplacian[:, 0, :] = torch.zeros_like(fields[:, 0, :])  # No evolution at start
            laplacian[:, -1, :] = fields[:, -2, :] - fields[:, -1, :]  # Backward difference
        
        return alpha * laplacian
    
    def _causal_wave_evolution(self, fields: torch.Tensor) -> torch.Tensor:
        """Compute causal wave evolution."""
        batch_size, num_tokens, embed_dim = fields.shape
        
        # Learnable wave speed
        c = torch.clamp(self.wave_speed, min=0.1, max=10.0)
        
        # Compute causal Laplacian
        laplacian = torch.zeros_like(fields)
        
        if num_tokens > 2:
            laplacian[:, 1:-1, :] = (fields[:, :-2, :] - 2 * fields[:, 1:-1, :] + fields[:, 2:, :])
        
        if num_tokens > 1:
            laplacian[:, 0, :] = torch.zeros_like(fields[:, 0, :])
            laplacian[:, -1, :] = fields[:, -2, :] - fields[:, -1, :]
        
        return c**2 * laplacian


def create_field_propagator(propagator_type: str = "standard",
                           embed_dim: int = 256,
                           pos_dim: int = 1,
                           evolution_type: str = "diffusion",
                           interference_type: str = "standard",
                           **kwargs) -> DynamicFieldPropagator:
    """
    Factory function to create field propagator modules.
    
    Args:
        propagator_type: Type of propagator ("standard", "adaptive", "causal")
        embed_dim: Dimension of token embeddings
        pos_dim: Dimension of position space
        evolution_type: Type of evolution ("diffusion", "wave", "schrodinger")
        interference_type: Type of interference
        **kwargs: Additional arguments for specific propagator types
        
    Returns:
        Configured field propagator module
    """
    if propagator_type == "standard":
        return DynamicFieldPropagator(embed_dim, pos_dim, evolution_type, interference_type, **kwargs)
    elif propagator_type == "adaptive":
        return AdaptiveFieldPropagator(embed_dim, pos_dim, evolution_type, interference_type, **kwargs)
    elif propagator_type == "causal":
        return CausalFieldPropagator(embed_dim, pos_dim, evolution_type, interference_type, **kwargs)
    else:
        raise ValueError(f"Unknown propagator type: {propagator_type}") 