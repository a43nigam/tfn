"""
TFN Model Package

Clean, reusable model classes for Token Field Networks.
"""

from .tfn_base import (
    LearnableKernels,
    TrainableEvolution,
    PositionEmbeddings,
    TrainableTFNLayer
)

from .tfn_unified import UnifiedTFN
from .tfn_pytorch import ImageTFN

# ---------------------------------------------------------------------------
# Backward-compatibility aliases (deprecated)
# ---------------------------------------------------------------------------

TFNClassifier = UnifiedTFN  # Legacy alias for tests / old scripts
TFNRegressor = UnifiedTFN   # Legacy alias

__all__ = [
    # Base components
    'LearnableKernels',
    'TrainableEvolution', 
    'PositionEmbeddings',
    'TrainableTFNLayer',
    
    # Unified model
    'UnifiedTFN', 'TFNClassifier', 'TFNRegressor',
    'ImageTFN',
]
