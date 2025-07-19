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

from .tfn_classifiers import (
    TFNClassifier,
    TFNRegressor
)

from .tfn_2d import (
    TrainableTFNLayer2D,
    TFNClassifier2D,
    create_tfn2d_variants,
)

__all__ = [
    # Base components
    'LearnableKernels',
    'TrainableEvolution', 
    'PositionEmbeddings',
    'TrainableTFNLayer',
    
    # Complete models
    'TFNClassifier',
    'TFNRegressor',
]

__all__ += [
    'TrainableTFNLayer2D',
    'TFNClassifier2D',
    'create_tfn2d_variants',
]
