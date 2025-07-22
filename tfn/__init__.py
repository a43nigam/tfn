"""
Token Field Network (TFN)

A novel deep learning architecture that replaces attention with continuous 
field projection, evolution, and sampling.

This package provides:
- Core TFN components (kernels, field projection, evolution, sampling)
- Trainable TFN models (classifiers, regressors)
- Utility functions for data, metrics, and visualization
- Training and evaluation scripts
"""

__version__ = "0.1.0"
__author__ = "Anubhav Nigam"
__email__ = "anubhav.nigam@example.com"

# Import main components for easy access
from .model import (
    UnifiedTFN,
    ImageTFN,
    TrainableTFNLayer,
    LearnableKernels,
    TrainableEvolution,
    PositionEmbeddings,
)

from .core import (
    KernelBasis,
    RBFKernel,
    CompactKernel,
    FourierKernel,
    LearnableKernel,
    KernelFactory,
    FieldProjector,
    UniformFieldGrid,
    FieldEvolver,
    CNNFieldEvolver,
    PDEFieldEvolver,
    TemporalGrid,
    FieldSampler
)

__all__ = [
    # Version info
    "__version__",
    "__author__",
    "__email__",
    
    # Model classes
    "UnifiedTFN",
    "ImageTFN",
    "TrainableTFNLayer",
    "LearnableKernels",
    "TrainableEvolution",
    "PositionEmbeddings",
    
    # Core components
    "KernelBasis",
    "RBFKernel",
    "CompactKernel", 
    "FourierKernel",
    "LearnableKernel",
    "KernelFactory",
    "FieldProjector",
    "UniformFieldGrid",
    "FieldEvolver",
    "CNNFieldEvolver",
    "PDEFieldEvolver",
    "TemporalGrid",
    "FieldSampler",
] 