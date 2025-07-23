from __future__ import annotations

"""tfn.core.dynamic_propagation (compat layer)

This module re-exports classes and factory functions that were 
previously provided by *dynamic_propagation.py* but are now located 
in *field_evolution.py*.  It exists solely to maintain backward 
compatibility with older tests and user code that import from the old 
path.
"""

from .field_evolution import (
    DynamicFieldPropagator,
    AdaptiveFieldPropagator,
    CausalFieldPropagator,
    create_field_evolver as create_field_propagator,
)

__all__ = [
    "DynamicFieldPropagator",
    "AdaptiveFieldPropagator",
    "CausalFieldPropagator",
    "create_field_propagator",
] 