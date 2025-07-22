"""Deprecated thin wrappers around UnifiedTFN for backward compatibility.

This module exists solely so that legacy code/tests importing
`tfn.model.tfn_classifiers` continue to work after the consolidation of
all 1-D TFN variants into :class:`tfn.model.tfn_unified.UnifiedTFN`.

Do **NOT** add new features here. Prefer the UnifiedTFN API instead.
"""

from __future__ import annotations

from .tfn_unified import UnifiedTFN
from .tfn_enhanced import EnhancedTFNModel  # Re-export as enhanced classifier

# Legacy aliases -------------------------------------------------------------

class TFNClassifier(UnifiedTFN):
    """Legacy name alias for UnifiedTFN (classification)."""

    pass


class TFNRegressor(UnifiedTFN):
    """Legacy name alias for UnifiedTFN (regression)."""

    pass
EnhancedTFNClassifier = EnhancedTFNModel

__all__ = [
    "TFNClassifier",
    "TFNRegressor",
    "EnhancedTFNClassifier",
] 