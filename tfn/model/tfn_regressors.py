"""Deprecated thin wrappers around UnifiedTFN for backward compatibility.

This module stays to satisfy old imports in registry/tests. It maps all
legacy regressor classes to :class:`tfn.model.tfn_unified.UnifiedTFN`.
"""

from __future__ import annotations

from .tfn_unified import UnifiedTFN


class TFNTimeSeriesRegressor(UnifiedTFN):
    pass


class TFNMultiStepRegressor(UnifiedTFN):
    pass


class TFNSequenceRegressor(UnifiedTFN):
    pass

__all__ = [
    "TFNTimeSeriesRegressor",
    "TFNMultiStepRegressor",
    "TFNSequenceRegressor",
] 