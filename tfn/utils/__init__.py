from __future__ import annotations

"""Utility subpackage for Token Field Network.

Re-exports common helper modules. Additional utilities should be added to this
package (see synthetic_sequence_tasks for synthetic datasets).
"""

from importlib import import_module as _import_module

# Common utilities – imported lazily to keep import cost low
__all__ = ["data_utils", "metrics", "plot_utils", "synthetic_sequence_tasks"]

globals().update({name: _import_module(f"tfn.utils.{name}") for name in __all__}) 