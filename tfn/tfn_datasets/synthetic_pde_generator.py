"""tfn.tfn_datasets.synthetic_pde_generator
Synthetic PDE dataset generator for validating physics-informed models.
This module was renamed from `physics_loader.py` to highlight that the datasets
are synthetic and generated on-the-fly rather than loaded from an external
corpus. All functionality remains identical.
"""

# Re-export all public symbols from the original physics_loader to keep
# behaviour identical while we transition module names.
from .physics_loader import *  # noqa: F401,F403 