"""Statistical inference methods for deepstats.

This module provides:
- Robust standard errors (HC0-HC3, clustered)
- Bootstrap standard errors (pairs, residual, wild)
- Influence function standard errors (Farrell et al. 2021)
- Jackknife standard errors
- SE validation utilities
"""

from .standard_errors import (
    compute_vcov,
    compute_vcov_cluster,
    compute_vcov_hc0,
    compute_vcov_hc1,
    compute_vcov_hc2,
    compute_vcov_hc3,
    compute_vcov_iid,
)
from .bootstrap import (
    BootstrapResult,
    bootstrap_pairs,
    bootstrap_residual,
    bootstrap_wild,
    create_nn_fit_function,
)
from .influence import (
    InfluenceFunctionResult,
    compute_influence_function_se,
)
from .jackknife import (
    JackknifeResult,
    jackknife_se,
    delete_d_jackknife_se,
    infinitesimal_jackknife_se,
)
from .validation import (
    SEValidationResult,
    validate_standard_errors,
    monte_carlo_se_comparison,
)

__all__ = [
    # Standard errors
    "compute_vcov",
    "compute_vcov_iid",
    "compute_vcov_hc0",
    "compute_vcov_hc1",
    "compute_vcov_hc2",
    "compute_vcov_hc3",
    "compute_vcov_cluster",
    # Bootstrap
    "BootstrapResult",
    "bootstrap_pairs",
    "bootstrap_residual",
    "bootstrap_wild",
    "create_nn_fit_function",
    # Influence function
    "InfluenceFunctionResult",
    "compute_influence_function_se",
    # Jackknife
    "JackknifeResult",
    "jackknife_se",
    "delete_d_jackknife_se",
    "infinitesimal_jackknife_se",
    # Validation
    "SEValidationResult",
    "validate_standard_errors",
    "monte_carlo_se_comparison",
]
