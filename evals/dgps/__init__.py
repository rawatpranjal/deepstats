"""
DGP definitions for the 3 Golden Tests.

Each regime validates a different Lambda strategy:
- Regime A: RCT Logit → ComputeLambda (Monte Carlo)
- Regime B: Linear → AnalyticLambda (E[TT'|X])
- Regime C: Obs Logit → EstimateLambda (3-way split)
"""

from .regime_a_rct_logit import RCTLogitDGP, generate_rct_logit_data
from .regime_b_linear import LinearDGP, generate_linear_data
from .regime_c_obs_logit import CanonicalDGP, generate_canonical_dgp

__all__ = [
    "RCTLogitDGP",
    "generate_rct_logit_data",
    "LinearDGP",
    "generate_linear_data",
    "CanonicalDGP",
    "generate_canonical_dgp",
]
