"""
Comprehensive Evals: Ground Truth Validation Protocol

Validates all 3 Lambda regimes for the deep_inference package.
Each regime tests specific mathematical objects from Theorem 2:

    ψ = H - H_θ · Λ⁻¹ · ℓ_θ

Regimes:
    - regime_a: RCT Logit with ComputeLambda (Monte Carlo)
    - regime_b: Linear Model with AnalyticLambda (E[TT'|X])
    - regime_c: Observational Logit with EstimateLambda (3-way split)

DGPs:
    - dgps/regime_a_rct_logit.py: Randomized T ~ Bernoulli(0.5)
    - dgps/regime_b_linear.py: Confounded linear with α*(x) = x²
    - dgps/regime_c_obs_logit.py: Confounded logit (hardest case)

Usage:
    python -m evals.run_all              # Run all 3 regimes
    python -m evals.run_all --regime a   # Run only Regime A
    python -m evals.regime_a.run_regime_a  # Run Regime A directly
"""

# Export DGPs (backward compatible)
from .dgps.regime_c_obs_logit import generate_canonical_dgp, CanonicalDGP

# Also keep dgp.py export for backward compatibility
from .dgp import generate_canonical_dgp, CanonicalDGP

__all__ = [
    "generate_canonical_dgp",
    "CanonicalDGP",
]
