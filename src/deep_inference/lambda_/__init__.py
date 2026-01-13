"""
Lambda (Λ) estimation strategies.

Λ(x) = E[ℓ_θθ(Y, T, θ(X)) | X = x]

Three regimes determine how to obtain Λ:

- Regime A: Randomized + ℓ_θθ doesn't depend on Y
  → COMPUTE via Monte Carlo: Λ(x) = ∫ ℓ_θθ(t, θ̂(x)) dF_T(t)

- Regime B: Linear model (ℓ_θθ = TT')
  → ANALYTIC: Λ(x) = E[TT'|X] (doesn't depend on θ!)

- Regime C: Observational + Nonlinear
  → ESTIMATE via neural network: fit X → Λ(X)
"""

from .base import LambdaStrategy, Regime
from .selector import detect_regime, select_lambda_strategy
from .compute import ComputeLambda
from .analytic import AnalyticLambda
from .estimate import EstimateLambda

__all__ = [
    # Base
    "LambdaStrategy",
    "Regime",
    # Selection
    "detect_regime",
    "select_lambda_strategy",
    # Strategies
    "ComputeLambda",
    "AnalyticLambda",
    "EstimateLambda",
]
