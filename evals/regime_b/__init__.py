"""
Regime B Evaluations: Linear Model with AnalyticLambda

This regime validates:
    - θ̂(x) recovery with nonlinear α*(x) = x²
    - Score/Hessian derivatives (constant Hessian!)
    - AnalyticLambda matches E[TT'|X]
    - ψ matches Robinson 1988 closed form
    - Frequentist coverage

The key insight: Linear models have Hessian that is CONSTANT w.r.t. θ,
so Λ(x) = E[ℓ_θθ|X] = E[TT'|X] can be computed directly without θ̂.
"""

from .eval_01_theta_recovery import run_eval_01_theta_recovery
from .eval_02_derivatives import run_eval_02_derivatives
from .eval_03_lambda_analytic import run_eval_03_lambda_analytic
from .eval_04_psi_closed_form import run_eval_04_psi_closed_form
from .eval_05_coverage import run_eval_05_coverage
from .run_regime_b import run_regime_b

__all__ = [
    "run_eval_01_theta_recovery",
    "run_eval_02_derivatives",
    "run_eval_03_lambda_analytic",
    "run_eval_04_psi_closed_form",
    "run_eval_05_coverage",
    "run_regime_b",
]
