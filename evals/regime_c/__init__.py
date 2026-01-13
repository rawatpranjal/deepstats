"""
Regime C Evaluations: Observational Logit with EstimateLambda

This regime validates the full generality of Theorem 2:
    - θ̂(x) recovery with nonlinear α*(x) = 0.5·sin(x)
    - Score/Hessian autodiff vs calculus
    - EstimateLambda (3-way split) for E[ℓ_θθ|X]
    - Target Jacobian H_θ computation
    - Influence function ψ assembly
    - Frequentist coverage

This is the "hard mode" regime where:
    - Treatment is confounded (T depends on X)
    - Hessian depends on θ
    - Λ(x) must be estimated via neural network regression
    - 3-way cross-fitting is required
"""

from .eval_01_theta import run_eval_01
from .eval_02_autodiff import run_eval_02
from .eval_03_lambda import run_eval_03
from .eval_04_jacobian import run_eval_04
from .eval_05_psi import run_eval_05
from .eval_06_coverage import run_eval_06
from .run_regime_c import run_regime_c

__all__ = [
    "run_eval_01",
    "run_eval_02",
    "run_eval_03",
    "run_eval_04",
    "run_eval_05",
    "run_eval_06",
    "run_regime_c",
]
