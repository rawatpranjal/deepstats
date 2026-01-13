"""
Regime A Evals: Randomized Controlled Trial (RCT) with Logit Model

Validates the ComputeLambda strategy (Monte Carlo integration over known F_T).

Key Tests:
1. Lambda Computation - MC matches analytical formula
2. Two-Way Split - No 3-way split occurs
3. Boundary Stability - No NaN at probability extremes
4. Coverage - ~95% frequentist coverage
"""

from .eval_01_lambda_compute import run_eval_01_lambda_compute
from .eval_02_two_way_split import run_eval_02_two_way_split
from .eval_03_boundary_stability import run_eval_03_boundary
from .eval_04_coverage import run_eval_04_coverage
from .run_regime_a import run_regime_a

__all__ = [
    "run_eval_01_lambda_compute",
    "run_eval_02_two_way_split",
    "run_eval_03_boundary",
    "run_eval_04_coverage",
    "run_regime_a",
]
