"""Classical estimators for comparison with deep learning methods.

Implements logit and Poisson regression using statsmodels.
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np
import statsmodels.api as sm


@dataclass
class ClassicalResult:
    """Result from classical estimator."""
    ate: float              # Treatment effect estimate (coefficient on T)
    se: float               # Standard error
    ci_lower: float         # 95% CI lower bound
    ci_upper: float         # 95% CI upper bound
    pvalue: float           # P-value for T coefficient
    model_summary: str      # Model summary string
    converged: bool         # Whether optimization converged


def classical_logit(
    X: np.ndarray,
    T: np.ndarray,
    Y: np.ndarray,
) -> ClassicalResult:
    """Logit regression: Y ~ T + X.

    Estimates the treatment effect as the coefficient on T in a logistic
    regression model with X as controls.

    Args:
        X: (n, d) covariate matrix (should be standardized)
        T: (n,) binary treatment variable
        Y: (n,) binary outcome variable

    Returns:
        ClassicalResult with ATE (log-odds ratio) and inference
    """
    # Build design matrix: [constant, T, X]
    n = len(T)
    design = np.column_stack([np.ones(n), T, X])

    # Fit logit model
    model = sm.Logit(Y, design)

    try:
        result = model.fit(disp=0, maxiter=100)
        converged = result.converged
    except Exception as e:
        # Return NaN result if optimization fails
        return ClassicalResult(
            ate=np.nan, se=np.nan, ci_lower=np.nan, ci_upper=np.nan,
            pvalue=np.nan, model_summary=f"Optimization failed: {e}",
            converged=False,
        )

    # Extract treatment coefficient (index 1)
    ate = float(result.params[1])
    se = float(result.bse[1])
    pvalue = float(result.pvalues[1])

    # 95% CI
    ci_lower = ate - 1.96 * se
    ci_upper = ate + 1.96 * se

    # Summary string
    summary = (
        f"Logit: coef(T)={ate:.4f}, SE={se:.4f}, "
        f"z={ate/se:.2f}, p={pvalue:.4f}, "
        f"Pseudo R²={result.prsquared:.4f}"
    )

    return ClassicalResult(
        ate=ate, se=se, ci_lower=ci_lower, ci_upper=ci_upper,
        pvalue=pvalue, model_summary=summary, converged=converged,
    )


def classical_poisson(
    X: np.ndarray,
    T: np.ndarray,
    Y: np.ndarray,
    offset: Optional[np.ndarray] = None,
) -> ClassicalResult:
    """Poisson regression: Y ~ T + X with optional offset.

    Estimates the treatment effect as the coefficient on T in a Poisson
    regression model with X as controls.

    Args:
        X: (n, d) covariate matrix (should be standardized)
        T: (n,) binary treatment variable
        Y: (n,) count outcome variable
        offset: (n,) offset (typically log(exposure))

    Returns:
        ClassicalResult with ATE (log-rate ratio) and inference
    """
    # Build design matrix: [constant, T, X]
    n = len(T)
    design = np.column_stack([np.ones(n), T, X])

    # Fit Poisson model
    model = sm.Poisson(Y, design, offset=offset)

    try:
        result = model.fit(disp=0, maxiter=100)
        converged = result.converged
    except Exception as e:
        return ClassicalResult(
            ate=np.nan, se=np.nan, ci_lower=np.nan, ci_upper=np.nan,
            pvalue=np.nan, model_summary=f"Optimization failed: {e}",
            converged=False,
        )

    # Extract treatment coefficient (index 1)
    ate = float(result.params[1])
    se = float(result.bse[1])
    pvalue = float(result.pvalues[1])

    # 95% CI
    ci_lower = ate - 1.96 * se
    ci_upper = ate + 1.96 * se

    # Incidence Rate Ratio
    irr = np.exp(ate)

    # Summary string
    summary = (
        f"Poisson: coef(T)={ate:.4f}, SE={se:.4f}, "
        f"z={ate/se:.2f}, p={pvalue:.4f}, "
        f"IRR={irr:.4f}, Pseudo R²={result.prsquared:.4f}"
    )

    return ClassicalResult(
        ate=ate, se=se, ci_lower=ci_lower, ci_upper=ci_upper,
        pvalue=pvalue, model_summary=summary, converged=converged,
    )


def classical_ols(
    X: np.ndarray,
    T: np.ndarray,
    Y: np.ndarray,
) -> ClassicalResult:
    """OLS regression: Y ~ T + X.

    Estimates the treatment effect as the coefficient on T in a linear
    regression model with X as controls.

    Args:
        X: (n, d) covariate matrix (should be standardized)
        T: (n,) treatment variable
        Y: (n,) continuous outcome variable

    Returns:
        ClassicalResult with ATE and inference
    """
    # Build design matrix: [constant, T, X]
    n = len(T)
    design = np.column_stack([np.ones(n), T, X])

    # Fit OLS model
    model = sm.OLS(Y, design)
    result = model.fit()

    # Extract treatment coefficient (index 1)
    ate = float(result.params[1])
    se = float(result.bse[1])
    pvalue = float(result.pvalues[1])

    # 95% CI
    ci_lower = ate - 1.96 * se
    ci_upper = ate + 1.96 * se

    # Summary string
    summary = (
        f"OLS: coef(T)={ate:.4f}, SE={se:.4f}, "
        f"t={ate/se:.2f}, p={pvalue:.4f}, "
        f"R²={result.rsquared:.4f}"
    )

    return ClassicalResult(
        ate=ate, se=se, ci_lower=ci_lower, ci_upper=ci_upper,
        pvalue=pvalue, model_summary=summary, converged=True,
    )
