"""
deep_inference: Structural Deep Learning with Valid Inference

A from-scratch implementation of the Farrell-Liang-Misra framework
for structural deep learning with valid inference.

Usage:
    # Pre-built family
    from deep_inference import structural_dml
    result = structural_dml(Y, T, X, family='logit')

    # Custom loss function
    def my_loss(y, t, theta):
        alpha, beta = theta[:, 0], theta[:, 1]
        mu = alpha + beta * t
        return (y - mu) ** 2

    result = structural_dml(Y, T, X, loss_fn=my_loss, theta_dim=2)

    # Access results
    print(f"Estimate: {result.mu_hat:.4f} +/- {result.se:.4f}")
    print(f"95% CI: [{result.ci_lower:.4f}, {result.ci_upper:.4f}]")
"""

import numpy as np
from typing import Callable, List, Optional
from torch import Tensor

from .core import structural_dml_core, DMLResult
from .families import get_family, FAMILY_REGISTRY, BaseFamily


def structural_dml(
    Y: np.ndarray,
    T: np.ndarray,
    X: np.ndarray,
    family: Optional[str] = None,
    target: Optional[str] = None,
    loss_fn: Optional[Callable] = None,
    target_fn: Optional[Callable] = None,
    theta_dim: Optional[int] = None,
    n_folds: int = 50,
    hidden_dims: List[int] = [64, 32],
    epochs: int = 200,
    lr: float = 0.01,
    verbose: bool = False,
    store_data: bool = True,
    **kwargs,
) -> DMLResult:
    """
    Structural deep learning with valid inference.

    Implements the Farrell-Liang-Misra framework for estimating
    heterogeneous structural parameters with neural networks,
    using influence functions for valid inference.

    Requirements for Valid Inference
    --------------------------------
    1. **Model is well-specified**: The structural family matches the data
       generating process. Influence functions correct for regularization
       bias but cannot correct for model misspecification.

    2. **Network approximates θ*(x) well**: The neural network must be able
       to learn the true heterogeneity function. For simple heterogeneity,
       [64, 32] architecture is sufficient. For complex patterns, consider
       larger networks and more data.

    3. **Sufficient folds**: K >= 50 is recommended for stable SE estimation.
       With K=20, SE may be overestimated by 3x or more.

    4. **Well-conditioned Λ(x)**: Check diagnostics.min_lambda_eigenvalue.
       Near-singular Hessians can cause unstable estimates.

    If these conditions fail, coverage may be below 95%.

    Args:
        Y: (n,) outcome vector
        T: (n,) treatment vector
        X: (n, d) covariate matrix
        family: Pre-built family name. Available families:
                - 'linear': Y = alpha + beta*T + eps
                - 'logit': P(Y=1) = sigmoid(alpha + beta*T)
                - 'poisson': Y ~ Poisson(exp(alpha + beta*T))
                - 'gamma': Y ~ Gamma(shape, exp(alpha + beta*T))
                - 'gumbel': Y ~ Gumbel(alpha + beta*T, scale)
                - 'tobit': Y = max(0, alpha + beta*T + sigma*eps)
                - 'negbin': Y ~ NegBin(exp(alpha + beta*T), r)
                - 'weibull': Y ~ Weibull(shape, exp(alpha + beta*T))
        target: Target functional for inference (family-specific):
                - logit: 'beta' (log-odds, default) or 'ame' (average marginal effect)
                - tobit: 'latent' (effect on Y*, default) or 'observed' (effect on E[Y])
        loss_fn: Custom loss function (y, t, theta) -> (n,) losses
        target_fn: Custom target function (x, theta) -> scalar
        theta_dim: Dimension of parameter vector (required if custom loss)
        n_folds: Number of cross-fitting folds (default=50, minimum recommended)
        hidden_dims: Neural network hidden layer sizes
        epochs: Training epochs per fold
        lr: Learning rate
        verbose: Print progress
        store_data: Store X for prediction methods (default=True)
        **kwargs: Additional arguments to structural_dml_core

    Returns:
        DMLResult with:
            - mu_hat: Point estimate
            - se: Standard error
            - ci_lower, ci_upper: 95% confidence interval
            - psi_values: Influence function values
            - theta_hat: Estimated parameters for all observations
            - diagnostics: Training and estimation diagnostics including:
                - min_lambda_eigenvalue: Check for near-singular Hessians
                - n_regularized: Count of observations needing extra regularization
                - correction_ratio: If > 2, consider more folds

    Warnings:
        - High Lambda regularization rate: Indicates numerical instability
        - High correction variance ratio: Suggests too few folds (K < 50)

    Examples:
        # Binary outcome with heterogeneous effects
        result = structural_dml(Y, T, X, family='logit')

        # Continuous outcome
        result = structural_dml(Y, T, X, family='linear')

        # Check diagnostics
        print(f"Min eigenvalue: {result.diagnostics['min_lambda_eigenvalue']:.6f}")
        print(f"Observations regularized: {result.diagnostics['n_regularized']}")

        # Custom structural model
        def tobit_loss(y, t, theta):
            import torch
            alpha, beta = theta[:, 0], theta[:, 1]
            mu = alpha + beta * t
            sigma = 1.0
            # Tobit log-likelihood
            censored = (y <= 0).float()
            uncensored = 1 - censored
            z = -mu / sigma
            ll = censored * torch.distributions.Normal(0, 1).cdf(z).log()
            ll += uncensored * (-0.5 * ((y - mu) / sigma) ** 2 - 0.5 * np.log(2 * np.pi) - np.log(sigma))
            return -ll

        result = structural_dml(Y, T, X, loss_fn=tobit_loss, theta_dim=2)
    """
    # Validate inputs
    if family is None and loss_fn is None:
        raise ValueError("Must provide either 'family' or 'loss_fn'")

    if family is not None and loss_fn is not None:
        raise ValueError("Cannot provide both 'family' and 'loss_fn'")

    if loss_fn is not None and theta_dim is None:
        raise ValueError("Must provide 'theta_dim' when using custom loss_fn")

    # Get family or use custom functions
    if family is not None:
        # Build family kwargs (e.g., target='ame' for logit)
        family_kwargs = {}
        if target is not None:
            family_kwargs['target'] = target

        fam = get_family(family, **family_kwargs)
        loss_fn = fam.loss
        theta_dim = fam.theta_dim
        three_way = fam.hessian_depends_on_theta()

        # Use closed-form functions if available
        # Create test inputs with correct theta dimension
        test_theta = Tensor([[0.0] * theta_dim])
        try:
            grad_result = fam.gradient(Tensor([0.0]), Tensor([0.0]), test_theta)
            gradient_fn = fam.gradient if grad_result is not None else None
        except Exception:
            gradient_fn = None

        try:
            hess_result = fam.hessian(Tensor([0.0]), Tensor([0.0]), test_theta)
            hessian_fn = fam.hessian if hess_result is not None else None
        except Exception:
            hessian_fn = None

        # Use family's target functions
        target_fn = fam.default_target
        per_obs_target_fn = fam.per_obs_target
        per_obs_target_grad_fn = fam.per_obs_target_gradient
    else:
        # Fully automatic mode
        three_way = kwargs.pop('three_way', None)  # Auto-detect
        gradient_fn = None
        hessian_fn = None
        per_obs_target_fn = None
        per_obs_target_grad_fn = None

        # Default target if not provided
        if target_fn is None:
            def target_fn(x, theta):
                return theta[:, 1].mean()

    result = structural_dml_core(
        Y=Y,
        T=T,
        X=X,
        loss_fn=loss_fn,
        target_fn=target_fn,
        theta_dim=theta_dim,
        n_folds=n_folds,
        hidden_dims=hidden_dims,
        epochs=epochs,
        lr=lr,
        three_way=three_way,
        gradient_fn=gradient_fn,
        hessian_fn=hessian_fn,
        per_obs_target_fn=per_obs_target_fn,
        per_obs_target_grad_fn=per_obs_target_grad_fn,
        verbose=verbose,
        **kwargs,
    )

    # Set metadata on result
    result._family = family
    result._target = target if target else "E[beta]"
    result._n_obs = len(Y)
    result._n_folds = n_folds

    # Store X for prediction capability
    if store_data:
        result._X_train = np.asarray(X).copy()

    return result


# New API: inference() with general loss/target
from dataclasses import dataclass, field
from typing import Any
import torch

from .utils.result_mixin import PredictVisualizeMixin


@dataclass
class InferenceResult(PredictVisualizeMixin):
    """Result from the new inference() API."""

    mu_hat: float
    se: float
    ci_lower: float
    ci_upper: float
    psi_values: Tensor
    theta_hat: Tensor
    diagnostics: dict

    # Metadata fields
    _model: Optional[str] = None
    _target: Optional[str] = None
    _n_obs: Optional[int] = None
    _n_folds: Optional[int] = None

    # Fields for prediction capability
    _X_train: Optional[np.ndarray] = field(default=None, repr=False)
    _theta_predictor: Optional[Any] = field(default=None, repr=False)

    def __repr__(self) -> str:
        """Short representation."""
        from .utils.formatting import format_short_repr
        return format_short_repr(
            class_name="InferenceResult",
            estimate=self.mu_hat,
            se=self.se,
            ci_lower=self.ci_lower,
            ci_upper=self.ci_upper,
        )

    def summary(self) -> str:
        """
        Generate statsmodels-style summary.

        Returns:
            Formatted summary string
        """
        from .utils.formatting import format_full_summary

        # Determine target name for display
        target_name = self._target if self._target else "E[beta]"

        return format_full_summary(
            title="Structural Inference Results",
            coef_name=target_name,
            estimate=self.mu_hat,
            se=self.se,
            ci_lower=self.ci_lower,
            ci_upper=self.ci_upper,
            diagnostics=self.diagnostics,
            family=self._model,
            target=target_name,
            n_obs=self._n_obs,
            n_folds=self._n_folds,
        )


def inference(
    Y: np.ndarray,
    T: np.ndarray,
    X: np.ndarray,
    # Option 1: Built-in model/target (strings)
    model: Optional[str] = None,
    target: Optional[str] = None,
    # Option 2: Custom loss/target functions
    loss: Optional[Callable] = None,
    target_fn: Optional[Callable] = None,
    theta_dim: Optional[int] = None,
    # Evaluation point
    t_tilde: Optional[float] = None,
    # Randomization settings (for Regime A)
    is_randomized: bool = False,
    treatment_dist: Optional["TreatmentDistribution"] = None,
    # Lambda estimation override
    lambda_method: Optional[str] = None,
    # Cross-fitting settings
    n_folds: int = 50,
    # Network settings
    hidden_dims: List[int] = [64, 32],
    epochs: int = 200,
    lr: float = 0.01,
    patience: int = 50,
    # Other
    ridge: float = 1e-4,
    verbose: bool = False,
    store_data: bool = True,
) -> InferenceResult:
    """
    General inference with user-provided loss and target.

    This is the new API that supports arbitrary loss functions and targets.
    Everything is derived via autodiff unless closed-forms are provided.

    Args:
        Y: (n,) outcomes
        T: (n,) treatments
        X: (n, d_x) covariates

        # Model specification (choose one):
        model: Built-in model name ("linear", "logit", etc.)
        loss: Custom loss function: loss(y, t, theta) -> scalar

        # Target specification (choose one):
        target: Built-in target name ("beta", "ame", etc.)
        target_fn: Custom target: h(x, theta, t_tilde) -> scalar

        theta_dim: Parameter dimension (required for custom loss)
        t_tilde: Evaluation point (default: mean(T))

        # Regime settings:
        is_randomized: True if T is randomly assigned
        treatment_dist: Distribution F_T (enables Regime A computation)
        lambda_method: Override auto-detection ("compute", "analytic", "estimate")

        # Cross-fitting:
        n_folds: Number of folds (default: 50)

        # Network:
        hidden_dims: Hidden layer sizes
        epochs: Training epochs
        lr: Learning rate

        ridge: Regularization for Lambda inversion
        verbose: Print progress
        store_data: Store X for prediction methods (default=True)

    Returns:
        InferenceResult with mu_hat, se, ci, psi_values, theta_hat, diagnostics

    Examples:
        # Built-in model and target
        result = inference(Y, T, X, model="logit", target="ame")

        # Custom loss and target
        def my_loss(y, t, theta):
            p = torch.sigmoid(theta[0] + theta[1] * t)
            return -y * torch.log(p) - (1-y) * torch.log(1-p)

        def my_target(x, theta, t_tilde):
            p = torch.sigmoid(theta[0] + theta[1] * t_tilde)
            return p * (1-p) * theta[1]

        result = inference(Y, T, X, loss=my_loss, target_fn=my_target, theta_dim=2)
    """
    from .models import Linear, Logit, CustomModel, model_from_loss
    from .targets import AverageParameter, AME, CustomTarget
    from .lambda_ import select_lambda_strategy, Regime, detect_regime
    from .engine import run_crossfit

    # Convert inputs to tensors
    Y_t = torch.tensor(Y, dtype=torch.float32)
    T_t = torch.tensor(T, dtype=torch.float32)
    X_t = torch.tensor(X, dtype=torch.float32)

    # Default t_tilde to mean treatment
    if t_tilde is None:
        t_tilde = T_t.mean()
    else:
        t_tilde = torch.tensor(t_tilde, dtype=torch.float32)

    # Resolve model
    if model is not None:
        # Built-in model
        model_map = {
            "linear": Linear(),
            "logit": Logit(),
        }
        if model not in model_map:
            raise ValueError(f"Unknown model: {model}. Available: {list(model_map.keys())}")
        struct_model = model_map[model]
    elif loss is not None:
        # Custom loss
        if theta_dim is None:
            raise ValueError("theta_dim required for custom loss")
        struct_model = model_from_loss(loss, theta_dim)
    else:
        raise ValueError("Must provide 'model' or 'loss'")

    # Resolve target
    if target is not None:
        # Built-in target
        target_map = {
            "beta": AverageParameter(param_index=1, theta_dim=struct_model.theta_dim),
            "ame": AME(param_index=1, model_type="logit" if model == "logit" else "linear"),
        }
        if target not in target_map:
            raise ValueError(f"Unknown target: {target}. Available: {list(target_map.keys())}")
        struct_target = target_map[target]
    elif target_fn is not None:
        # Custom target
        struct_target = CustomTarget(h_fn=target_fn)
    else:
        # Default: average beta
        struct_target = AverageParameter(param_index=1, theta_dim=struct_model.theta_dim)

    # Select Lambda strategy
    lambda_strategy = select_lambda_strategy(
        model=struct_model,
        is_randomized=is_randomized,
        treatment_dist=treatment_dist,
        lambda_method=lambda_method,
    )

    # Detect regime for diagnostics
    regime = detect_regime(struct_model, is_randomized, treatment_dist is not None)

    if verbose:
        from .lambda_.selector import describe_regime
        print(f"Detected: {describe_regime(regime)}")

    # Run cross-fitting
    result = run_crossfit(
        Y=Y_t,
        T=T_t,
        X=X_t,
        t_tilde=t_tilde,
        model=struct_model,
        target=struct_target,
        lambda_strategy=lambda_strategy,
        n_folds=n_folds,
        epochs=epochs,
        lr=lr,
        patience=patience,
        hidden_dims=hidden_dims,
        ridge=ridge,
        verbose=verbose,
    )

    inf_result = InferenceResult(
        mu_hat=result.mu_hat,
        se=result.se,
        ci_lower=result.ci_lower,
        ci_upper=result.ci_upper,
        psi_values=result.psi_values,
        theta_hat=result.theta_hat,
        diagnostics={
            "regime": regime.name,
            "n_folds": n_folds,
            "lambda_method": lambda_strategy.__class__.__name__,
        },
        _model=model,
        _target=target if target else "E[beta]",
        _n_obs=len(Y),
        _n_folds=n_folds,
    )

    # Store X for prediction capability
    if store_data:
        inf_result._X_train = np.asarray(X).copy()

    return inf_result


# Re-export key classes
from .core import DMLResult, compute_coverage, compute_se_ratio
from .families import (
    LinearFamily,
    LogitFamily,
    PoissonFamily,
    GammaFamily,
    GumbelFamily,
    TobitFamily,
    NegBinFamily,
    WeibullFamily,
    BaseFamily,
)

# New architecture exports
from .models import StructuralModel, CustomModel, Linear, Logit
from .targets import Target, CustomTarget, AverageParameter, AME
from .lambda_ import Regime, detect_regime, select_lambda_strategy

__all__ = [
    # New API
    'inference',
    'InferenceResult',
    # Legacy API
    'structural_dml',
    # Result class
    'DMLResult',
    # Families (legacy)
    'LinearFamily',
    'LogitFamily',
    'PoissonFamily',
    'GammaFamily',
    'GumbelFamily',
    'TobitFamily',
    'NegBinFamily',
    'WeibullFamily',
    'BaseFamily',
    'get_family',
    'FAMILY_REGISTRY',
    # New architecture
    'StructuralModel',
    'CustomModel',
    'Linear',
    'Logit',
    'Target',
    'CustomTarget',
    'AverageParameter',
    'AME',
    'Regime',
    'detect_regime',
    'select_lambda_strategy',
    # Utilities
    'compute_coverage',
    'compute_se_ratio',
]
