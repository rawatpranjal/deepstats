"""Core DML algorithm with proper cross-fitting and splitting."""

import torch
from torch import Tensor
import numpy as np
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Dict, Any, Tuple

from .autodiff import compute_gradient, compute_hessian, detect_theta_dependence
from .lambda_estimator import LambdaEstimator, AggregateLambdaEstimator
from ..models import StructuralNet, train_structural_net, TrainingHistory
from ..utils import batch_inverse

# Optional tqdm import
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


@dataclass
class DMLResult:
    """Result of structural DML estimation."""

    mu_hat: float                          # Point estimate
    se: float                              # Standard error
    ci_lower: float                        # 95% CI lower bound
    ci_upper: float                        # 95% CI upper bound
    psi_values: np.ndarray                 # Influence function values
    theta_hat: np.ndarray                  # Estimated parameters theta(x) for all x
    mu_naive: float                        # Naive estimate (for comparison)
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    # Metadata fields (set by structural_dml wrapper)
    _family: Optional[str] = field(default=None, repr=False)
    _target: Optional[str] = field(default=None, repr=False)
    _n_obs: Optional[int] = field(default=None, repr=False)
    _n_folds: Optional[int] = field(default=None, repr=False)

    # Fields for prediction capability
    _X_train: Optional[np.ndarray] = field(default=None, repr=False)
    _theta_predictor: Optional[Any] = field(default=None, repr=False)  # sklearn Ridge

    def __repr__(self) -> str:
        """Short representation."""
        from ..utils.formatting import format_short_repr
        return format_short_repr(
            class_name="DMLResult",
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
        from ..utils.formatting import format_full_summary

        # Determine target name for display
        target_name = self._target if self._target else "E[beta]"

        return format_full_summary(
            title="Structural DML Results",
            coef_name=target_name,
            estimate=self.mu_hat,
            se=self.se,
            ci_lower=self.ci_lower,
            ci_upper=self.ci_upper,
            diagnostics=self.diagnostics,
            family=self._family,
            target=target_name,
            n_obs=self._n_obs,
            n_folds=self._n_folds,
        )

    # =========================================================================
    # Prediction Methods
    # =========================================================================

    def _fit_theta_predictor(self) -> None:
        """Fit sklearn Ridge on (X_train, theta_hat) - lazy, called on first predict."""
        if self._theta_predictor is not None:
            return
        if self._X_train is None:
            raise RuntimeError("X_train not stored. Re-run with store_data=True.")
        from sklearn.linear_model import Ridge
        self._theta_predictor = Ridge(alpha=1.0)
        self._theta_predictor.fit(self._X_train, self.theta_hat)

    def predict_theta(self, X_new: np.ndarray) -> np.ndarray:
        """
        Predict theta(X) = [alpha(X), beta(X)] for new X.

        Args:
            X_new: (m, d) covariate matrix for new observations

        Returns:
            (m, theta_dim) predicted parameters
        """
        self._fit_theta_predictor()
        return self._theta_predictor.predict(np.atleast_2d(X_new))

    def predict_alpha(self, X_new: np.ndarray) -> np.ndarray:
        """
        Predict alpha(X) for new X.

        Args:
            X_new: (m, d) covariate matrix for new observations

        Returns:
            (m,) predicted baseline parameters
        """
        return self.predict_theta(X_new)[:, 0]

    def predict_beta(self, X_new: np.ndarray) -> np.ndarray:
        """
        Predict beta(X) for new X.

        Args:
            X_new: (m, d) covariate matrix for new observations

        Returns:
            (m,) predicted treatment effect parameters
        """
        return self.predict_theta(X_new)[:, 1]

    def predict_proba(
        self,
        X_new: np.ndarray,
        T_new: np.ndarray = None,
        t_value: float = 0.0,
    ) -> np.ndarray:
        """
        Predict P(Y=1|X,T) for logit/probit families.

        Args:
            X_new: (m, d) covariate matrix for new observations
            T_new: (m,) treatment values. If None, uses t_value for all.
            t_value: Scalar treatment value (used if T_new is None)

        Returns:
            (m,) predicted probabilities
        """
        if self._family not in ('logit', 'probit'):
            raise ValueError(f"predict_proba requires logit/probit, got '{self._family}'")
        theta = self.predict_theta(X_new)
        T = np.full(len(theta), t_value) if T_new is None else np.asarray(T_new)
        logits = theta[:, 0] + theta[:, 1] * T
        if self._family == 'logit':
            return 1 / (1 + np.exp(-logits))
        else:  # probit
            from scipy.stats import norm
            return norm.cdf(logits)

    def predict(
        self,
        X_new: np.ndarray,
        T_new: np.ndarray = None,
        t_value: float = 0.0,
    ) -> np.ndarray:
        """
        Predict E[Y|X,T]. Routes to appropriate method based on family.

        Args:
            X_new: (m, d) covariate matrix for new observations
            T_new: (m,) treatment values. If None, uses t_value for all.
            t_value: Scalar treatment value (used if T_new is None)

        Returns:
            (m,) predicted outcomes
        """
        theta = self.predict_theta(X_new)
        T = np.full(len(theta), t_value) if T_new is None else np.asarray(T_new)
        eta = theta[:, 0] + theta[:, 1] * T

        if self._family in ('logit', 'probit'):
            return self.predict_proba(X_new, T_new, t_value)
        elif self._family in ('poisson', 'negbin'):
            return np.exp(np.clip(eta, -20, 20))
        else:
            return eta  # linear predictor

    # =========================================================================
    # Visualization Methods
    # =========================================================================

    def plot_distributions(
        self,
        figsize: Tuple[int, int] = (12, 5),
        bins: int = 30,
        kde: bool = True,
        alpha_true: np.ndarray = None,
        beta_true: np.ndarray = None,
        save_path: str = None,
    ):
        """
        Plot KDE distributions of alpha(X) and beta(X).

        Args:
            figsize: Figure size (width, height)
            bins: Number of histogram bins
            kde: Whether to overlay KDE curve
            alpha_true: True alpha values (if known, for comparison)
            beta_true: True beta values (if known, for comparison)
            save_path: Path to save figure (optional)

        Returns:
            matplotlib Figure object
        """
        try:
            import matplotlib.pyplot as plt
            from scipy.stats import gaussian_kde
        except ImportError:
            raise ImportError("Install matplotlib: pip install deep-inference[plotting]")

        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # Alpha distribution
        alpha_hat = self.theta_hat[:, 0]
        axes[0].hist(alpha_hat, bins=bins, alpha=0.6, density=True, edgecolor='black')
        if kde:
            kde_a = gaussian_kde(alpha_hat)
            x = np.linspace(alpha_hat.min(), alpha_hat.max(), 200)
            axes[0].plot(x, kde_a(x), 'b-', lw=2)
        axes[0].axvline(alpha_hat.mean(), color='blue', ls=':', lw=2, label=f'Mean: {alpha_hat.mean():.3f}')
        if alpha_true is not None:
            axes[0].axvline(np.mean(alpha_true), color='red', ls='--', lw=2, label=f'True: {np.mean(alpha_true):.3f}')
        axes[0].set_xlabel(r'$\alpha(X)$')
        axes[0].set_title(r'Distribution of $\alpha(X)$')
        axes[0].legend()

        # Beta distribution
        beta_hat = self.theta_hat[:, 1]
        axes[1].hist(beta_hat, bins=bins, alpha=0.6, density=True, edgecolor='black')
        if kde:
            kde_b = gaussian_kde(beta_hat)
            x = np.linspace(beta_hat.min(), beta_hat.max(), 200)
            axes[1].plot(x, kde_b(x), 'g-', lw=2)
        axes[1].axvline(self.mu_hat, color='purple', ls='-', lw=2, label=f'IF est: {self.mu_hat:.3f}')
        axes[1].axvline(beta_hat.mean(), color='green', ls=':', lw=2, label=f'Naive: {beta_hat.mean():.3f}')
        if beta_true is not None:
            axes[1].axvline(np.mean(beta_true), color='red', ls='--', lw=2, label=f'True: {np.mean(beta_true):.3f}')
        axes[1].set_xlabel(r'$\beta(X)$')
        axes[1].set_title(r'Distribution of $\beta(X)$ (Treatment Effect)')
        axes[1].legend()

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        return fig

    def plot_heterogeneity(
        self,
        feature_idx: int = 0,
        n_points: int = 100,
        alpha_true_fn: Callable = None,
        beta_true_fn: Callable = None,
        figsize: Tuple[int, int] = (12, 5),
        save_path: str = None,
    ):
        """
        Plot alpha(X) and beta(X) as functions of a covariate.

        Args:
            feature_idx: Index of feature to vary (others held at mean)
            n_points: Number of points in the x-axis grid
            alpha_true_fn: True alpha function (X -> alpha), optional
            beta_true_fn: True beta function (X -> beta), optional
            figsize: Figure size (width, height)
            save_path: Path to save figure (optional)

        Returns:
            matplotlib Figure object
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("Install matplotlib: pip install deep-inference[plotting]")

        if self._X_train is None:
            raise RuntimeError("X_train not stored. Re-run with store_data=True.")

        # Generate X grid varying one feature, others at mean
        x_col = self._X_train[:, feature_idx]
        x_range = np.linspace(x_col.min(), x_col.max(), n_points)
        X_grid = np.tile(self._X_train.mean(axis=0), (n_points, 1))
        X_grid[:, feature_idx] = x_range

        theta_grid = self.predict_theta(X_grid)

        fig, axes = plt.subplots(1, 2, figsize=figsize)

        axes[0].plot(x_range, theta_grid[:, 0], 'b-', lw=2, label=r'$\hat{\alpha}(X)$')
        if alpha_true_fn:
            axes[0].plot(x_range, alpha_true_fn(X_grid), 'r--', lw=2, label=r'True $\alpha(X)$')
        axes[0].set_xlabel(f'$X_{{{feature_idx}}}$')
        axes[0].set_ylabel(r'$\alpha(X)$')
        axes[0].set_title(r'Baseline Parameter $\alpha(X)$')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(x_range, theta_grid[:, 1], 'g-', lw=2, label=r'$\hat{\beta}(X)$')
        if beta_true_fn:
            axes[1].plot(x_range, beta_true_fn(X_grid), 'r--', lw=2, label=r'True $\beta(X)$')
        axes[1].axhline(self.mu_hat, color='purple', ls=':', lw=2, label=f'$E[\\beta]$ = {self.mu_hat:.3f}')
        axes[1].set_xlabel(f'$X_{{{feature_idx}}}$')
        axes[1].set_ylabel(r'$\beta(X)$')
        axes[1].set_title(r'Treatment Effect $\beta(X)$')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        return fig


def structural_dml_core(
    Y: np.ndarray,
    T: np.ndarray,
    X: np.ndarray,
    loss_fn: Callable[[Tensor, Tensor, Tensor], Tensor],
    target_fn: Callable[[Tensor, Tensor], Tensor],
    theta_dim: int,
    n_folds: int = 50,
    hidden_dims: List[int] = [64, 32],
    epochs: int = 100,
    lr: float = 0.01,
    three_way: Optional[bool] = None,
    gradient_fn: Optional[Callable] = None,
    hessian_fn: Optional[Callable] = None,
    per_obs_target_fn: Optional[Callable] = None,
    per_obs_target_grad_fn: Optional[Callable] = None,
    ridge: float = 1e-4,
    lambda_method: str = 'ridge',
    ridge_alpha: float = 1000.0,
    verbose: bool = False,
) -> DMLResult:
    """
    Core structural DML algorithm with proper cross-fitting.

    Algorithm:
    1. Split data into K folds
    2. For each fold k:
       a. Train theta_k on training data (or theta-split for 3-way)
       b. Compute Hessians on lambda-split
       c. Fit Lambda_k(x) nonparametrically (or aggregate for 2-way)
       d. For held-out i in I_k:
          - Compute psi_i = H - H_theta @ Lambda_k(x_i)^{-1} @ l_theta
    3. Aggregate: mu_hat = mean(psi), SE via within-fold formula

    Args:
        Y: (n,) outcomes
        T: (n,) treatments
        X: (n, d_x) covariates
        loss_fn: Structural loss (y, t, theta) -> (n,) losses
        target_fn: Target function (x, theta) -> scalar
        theta_dim: Dimension of parameter vector
        n_folds: Number of cross-validation folds
        hidden_dims: Neural network hidden layer sizes
        epochs: Training epochs
        lr: Learning rate
        three_way: Use three-way splitting (auto-detect if None)
        gradient_fn: Optional closed-form gradient (falls back to autodiff)
        hessian_fn: Optional closed-form Hessian (falls back to autodiff)
        per_obs_target_fn: Per-observation target h(theta_i, t_i) (default: beta_i)
        per_obs_target_grad_fn: Gradient of h w.r.t. theta (default: (0, 1))
        ridge: Ridge regularization for Hessian inversion
        lambda_method: Method for Lambda estimation ('ridge', 'lgbm', 'rf', 'mlp', 'aggregate').
            Default 'ridge' is recommended for validated coverage.
        ridge_alpha: Regularization strength for Ridge Lambda (default 1000.0)
        verbose: Print progress

    Returns:
        DMLResult with estimates and diagnostics
    """
    n = len(Y)
    d_x = X.shape[1]

    # Convert to tensors
    Y_t = torch.tensor(Y, dtype=torch.float32)
    T_t = torch.tensor(T, dtype=torch.float32)
    X_t = torch.tensor(X, dtype=torch.float32)

    # Auto-detect three-way splitting need
    if three_way is None:
        three_way = detect_theta_dependence(loss_fn, Y_t[:100], T_t[:100], theta_dim)
        if verbose:
            print(f"Auto-detected three_way={three_way}")

    # Create fold indices
    fold_indices = np.zeros(n, dtype=np.int32)
    perm = np.random.permutation(n)
    fold_size = n // n_folds

    for k in range(n_folds):
        start = k * fold_size
        if k == n_folds - 1:
            end = n
        else:
            end = (k + 1) * fold_size
        fold_indices[perm[start:end]] = k

    # Storage for results
    psi_values = np.zeros(n)
    theta_hat_all = np.zeros((n, theta_dim))
    corrections = np.zeros(n)
    lambda_cond_numbers = []
    lambda_min_eigenvalues = []
    n_regularized = 0  # Count of observations needing extra regularization
    histories = []

    # Default per-obs target: h(theta, t) = beta (ignores t)
    if per_obs_target_fn is None:
        def per_obs_target_fn(theta, t):
            return theta[:, 1]

    # Default per-obs target gradient: (0, 1) (ignores t)
    if per_obs_target_grad_fn is None:
        def per_obs_target_grad_fn(theta, t):
            n = theta.shape[0]
            grad = torch.zeros(n, theta_dim, dtype=theta.dtype, device=theta.device)
            grad[:, 1] = 1.0
            return grad

    # Cross-fitting loop
    fold_iterator = range(n_folds)
    if verbose and HAS_TQDM:
        fold_iterator = tqdm(fold_iterator, desc="Cross-fitting", ncols=80)

    for k in fold_iterator:
        if verbose and not HAS_TQDM:
            print(f"Processing fold {k+1}/{n_folds}")

        # Get fold indices
        eval_mask = fold_indices == k
        train_mask = ~eval_mask

        eval_idx = np.where(eval_mask)[0]
        train_idx = np.where(train_mask)[0]

        X_eval = X_t[eval_idx]
        T_eval = T_t[eval_idx]
        Y_eval = Y_t[eval_idx]

        X_train = X_t[train_idx]
        T_train = T_t[train_idx]
        Y_train = Y_t[train_idx]

        # Three-way splitting: split train into theta-train and lambda-train
        if three_way:
            n_train = len(train_idx)
            n_theta = int(0.6 * n_train)
            perm_train = np.random.permutation(n_train)
            theta_idx_local = perm_train[:n_theta]
            lambda_idx_local = perm_train[n_theta:]

            X_theta = X_train[theta_idx_local]
            T_theta = T_train[theta_idx_local]
            Y_theta = Y_train[theta_idx_local]

            X_lambda = X_train[lambda_idx_local]
            T_lambda = T_train[lambda_idx_local]
            Y_lambda = Y_train[lambda_idx_local]
        else:
            # Two-way: use all training data for both
            X_theta = X_train
            T_theta = T_train
            Y_theta = Y_train

            X_lambda = X_train
            T_lambda = T_train
            Y_lambda = Y_train

        # Train structural network
        model = StructuralNet(
            input_dim=d_x,
            theta_dim=theta_dim,
            hidden_dims=hidden_dims,
        )

        history = train_structural_net(
            model=model,
            X=X_theta,
            T=T_theta,
            Y=Y_theta,
            loss_fn=loss_fn,
            epochs=epochs,
            lr=lr,
            verbose=False,
        )
        histories.append(history)

        # Get theta predictions on lambda data and eval data
        model.eval()
        with torch.no_grad():
            theta_lambda = model(X_lambda)
            theta_eval = model(X_eval)

        # Compute Hessians on lambda data
        if hessian_fn is not None:
            hessians_lambda = hessian_fn(Y_lambda, T_lambda, theta_lambda)
        else:
            hessians_lambda = compute_hessian(loss_fn, Y_lambda, T_lambda, theta_lambda)

        # Fit Lambda estimator
        if three_way:
            if lambda_method == 'aggregate':
                # Use aggregate even for three-way (ensures full-rank for binary T)
                lambda_est = AggregateLambdaEstimator(theta_dim=theta_dim)
                lambda_est.fit(X_lambda, hessians_lambda)
            else:
                # Nonparametric Lambda(x)
                lambda_est = LambdaEstimator(
                    method=lambda_method,
                    theta_dim=theta_dim,
                    ridge_alpha=ridge_alpha,
                )
                lambda_est.fit(X_lambda, hessians_lambda)
            Lambda_eval = lambda_est.predict(X_eval)  # (n_eval, d, d)
        else:
            # Aggregate Lambda (same for all x)
            lambda_est = AggregateLambdaEstimator(theta_dim=theta_dim)
            lambda_est.fit(X_lambda, hessians_lambda)
            Lambda_eval = lambda_est.predict(X_eval)  # (n_eval, d, d)

        # Invert Lambda matrices
        Lambda_inv_eval = batch_inverse(Lambda_eval, ridge=ridge)

        # Track condition numbers and min eigenvalues
        for i in range(len(Lambda_eval)):
            s = torch.linalg.svdvals(Lambda_eval[i])
            if s.min() > 1e-10:
                cond = (s.max() / s.min()).item()
            else:
                cond = float('inf')
            lambda_cond_numbers.append(cond)

            # Track minimum eigenvalue
            try:
                eigvals = torch.linalg.eigvalsh(Lambda_eval[i])
                min_eig = eigvals.min().item()
                lambda_min_eigenvalues.append(min_eig)
                if min_eig < 1e-6:
                    n_regularized += 1
            except RuntimeError:
                lambda_min_eigenvalues.append(0.0)
                n_regularized += 1

        # Compute gradients on eval data
        # Need theta with grad for autodiff
        theta_eval_grad = theta_eval.clone().requires_grad_(True)

        if gradient_fn is not None:
            l_theta_eval = gradient_fn(Y_eval, T_eval, theta_eval)
        else:
            l_theta_eval = compute_gradient(loss_fn, Y_eval, T_eval, theta_eval)

        # Compute per-observation target and gradient (pass T for T-dependent targets like AME)
        h_eval = per_obs_target_fn(theta_eval, T_eval)  # (n_eval,)
        h_grad_eval = per_obs_target_grad_fn(theta_eval, T_eval)  # (n_eval, d) or (d,)

        # Handle both per-obs and constant target gradient
        if h_grad_eval.dim() == 1:
            h_grad_eval = h_grad_eval.unsqueeze(0).expand(len(eval_idx), -1)

        # Compute influence function: psi = h - h_grad @ Lambda_inv @ l_theta
        for i in range(len(eval_idx)):
            global_idx = eval_idx[i]

            h_i = h_eval[i].item()
            h_grad_i = h_grad_eval[i]  # (d,)
            Lambda_inv_i = Lambda_inv_eval[i]  # (d, d)
            l_theta_i = l_theta_eval[i]  # (d,)

            # Correction term: h_grad @ Lambda_inv @ l_theta
            correction_i = (h_grad_i @ Lambda_inv_i @ l_theta_i).item()

            psi_i = h_i - correction_i

            psi_values[global_idx] = psi_i
            theta_hat_all[global_idx] = theta_eval[i].detach().numpy()
            corrections[global_idx] = correction_i

    # Aggregate results
    mu_hat = psi_values.mean()

    # Within-fold variance (paper formula)
    variance_sum = 0.0
    for k in range(n_folds):
        psi_k = psi_values[fold_indices == k]
        mu_k = psi_k.mean()
        variance_k = ((psi_k - mu_k) ** 2).mean()
        variance_sum += variance_k

    psi_variance = variance_sum / n_folds
    se = np.sqrt(psi_variance / n)

    # 95% CI
    ci_lower = mu_hat - 1.96 * se
    ci_upper = mu_hat + 1.96 * se

    # Naive estimate for comparison
    mu_naive = theta_hat_all[:, 1].mean()  # Just average beta

    # Diagnostics
    min_lambda_eigenvalue = min(lambda_min_eigenvalues) if lambda_min_eigenvalues else 0.0
    mean_lambda_eigenvalue = np.mean(lambda_min_eigenvalues) if lambda_min_eigenvalues else 0.0

    diagnostics = {
        'lambda_cond_numbers': lambda_cond_numbers,
        'mean_cond_number': np.mean([c for c in lambda_cond_numbers if c < float('inf')]),
        'min_lambda_eigenvalue': min_lambda_eigenvalue,
        'mean_lambda_eigenvalue': mean_lambda_eigenvalue,
        'n_regularized': n_regularized,
        'pct_regularized': 100 * n_regularized / n if n > 0 else 0,
        'correction_mean': corrections.mean(),
        'correction_std': corrections.std(),
        'correction_ratio': corrections.std() / se if se > 0 else 0,
        'three_way': three_way,
        'n_folds': n_folds,
        'histories': histories,
    }

    # Warning for potential instability
    import warnings
    if n_regularized > 0.1 * n:
        warnings.warn(
            f"High Lambda regularization rate ({diagnostics['pct_regularized']:.1f}% of observations). "
            "This may indicate numerical instability. Consider: "
            "(1) larger sample size, (2) more regularization, or (3) simpler model.",
            UserWarning
        )

    if diagnostics['correction_ratio'] > 2.0:
        warnings.warn(
            f"High correction variance ratio ({diagnostics['correction_ratio']:.2f}). "
            "This suggests the influence function correction dominates the estimate variance. "
            "Consider using more cross-fitting folds (K >= 50).",
            UserWarning
        )

    return DMLResult(
        mu_hat=mu_hat,
        se=se,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        psi_values=psi_values,
        theta_hat=theta_hat_all,
        mu_naive=mu_naive,
        diagnostics=diagnostics,
    )


def compute_coverage(mu_true: float, result: DMLResult) -> bool:
    """Check if true mu is within 95% CI."""
    return result.ci_lower <= mu_true <= result.ci_upper


def compute_se_ratio(mu_true: float, results: List[DMLResult]) -> float:
    """
    Compute SE calibration ratio.

    SE_ratio = mean(estimated SE) / empirical SD of point estimates
    Should be close to 1.0 for well-calibrated SEs.
    """
    mu_hats = np.array([r.mu_hat for r in results])
    se_hats = np.array([r.se for r in results])

    empirical_sd = mu_hats.std()
    mean_se = se_hats.mean()

    return mean_se / empirical_sd if empirical_sd > 0 else float('inf')
