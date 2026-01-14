"""
Regime C: Estimate Lambda via neural network regression.

For observational data with nonlinear models where:
- Hessian depends on theta
- We cannot compute Λ analytically

Λ(x) = E[ℓ_θθ(Y, T, θ(X)) | X = x]

This requires 3-way cross-fitting:
1. Fit θ̂ on fold A
2. Compute Hessians on fold B using θ̂
3. Fit Λ̂ by regressing Hessians on X in fold B
4. Evaluate ψ on fold C
"""

from typing import Optional, TYPE_CHECKING, Literal
import warnings

# Suppress sklearn warnings about feature names (LightGBM fitted with names, predict without)
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

import torch
from torch import Tensor

from .base import BaseLambdaStrategy

if TYPE_CHECKING:
    from deep_inference.models import StructuralModel


class EstimateLambda(BaseLambdaStrategy):
    """
    Regime C: Estimate Lambda via regression.

    Fits a model to predict Λ(x) = E[ℓ_θθ | X=x] from covariates X.

    IMPORTANT: This requires 3-way cross-fitting because:
    - Hessians depend on θ̂
    - Fitting both θ̂ and Λ̂ on same data causes bias
    """

    requires_theta = True
    requires_separate_fold = True  # KEY: Needs 3-way split!

    def __init__(
        self,
        method: Literal["mlp", "rf", "ridge", "aggregate", "lgbm"] = "ridge",
        ridge_alpha: float = 1000.0,
        mlp_alpha: float = 0.0001,
        rf_max_depth: Optional[int] = 10,
        lgbm_reg_lambda: float = 0.0,
    ):
        """
        Initialize EstimateLambda strategy.

        Args:
            method: Regression method ("ridge" [default], "aggregate", "lgbm", "mlp", "rf")
                    "ridge" is recommended for validated coverage.
                    "aggregate" is stable when Hessian doesn't depend on X.
                    "mlp" can produce invalid standard errors - use with caution.
            ridge_alpha: L2 regularization for ridge regression (default 1000.0)
            mlp_alpha: L2 regularization for MLP (default 0.0001)
            rf_max_depth: Max tree depth for RF (default 10, None=unlimited)
            lgbm_reg_lambda: L2 regularization for LightGBM (default 0.0)
        """
        # Warn about potentially dangerous methods
        if method in ("mlp", "neural"):
            warnings.warn(
                f"Lambda method '{method}' can produce invalid standard errors "
                "despite high correlation with oracle. Consider method='ridge' "
                "(default) or 'lgbm' for validated coverage. "
                "See docs/algorithm/index.md for details.",
                UserWarning
            )

        self.method = method
        self.ridge_alpha = ridge_alpha
        self.mlp_alpha = mlp_alpha
        self.rf_max_depth = rf_max_depth
        self.lgbm_reg_lambda = lgbm_reg_lambda
        self._model = None
        self._mean_hessian = None
        self._d_theta = None
        self._triu_idx = None

    def fit(
        self,
        X: Tensor,
        T: Tensor,
        Y: Tensor,
        theta_hat: Optional[Tensor],
        model: "StructuralModel",
    ) -> None:
        """
        Fit the Lambda estimator.

        1. Compute Hessians at each observation using θ̂
        2. Fit regression model: X → Hessian

        Args:
            X: (n, d_x) covariates
            T: (n,) treatments
            Y: (n,) outcomes
            theta_hat: (n, d_theta) estimated parameters
            model: The structural model
        """
        if theta_hat is None:
            raise ValueError("EstimateLambda requires theta_hat")

        n = X.shape[0]
        self._d_theta = theta_hat.shape[1]
        d_theta = self._d_theta
        device = X.device
        dtype = X.dtype

        # Compute Hessians at each observation
        hessians = torch.zeros(n, d_theta, d_theta, dtype=dtype, device=device)

        # Try closed-form first
        closed_form_available = True
        for i in range(min(n, 1)):  # Check first observation
            h = model.hessian(Y[i], T[i], theta_hat[i])
            if h is None:
                closed_form_available = False
                break

        if closed_form_available:
            # Use closed-form Hessians
            for i in range(n):
                hessians[i] = model.hessian(Y[i], T[i], theta_hat[i])
        else:
            # Fall back to autodiff
            from deep_inference.autodiff.hessian import compute_hessian_vmap
            hessians = compute_hessian_vmap(model.loss, Y, T, theta_hat)

        # Store upper triangle indices for reconstruction
        self._triu_idx = torch.triu_indices(d_theta, d_theta)

        if self.method == "aggregate":
            # Simple: Λ = mean(Hessians)
            self._mean_hessian = hessians.mean(dim=0)

        elif self.method == "mlp":
            self._fit_mlp(X, hessians)

        elif self.method == "ridge":
            self._fit_ridge(X, hessians)

        elif self.method == "rf":
            self._fit_rf(X, hessians)

        elif self.method == "lgbm":
            self._fit_lgbm(X, hessians)

    def _fit_mlp(self, X: Tensor, hessians: Tensor) -> None:
        """Fit MLP regression for Λ(x)."""
        from sklearn.neural_network import MLPRegressor
        import numpy as np

        d_theta = hessians.shape[1]

        # Flatten to upper triangle
        targets = hessians[:, self._triu_idx[0], self._triu_idx[1]]

        # Convert to numpy
        X_np = X.detach().cpu().numpy()
        targets_np = targets.detach().cpu().numpy()

        self._mlp = MLPRegressor(
            hidden_layer_sizes=(64, 32),
            max_iter=200,
            early_stopping=True,
            alpha=self.mlp_alpha,  # L2 regularization
            random_state=42,
        )
        self._mlp.fit(X_np, targets_np)

    def _fit_ridge(self, X: Tensor, hessians: Tensor) -> None:
        """Fit ridge regression for Λ(x)."""
        from sklearn.linear_model import Ridge

        targets = hessians[:, self._triu_idx[0], self._triu_idx[1]]

        X_np = X.detach().cpu().numpy()
        targets_np = targets.detach().cpu().numpy()

        self._ridge = Ridge(alpha=self.ridge_alpha)
        self._ridge.fit(X_np, targets_np)

    def _fit_rf(self, X: Tensor, hessians: Tensor) -> None:
        """Fit random forest for Λ(x)."""
        from sklearn.ensemble import RandomForestRegressor

        targets = hessians[:, self._triu_idx[0], self._triu_idx[1]]

        X_np = X.detach().cpu().numpy()
        targets_np = targets.detach().cpu().numpy()

        self._rf = RandomForestRegressor(
            n_estimators=100,
            max_depth=self.rf_max_depth,  # Depth regularization
            random_state=42,
        )
        self._rf.fit(X_np, targets_np)

    def _fit_lgbm(self, X: Tensor, hessians: Tensor) -> None:
        """Fit LightGBM for Λ(x)."""
        from lightgbm import LGBMRegressor
        from sklearn.multioutput import MultiOutputRegressor

        targets = hessians[:, self._triu_idx[0], self._triu_idx[1]]

        X_np = X.detach().cpu().numpy()
        targets_np = targets.detach().cpu().numpy()

        # Wrap in MultiOutputRegressor for multi-target support
        base_lgbm = LGBMRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            reg_lambda=self.lgbm_reg_lambda,  # L2 regularization
            random_state=42,
            verbose=-1,
        )
        self._lgbm = MultiOutputRegressor(base_lgbm)
        self._lgbm.fit(X_np, targets_np)

    def predict(self, X: Tensor, theta_hat: Optional[Tensor] = None) -> Tensor:
        """
        Predict Λ(x) for new observations.

        Args:
            X: (n, d_x) covariates
            theta_hat: Not used for prediction (already fitted)

        Returns:
            (n, d_theta, d_theta) Lambda matrices (guaranteed PSD)
        """
        n = X.shape[0]
        d_theta = self._d_theta
        device = X.device
        dtype = X.dtype

        Lambda = torch.zeros(n, d_theta, d_theta, dtype=dtype, device=device)

        if self.method == "aggregate":
            Lambda[:] = self._mean_hessian

        elif self.method == "mlp":
            X_np = X.detach().cpu().numpy()
            pred = self._mlp.predict(X_np)
            pred = torch.tensor(pred, dtype=dtype, device=device)
            Lambda[:, self._triu_idx[0], self._triu_idx[1]] = pred
            Lambda[:, self._triu_idx[1], self._triu_idx[0]] = pred

        elif self.method == "ridge":
            X_np = X.detach().cpu().numpy()
            pred = self._ridge.predict(X_np)
            pred = torch.tensor(pred, dtype=dtype, device=device)
            Lambda[:, self._triu_idx[0], self._triu_idx[1]] = pred
            Lambda[:, self._triu_idx[1], self._triu_idx[0]] = pred

        elif self.method == "rf":
            X_np = X.detach().cpu().numpy()
            pred = self._rf.predict(X_np)
            pred = torch.tensor(pred, dtype=dtype, device=device)
            Lambda[:, self._triu_idx[0], self._triu_idx[1]] = pred
            Lambda[:, self._triu_idx[1], self._triu_idx[0]] = pred

        elif self.method == "lgbm":
            X_np = X.detach().cpu().numpy()
            pred = self._lgbm.predict(X_np)
            pred = torch.tensor(pred, dtype=dtype, device=device)
            Lambda[:, self._triu_idx[0], self._triu_idx[1]] = pred
            Lambda[:, self._triu_idx[1], self._triu_idx[0]] = pred

        # Project to PSD to ensure valid matrices (fixes numerical instability)
        Lambda = self._project_to_psd(Lambda)

        return Lambda

    def _shrink_lambda(
        self,
        Lambda: Tensor,
        shrinkage: float = 0.1,
        shrinkage_target: str = "scaled_identity",
    ) -> Tensor:
        """
        Apply Ledoit-Wolf style shrinkage toward a well-conditioned target.

        Shrinkage reduces extreme eigenvalues while preserving trace,
        providing a bias-variance tradeoff.

        Args:
            Lambda: (n, d, d) batch of matrices
            shrinkage: Shrinkage intensity α ∈ [0, 1]
                       Λ_shrunk = (1 - α) * Λ + α * target
            shrinkage_target: Target type
                - "scaled_identity": target = (trace/d) * I
                - "diagonal": target = diag(Λ)

        Returns:
            (n, d, d) shrunk matrices
        """
        n, d, _ = Lambda.shape
        device = Lambda.device
        dtype = Lambda.dtype

        if shrinkage_target == "scaled_identity":
            # Target = (trace/d) * I for each matrix
            traces = torch.einsum('nii->n', Lambda)  # (n,)
            eye = torch.eye(d, dtype=dtype, device=device)
            targets = (traces / d).unsqueeze(-1).unsqueeze(-1) * eye.unsqueeze(0)
        elif shrinkage_target == "diagonal":
            # Target = diag(Λ)
            diag_vals = torch.diagonal(Lambda, dim1=1, dim2=2)  # (n, d)
            targets = torch.diag_embed(diag_vals)  # (n, d, d)
        else:
            raise ValueError(f"Unknown shrinkage_target: {shrinkage_target}")

        return (1 - shrinkage) * Lambda + shrinkage * targets

    def _project_to_psd(
        self,
        Lambda: Tensor,
        min_eigenvalue: float = 1e-4,
        max_condition: float = 100.0,
        use_relative: bool = True,
    ) -> Tensor:
        """
        Project matrices to the nearest positive semi-definite matrices.

        Uses eigendecomposition and clamps eigenvalues.

        Args:
            Lambda: (n, d, d) batch of matrices
            min_eigenvalue: Minimum eigenvalue for absolute floor (used if use_relative=False)
            max_condition: Maximum condition number for relative floor (used if use_relative=True)
            use_relative: If True, use relative floor (min_eig = max_eig / max_condition)
                          If False, use absolute floor (min_eig = min_eigenvalue)

        Returns:
            (n, d, d) PSD matrices with bounded condition number
        """
        n = Lambda.shape[0]
        Lambda_psd = Lambda.clone()

        for i in range(n):
            # Eigendecomposition
            eigvals, eigvecs = torch.linalg.eigh(Lambda[i])

            if use_relative:
                # Relative floor: bound condition number
                max_eig = eigvals[-1]  # Largest eigenvalue (eigh returns sorted)
                min_allowed = max_eig / max_condition
                # Still enforce a small absolute floor to avoid numerical issues
                min_allowed = max(min_allowed, 1e-10)
                eigvals_clamped = torch.clamp(eigvals, min=min_allowed)
            else:
                # Absolute floor (legacy behavior)
                eigvals_clamped = torch.clamp(eigvals, min=min_eigenvalue)

            # Reconstruct: V @ diag(D) @ V'
            Lambda_psd[i] = eigvecs @ torch.diag(eigvals_clamped) @ eigvecs.T

        return Lambda_psd
