"""Mixin for prediction and visualization methods.

Provides shared functionality for DMLResult and InferenceResult classes.
"""

from typing import Callable, Optional, Tuple, Any
import numpy as np


class PredictVisualizeMixin:
    """
    Mixin providing prediction and visualization methods.

    Requires implementing classes to have:
        - theta_hat: np.ndarray or Tensor (n, theta_dim)
        - _X_train: Optional[np.ndarray] (n, d_x)
        - _theta_predictor: Optional[sklearn Ridge model]
        - mu_hat: float (for plotting)

    And one of:
        - _family: Optional[str] (DMLResult)
        - _model: Optional[str] (InferenceResult)
    """

    @property
    def _family_name(self) -> Optional[str]:
        """Get family/model name (handles both _family and _model attrs)."""
        return getattr(self, '_family', None) or getattr(self, '_model', None)

    def _get_theta_hat_array(self) -> np.ndarray:
        """Convert theta_hat to numpy (handles Tensor or ndarray)."""
        theta = self.theta_hat
        if hasattr(theta, 'detach'):
            # It's a PyTorch Tensor
            return theta.detach().cpu().numpy()
        return np.asarray(theta)

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
        self._theta_predictor.fit(self._X_train, self._get_theta_hat_array())

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
        family = self._family_name
        if family not in ('logit', 'probit'):
            raise ValueError(f"predict_proba requires logit/probit, got '{family}'")
        theta = self.predict_theta(X_new)
        T = np.full(len(theta), t_value) if T_new is None else np.asarray(T_new)
        logits = theta[:, 0] + theta[:, 1] * T
        if family == 'logit':
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
        family = self._family_name

        if family in ('logit', 'probit'):
            return self.predict_proba(X_new, T_new, t_value)
        elif family in ('poisson', 'negbin'):
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

        theta_hat = self._get_theta_hat_array()
        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # Alpha distribution
        alpha_hat = theta_hat[:, 0]
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
        beta_hat = theta_hat[:, 1]
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
