"""Double Machine Learning estimator.

This module implements the Double/Debiased Machine Learning (DML)
estimator for causal inference with neural networks.

References
----------
- Chernozhukov et al. (2018). "Double/Debiased Machine Learning for
  Treatment and Structural Parameters"
- Farrell, Liang, Misra (2021). "Deep Neural Networks for Estimation
  and Inference"
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from .._typing import Float64Array, Learner
from ..networks.mlp import MLP, MLPClassifier
from ..results.deep_results import DeepResults


class CausalResults:
    """Results container for causal inference.

    Attributes
    ----------
    ate : float
        Average treatment effect estimate.
    ate_se : float
        Standard error of ATE.
    att : float, optional
        Average treatment effect on the treated.
    att_se : float, optional
        Standard error of ATT.
    scores : Float64Array
        Influence function scores for each observation.
    n_obs : int
        Number of observations.
    n_folds : int
        Number of cross-fitting folds.
    """

    def __init__(
        self,
        ate: float,
        ate_se: float,
        scores: Float64Array,
        n_obs: int,
        n_folds: int,
        att: float | None = None,
        att_se: float | None = None,
        model_y_: Any = None,
        model_t_: Any = None,
    ) -> None:
        self.ate = ate
        self.ate_se = ate_se
        self.att = att
        self.att_se = att_se
        self.scores = scores
        self.n_obs = n_obs
        self.n_folds = n_folds
        self.model_y_ = model_y_
        self.model_t_ = model_t_

    @property
    def pvalue(self) -> float:
        """Two-sided p-value for H0: ATE = 0."""
        from scipy import stats
        return float(2 * (1 - stats.norm.cdf(abs(self.ate / self.ate_se))))

    def confint(self, alpha: float = 0.05) -> tuple[float, float]:
        """Confidence interval for ATE."""
        from scipy import stats
        z = stats.norm.ppf(1 - alpha / 2)
        return (self.ate - z * self.ate_se, self.ate + z * self.ate_se)

    def summary(self) -> str:
        """Generate summary table."""
        ci = self.confint()
        lines = [
            "=" * 60,
            "       Double Machine Learning Results",
            "=" * 60,
            f"ATE:              {self.ate:.6f}",
            f"Std. Error:       {self.ate_se:.6f}",
            f"t-statistic:      {self.ate / self.ate_se:.3f}",
            f"P-value:          {self.pvalue:.4f}",
            f"95% CI:           [{ci[0]:.4f}, {ci[1]:.4f}]",
            "-" * 60,
            f"No. Observations: {self.n_obs}",
            f"No. Folds:        {self.n_folds}",
            "=" * 60,
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"CausalResults(ate={self.ate:.4f}, se={self.ate_se:.4f})"


class DoubleMachineLearning(BaseEstimator):
    """Double/Debiased Machine Learning for ATE estimation.

    This meta-estimator implements the DML framework with neural networks.
    It accepts arbitrary sklearn-compatible learners for nuisance estimation
    (outcome and propensity models).

    The key insight of DML is to use cross-fitting to avoid overfitting
    bias when using flexible ML methods for nuisance estimation.

    Parameters
    ----------
    outcome_learner : estimator, optional
        Model for E[Y|X]. Default is MLP.
    treatment_learner : estimator, optional
        Model for E[T|X] (propensity score). Default is MLPClassifier.
    n_folds : int, default=5
        Number of cross-fitting folds.
    epochs : int, default=100
        Training epochs for neural network learners.
    lr : float, default=1e-3
        Learning rate for neural network learners.
    random_state : int, optional
        Random seed for reproducibility.
    verbose : int, default=1
        Verbosity level.

    Attributes
    ----------
    results_ : CausalResults
        Estimation results after fitting.
    model_y_ : list
        Fitted outcome models (one per fold).
    model_t_ : list
        Fitted treatment models (one per fold).

    Examples
    --------
    >>> from deepstats.estimators import DoubleMachineLearning
    >>> import numpy as np
    >>>
    >>> # Generate data with known ATE
    >>> n = 1000
    >>> X = np.random.randn(n, 5)
    >>> T = (np.random.rand(n) > 0.5).astype(float)
    >>> Y = 2 * T + X[:, 0] + np.random.randn(n) * 0.5  # True ATE = 2
    >>>
    >>> dml = DoubleMachineLearning(n_folds=5)
    >>> result = dml.fit(Y=Y, T=T, X=X)
    >>> print(result.summary())

    Notes
    -----
    DML uses the "partially linear" model:
        Y = T * theta + g(X) + epsilon
        T = m(X) + eta

    where theta is the causal parameter (ATE), g(X) is the outcome
    confounding, and m(X) is the propensity score.
    """

    def __init__(
        self,
        outcome_learner: Learner | None = None,
        treatment_learner: Learner | None = None,
        n_folds: int = 5,
        epochs: int = 100,
        lr: float = 1e-3,
        random_state: int | None = None,
        verbose: int = 1,
    ) -> None:
        self.outcome_learner = outcome_learner
        self.treatment_learner = treatment_learner
        self.n_folds = n_folds
        self.epochs = epochs
        self.lr = lr
        self.random_state = random_state
        self.verbose = verbose

    def fit(
        self,
        Y: Float64Array,
        T: Float64Array,
        X: Float64Array,
    ) -> CausalResults:
        """Fit the DML model.

        Parameters
        ----------
        Y : Float64Array
            Outcome variable (n,).
        T : Float64Array
            Treatment indicator (n,). Binary 0/1.
        X : Float64Array
            Covariates/confounders (n, p).

        Returns
        -------
        CausalResults
            Results object with ATE, standard error, and inference methods.
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)
            torch.manual_seed(self.random_state)

        Y = np.asarray(Y).flatten()
        T = np.asarray(T).flatten()
        X = np.asarray(X)

        n = len(Y)
        p = X.shape[1]

        # Determine device
        device = self._get_device()

        # Initialize residual arrays
        y_residuals = np.zeros(n)
        t_residuals = np.zeros(n)

        # Cross-fitting
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)

        self.model_y_: list[Any] = []
        self.model_t_: list[Any] = []

        if self.verbose >= 1:
            print(f"Cross-fitting with {self.n_folds} folds...")

        for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X)):
            if self.verbose >= 1:
                print(f"  Fold {fold_idx + 1}/{self.n_folds}")

            # Split data
            X_train, X_test = X[train_idx], X[test_idx]
            Y_train, Y_test = Y[train_idx], Y[test_idx]
            T_train, T_test = T[train_idx], T[test_idx]

            # Fit outcome model E[Y|X]
            model_y = self._fit_outcome_model(X_train, Y_train, p, device)
            y_pred = self._predict(model_y, X_test, device)
            y_residuals[test_idx] = Y_test - y_pred
            self.model_y_.append(model_y)

            # Fit treatment model E[T|X] (propensity)
            model_t = self._fit_treatment_model(X_train, T_train, p, device)
            t_pred = self._predict(model_t, X_test, device, is_classifier=True)
            t_residuals[test_idx] = T_test - t_pred
            self.model_t_.append(model_t)

        # Estimate ATE using residuals (Robinson transformation)
        # theta = E[Y_res * T_res] / E[T_res^2]
        ate = np.mean(y_residuals * t_residuals) / np.mean(t_residuals**2)

        # Compute influence function scores for inference
        scores = (y_residuals - ate * t_residuals) * t_residuals / np.mean(t_residuals**2)

        # Standard error via influence function
        ate_se = np.std(scores) / np.sqrt(n)

        self.results_ = CausalResults(
            ate=ate,
            ate_se=ate_se,
            scores=scores,
            n_obs=n,
            n_folds=self.n_folds,
            model_y_=self.model_y_,
            model_t_=self.model_t_,
        )

        return self.results_

    def _get_device(self) -> torch.device:
        """Determine computation device."""
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _fit_outcome_model(
        self,
        X: Float64Array,
        Y: Float64Array,
        p: int,
        device: torch.device,
    ) -> nn.Module:
        """Fit the outcome model."""
        if self.outcome_learner is not None:
            # Use provided learner (sklearn-compatible)
            model = clone(self.outcome_learner)
            model.fit(X, Y)
            return model

        # Default: train MLP
        model = MLP(input_dim=p, hidden_dims=[64, 32]).to(device)
        self._train_network(model, X, Y, device, is_classifier=False)
        return model.cpu()

    def _fit_treatment_model(
        self,
        X: Float64Array,
        T: Float64Array,
        p: int,
        device: torch.device,
    ) -> nn.Module:
        """Fit the treatment/propensity model."""
        if self.treatment_learner is not None:
            model = clone(self.treatment_learner)
            model.fit(X, T)
            return model

        # Default: train MLPClassifier
        model = MLPClassifier(input_dim=p, hidden_dims=[32, 16]).to(device)
        self._train_network(model, X, T, device, is_classifier=True)
        return model.cpu()

    def _train_network(
        self,
        model: nn.Module,
        X: Float64Array,
        y: Float64Array,
        device: torch.device,
        is_classifier: bool,
    ) -> None:
        """Train a neural network."""
        X_tensor = torch.from_numpy(X).float().to(device)
        y_tensor = torch.from_numpy(y).float().to(device).unsqueeze(1)

        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

        optimizer = optim.Adam(model.parameters(), lr=self.lr)

        if is_classifier:
            loss_fn = nn.BCELoss()
        else:
            loss_fn = nn.MSELoss()

        model.train()
        for _ in range(self.epochs):
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                y_pred = model(batch_X)
                loss = loss_fn(y_pred, batch_y)
                loss.backward()
                optimizer.step()

    def _predict(
        self,
        model: Any,
        X: Float64Array,
        device: torch.device,
        is_classifier: bool = False,
    ) -> Float64Array:
        """Generate predictions from a model."""
        # Check if it's an sklearn-compatible model
        if hasattr(model, "predict") and not isinstance(model, nn.Module):
            if is_classifier and hasattr(model, "predict_proba"):
                return model.predict_proba(X)[:, 1]
            return model.predict(X)

        # PyTorch model
        model = model.to(device)
        model.eval()
        with torch.no_grad():
            X_tensor = torch.from_numpy(X).float().to(device)
            pred = model(X_tensor).squeeze().cpu().numpy()
        return pred

    def refute(self, method: str = "placebo") -> dict[str, Any]:
        """Run refutation/sensitivity analysis.

        Parameters
        ----------
        method : str, default="placebo"
            Refutation method:
            - "placebo": Replace treatment with random treatment
            - "subset": Remove random subset of data

        Returns
        -------
        dict
            Refutation results with original and refuted estimates.
        """
        if not hasattr(self, "results_"):
            raise ValueError("Model must be fitted before refutation")

        # This is a placeholder for full refutation implementation
        # In production, would implement various sensitivity analyses
        raise NotImplementedError("Refutation methods coming in future version")
