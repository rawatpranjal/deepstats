"""
Cross-fitting engine.

Orchestrates 2-way or 3-way cross-fitting based on Lambda regime.

2-way (Regimes A, B):
    For fold k:
        Train: fit θ̂, compute/fit Λ̂
        Test: evaluate ψ

3-way (Regime C):
    For fold k:
        Fold A: fit θ̂
        Fold B: compute Hessians, fit Λ̂
        Test: evaluate ψ
"""

from typing import TYPE_CHECKING, Optional, List, Tuple
from dataclasses import dataclass, field
import numpy as np
import torch
from torch import Tensor

# Optional tqdm import
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

if TYPE_CHECKING:
    from deep_inference.models import StructuralModel
    from deep_inference.targets import Target
    from deep_inference.lambda_ import LambdaStrategy


@dataclass
class FoldResult:
    """Results from a single fold."""

    fold_idx: int
    psi_values: Tensor
    theta_hat: Tensor
    eval_indices: List[int]
    train_history: Optional[dict] = None


@dataclass
class CrossFitResult:
    """Complete cross-fitting results."""

    psi_values: Tensor  # (n,) all ψ values
    theta_hat: Tensor  # (n, d_theta) all θ̂ values
    mu_hat: float
    se: float
    ci_lower: float
    ci_upper: float
    fold_results: List[FoldResult]

    def __repr__(self) -> str:
        """Short representation."""
        return (
            f"<CrossFitResult: mu_hat={self.mu_hat:.4f}, se={self.se:.4f}, "
            f"95% CI=[{self.ci_lower:.4f}, {self.ci_upper:.4f}]>"
        )

    def summary(self) -> str:
        """Generate cross-fitting summary."""
        n_folds = len(self.fold_results)
        n_obs = len(self.psi_values)

        lines = [
            "Cross-Fitting Results:",
            f"  Number of folds:    {n_folds}",
            f"  Observations:       {n_obs}",
            f"  Point estimate:     {self.mu_hat:.6f}",
            f"  Standard error:     {self.se:.6f}",
            f"  95% CI:             [{self.ci_lower:.6f}, {self.ci_upper:.6f}]",
        ]

        # Add fold-level info if available
        if self.fold_results and self.fold_results[0].train_history:
            val_losses = [
                fr.train_history.get("val_loss", float("nan"))
                for fr in self.fold_results
                if fr.train_history
            ]
            if val_losses:
                mean_val = np.mean(val_losses)
                lines.append(f"  Mean val loss:      {mean_val:.6f}")

        return "\n".join(lines)


class CrossFitter:
    """
    Cross-fitting orchestrator.

    Handles both 2-way and 3-way splitting based on Lambda strategy.
    """

    def __init__(
        self,
        n_folds: int = 5,
        three_way_theta_frac: float = 0.6,
        shuffle: bool = True,
        random_state: Optional[int] = None,
    ):
        """
        Initialize cross-fitter.

        Args:
            n_folds: Number of folds (K)
            three_way_theta_frac: Fraction for theta training in 3-way split
            shuffle: Whether to shuffle data before splitting
            random_state: Random seed
        """
        self.n_folds = n_folds
        self.three_way_theta_frac = three_way_theta_frac
        self.shuffle = shuffle
        self.random_state = random_state

    def create_folds(self, n: int) -> List[np.ndarray]:
        """
        Create fold indices.

        Args:
            n: Total number of observations

        Returns:
            List of arrays, each containing indices for one fold
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)

        indices = np.arange(n)
        if self.shuffle:
            np.random.shuffle(indices)

        # Split into K folds
        fold_size = n // self.n_folds
        folds = []

        for k in range(self.n_folds):
            start = k * fold_size
            if k == self.n_folds - 1:
                # Last fold gets remainder
                end = n
            else:
                end = (k + 1) * fold_size
            folds.append(indices[start:end])

        return folds

    def split_three_way(
        self, train_indices: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Split training indices for 3-way cross-fitting.

        Args:
            train_indices: Indices not in test fold

        Returns:
            (theta_indices, lambda_indices)
        """
        n_train = len(train_indices)
        n_theta = int(n_train * self.three_way_theta_frac)

        # Shuffle before splitting
        shuffled = train_indices.copy()
        np.random.shuffle(shuffled)

        theta_indices = shuffled[:n_theta]
        lambda_indices = shuffled[n_theta:]

        return theta_indices, lambda_indices


def run_crossfit(
    Y: Tensor,
    T: Tensor,
    X: Tensor,
    t_tilde: Tensor,
    model: "StructuralModel",
    target: "Target",
    lambda_strategy: "LambdaStrategy",
    n_folds: int = 5,
    epochs: int = 100,
    lr: float = 0.01,
    patience: int = 50,
    hidden_dims: List[int] = [64, 32],
    ridge: float = 1e-4,
    verbose: bool = False,
) -> CrossFitResult:
    """
    Run cross-fitting procedure.

    Args:
        Y: (n,) outcomes
        T: (n,) treatments
        X: (n, d_x) covariates
        t_tilde: Evaluation point
        model: Structural model
        target: Target functional
        lambda_strategy: Lambda estimation strategy
        n_folds: Number of folds
        epochs: Training epochs per fold
        lr: Learning rate
        hidden_dims: Hidden layer sizes
        ridge: Regularization for Lambda inversion
        verbose: Print progress

    Returns:
        CrossFitResult with all inference outputs
    """
    from deep_inference.models import StructuralNet, train_structural_net
    from deep_inference.engine.assembler import compute_psi
    from deep_inference.engine.variance import compute_inference_results

    n = Y.shape[0]
    d_x = X.shape[1]
    d_theta = model.theta_dim
    device = Y.device
    dtype = Y.dtype

    # Create cross-fitter
    crossfitter = CrossFitter(n_folds=n_folds)
    folds = crossfitter.create_folds(n)

    # Determine if 3-way split needed
    three_way = lambda_strategy.requires_separate_fold

    # Storage for all observations
    all_psi = torch.zeros(n, dtype=dtype, device=device)
    all_theta = torch.zeros(n, d_theta, dtype=dtype, device=device)
    fold_results = []

    # Cross-fitting loop
    fold_iterator = list(enumerate(folds))
    if verbose and HAS_TQDM:
        fold_iterator = tqdm(fold_iterator, desc="Cross-fitting", ncols=80)

    for k, eval_fold in fold_iterator:
        if verbose and not HAS_TQDM:
            print(f"Fold {k+1}/{n_folds}")

        # Get training indices (everything except eval fold)
        train_mask = np.ones(n, dtype=bool)
        train_mask[eval_fold] = False
        train_indices = np.where(train_mask)[0]

        if three_way:
            # 3-way split: separate theta and lambda training
            theta_indices, lambda_indices = crossfitter.split_three_way(train_indices)
        else:
            # 2-way split: use all training data for both
            theta_indices = train_indices
            lambda_indices = train_indices

        # Get data subsets
        X_theta = X[theta_indices]
        T_theta = T[theta_indices]
        Y_theta = Y[theta_indices]

        X_lambda = X[lambda_indices]
        T_lambda = T[lambda_indices]
        Y_lambda = Y[lambda_indices]

        X_eval = X[eval_fold]
        T_eval = T[eval_fold]
        Y_eval = Y[eval_fold]

        # 1. Train theta network
        theta_net = StructuralNet(
            input_dim=d_x,
            theta_dim=d_theta,
            hidden_dims=hidden_dims,
        )

        def loss_fn_batched(y, t, theta):
            """Batched loss for training."""
            losses = torch.zeros(len(y), dtype=dtype, device=device)
            for i in range(len(y)):
                losses[i] = model.loss(y[i], t[i], theta[i])
            return losses

        history = train_structural_net(
            model=theta_net,
            X=X_theta,
            T=T_theta,
            Y=Y_theta,
            loss_fn=loss_fn_batched,
            epochs=epochs,
            lr=lr,
            patience=patience,
            verbose=False,
        )

        # Get theta predictions
        with torch.no_grad():
            theta_hat_lambda = theta_net(X_lambda)
            theta_hat_eval = theta_net(X_eval)

        # 2. Fit/compute Lambda
        if lambda_strategy.requires_theta:
            lambda_strategy.fit(
                X=X_lambda,
                T=T_lambda,
                Y=Y_lambda,
                theta_hat=theta_hat_lambda,
                model=model,
            )
        else:
            lambda_strategy.fit(
                X=X_lambda,
                T=T_lambda,
                Y=Y_lambda,
                theta_hat=None,
                model=model,
            )

        # 3. Get Lambda for eval observations
        lambda_eval = lambda_strategy.predict(X_eval, theta_hat_eval)

        # 4. Compute psi on eval fold
        psi_eval = compute_psi(
            Y=Y_eval,
            T=T_eval,
            X=X_eval,
            theta_hat=theta_hat_eval,
            t_tilde=t_tilde,
            lambda_matrices=lambda_eval,
            model=model,
            target=target,
            ridge=ridge,
        )

        # Store results
        all_psi[eval_fold] = psi_eval
        all_theta[eval_fold] = theta_hat_eval

        fold_results.append(
            FoldResult(
                fold_idx=k,
                psi_values=psi_eval,
                theta_hat=theta_hat_eval,
                eval_indices=list(eval_fold),
                train_history={"val_loss": history.val_losses[-1]} if history.val_losses else None,
            )
        )

    # Compute final inference results
    results = compute_inference_results(all_psi)

    return CrossFitResult(
        psi_values=all_psi,
        theta_hat=all_theta,
        mu_hat=results["mu_hat"],
        se=results["se"],
        ci_lower=results["ci_lower"],
        ci_upper=results["ci_upper"],
        fold_results=fold_results,
    )
