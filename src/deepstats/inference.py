"""Inference methods: naive, influence, bootstrap.

Three approaches to estimate E[β(X)] and its standard error.

NOT DML! This implements the FLM Influence Function approach where neural nets
output structural parameters directly and the influence function corrects for
regularization bias.
"""

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
from sklearn.model_selection import KFold

from .models import StructuralNet, NuisanceNet, train_structural, train_nuisance


# =============================================================================
# Result Containers
# =============================================================================

@dataclass
class InferenceResult:
    """Result from any inference method."""
    mu_hat: float
    se: float
    values: np.ndarray  # beta values (naive/bootstrap) or psi scores (influence)


# =============================================================================
# Naive Estimator
# =============================================================================

def naive(
    X: np.ndarray,
    T: np.ndarray,
    Y: np.ndarray,
    family,
    config,
) -> Tuple[float, float]:
    """Naive plug-in estimator.

    1. Train model on full data
    2. μ̂ = mean(β(X))
    3. SE = std(β) / √n

    This underestimates uncertainty because it ignores regularization bias.
    Expected: ~30-50% coverage instead of 95%.
    """
    n = len(X)

    model = StructuralNet(
        input_dim=X.shape[1],
        hidden_dims=config.hidden_dims,
        dropout=config.dropout,
    )

    train_structural(
        model=model,
        family=family,
        X=X, T=T, Y=Y,
        epochs=config.epochs,
        lr=config.lr,
        batch_size=config.batch_size,
        weight_decay=config.weight_decay,
    )

    model.eval()
    with torch.no_grad():
        theta = model(torch.FloatTensor(X))
        beta = theta[:, 1].numpy()

    mu_hat = float(np.mean(beta))
    se = float(np.std(beta, ddof=1) / np.sqrt(n))

    return mu_hat, se


# =============================================================================
# Influence Function Estimator (50-Fold Cross-Fitting)
# =============================================================================

def influence(
    X: np.ndarray,
    T: np.ndarray,
    Y: np.ndarray,
    family,
    config,
) -> Tuple[float, float]:
    """Influence function estimator with 50-fold structural cross-fitting.

    The FLM protocol (NOT DML):
    1. K=50 folds (each model sees 98% of data)
    2. For each fold k:
       - Train structural model on D_train (folds ≠ k)
       - Compute Hessian Λ on same D_train (two-way, NOT three-way split)
       - Compute ψᵢ on D_test (fold k)
    3. μ̂ = mean(ψ), SE = std(ψ) / √n

    The influence score corrects for regularization bias:
    ψᵢ = Ĥ(θ̂ᵢ) + Ĥ_θᵀ · Λ̂⁻¹ · ℓ̂_θ,ᵢ

    Expected: ~95% coverage.
    """
    n = len(X)
    psi_all = np.zeros(n)

    n_folds = getattr(config, 'n_folds', 50)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=config.seed)

    for train_idx, eval_idx in kf.split(X):
        X_tr, T_tr, Y_tr = X[train_idx], T[train_idx], Y[train_idx]
        X_ev, T_ev, Y_ev = X[eval_idx], T[eval_idx], Y[eval_idx]

        # Step A: Train structural model on D_train
        struct_model = StructuralNet(
            input_dim=X.shape[1],
            hidden_dims=config.hidden_dims,
            dropout=config.dropout,
        )
        train_structural(
            model=struct_model,
            family=family,
            X=X_tr, T=T_tr, Y=Y_tr,
            epochs=config.epochs,
            lr=config.lr,
            batch_size=config.batch_size,
            weight_decay=config.weight_decay,
        )

        # Step B: Train nuisance model on same D_train (two-way split)
        nuisance_model = NuisanceNet(input_dim=X.shape[1])
        train_nuisance(
            model=nuisance_model,
            X=X_tr, T=T_tr,
            epochs=50,
            lr=config.lr,
            batch_size=config.batch_size,
            weight_decay=config.weight_decay,
        )

        # Compute Λ from D_train
        struct_model.eval()
        nuisance_model.eval()

        with torch.no_grad():
            X_tr_t = torch.FloatTensor(X_tr)
            T_tr_t = torch.FloatTensor(T_tr)
            theta_tr = struct_model(X_tr_t)
            t_mean_tr, _ = nuisance_model(X_tr_t)

            # Hessian from training fold
            Lambda = family.compute_hessian(theta_tr, T_tr_t, t_mean_tr)
            Lambda_inv = torch.linalg.pinv(Lambda)

            # Step C: Compute ψ on D_test (eval fold)
            X_ev_t = torch.FloatTensor(X_ev)
            T_ev_t = torch.FloatTensor(T_ev)
            Y_ev_t = torch.FloatTensor(Y_ev)

            theta_ev = struct_model(X_ev_t)
            t_mean_ev, t_var_ev = nuisance_model(X_ev_t)

            psi = family.influence_score(
                y=Y_ev_t,
                t=T_ev_t,
                theta=theta_ev,
                t_mean=t_mean_ev,
                t_var=t_var_ev,
                lambda_inv=Lambda_inv,
            )

            psi_all[eval_idx] = psi.numpy()

    mu_hat = float(np.mean(psi_all))
    se = float(np.std(psi_all, ddof=1) / np.sqrt(n))

    return mu_hat, se


# =============================================================================
# Bootstrap Estimator
# =============================================================================

def bootstrap(
    X: np.ndarray,
    T: np.ndarray,
    Y: np.ndarray,
    family,
    config,
    B: int = 100,
) -> Tuple[float, float]:
    """Bootstrap refit estimator.

    1. Train model on full data for μ̂
    2. For b = 1...B:
       - Resample (X, T, Y) with replacement
       - Train new model from scratch
       - Compute μ_b = mean(β_b)
    3. SE = std(μ_1, ..., μ_B)

    This captures sampling variability but NOT regularization bias.
    Expected: Still poor coverage, slightly better than naive.
    """
    n = len(X)
    rng = np.random.default_rng(config.seed)

    # Full model for point estimate
    model_full = StructuralNet(
        input_dim=X.shape[1],
        hidden_dims=config.hidden_dims,
        dropout=config.dropout,
    )
    train_structural(
        model=model_full,
        family=family,
        X=X, T=T, Y=Y,
        epochs=config.epochs,
        lr=config.lr,
        batch_size=config.batch_size,
        weight_decay=config.weight_decay,
    )

    model_full.eval()
    with torch.no_grad():
        theta_full = model_full(torch.FloatTensor(X))
        mu_hat = float(theta_full[:, 1].numpy().mean())

    # Bootstrap
    boot_epochs = max(config.epochs // 2, 30)
    bootstrap_mus = []

    for _ in range(B):
        idx = rng.choice(n, n, replace=True)
        X_b, T_b, Y_b = X[idx], T[idx], Y[idx]

        model_b = StructuralNet(
            input_dim=X.shape[1],
            hidden_dims=config.hidden_dims,
            dropout=config.dropout,
        )
        train_structural(
            model=model_b,
            family=family,
            X=X_b, T=T_b, Y=Y_b,
            epochs=boot_epochs,
            lr=config.lr,
            batch_size=config.batch_size,
            weight_decay=config.weight_decay,
        )

        model_b.eval()
        with torch.no_grad():
            theta_b = model_b(torch.FloatTensor(X_b))
            mu_b = float(theta_b[:, 1].numpy().mean())

        bootstrap_mus.append(mu_b)

    se = float(np.std(bootstrap_mus, ddof=1))

    return mu_hat, se


# =============================================================================
# Method Registry
# =============================================================================

METHODS = {
    "naive": naive,
    "influence": influence,
    "bootstrap": bootstrap,
}


def get_method(name: str):
    """Get inference method by name."""
    if name not in METHODS:
        raise ValueError(f"Unknown method: {name}. Available: {list(METHODS.keys())}")
    return METHODS[name]
