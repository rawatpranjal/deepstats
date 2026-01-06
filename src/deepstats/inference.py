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

from .models import StructuralNet, StructuralNetSeparate, NuisanceNet, train_structural, train_nuisance, TrainingHistory


def _create_structural_model(config, input_dim: int, n_params: int):
    """Create structural model based on config."""
    if getattr(config, 'separate_nets', False):
        return StructuralNetSeparate(
            input_dim=input_dim,
            hidden_dims=config.hidden_dims,
            dropout=config.dropout,
            n_params=n_params,
        )
    else:
        return StructuralNet(
            input_dim=input_dim,
            hidden_dims=config.hidden_dims,
            dropout=config.dropout,
            n_params=n_params,
        )


# =============================================================================
# Result Containers
# =============================================================================

@dataclass
class InferenceResult:
    """Result from any inference method.

    Phase 3 diagnostics (influence method only):
    - corrections: φᵢ values (correction term in influence score)
    - correction_ratio: |mean(φ)| / SE (should be 0.1-1.0)
    - hessian_diagnostics: Per-fold stability info (min_eig, condition)
    """
    mu_hat: float
    se: float
    values: np.ndarray  # beta values (naive/bootstrap) or psi scores (influence)
    alpha_hat: np.ndarray = None  # Predicted α(X) for parameter recovery
    beta_hat: np.ndarray = None   # Predicted β(X) for parameter recovery
    histories: list = None  # TrainingHistory objects (optional)
    # Phase 3: Correction diagnostics (influence method only)
    corrections: np.ndarray = None  # φᵢ values per observation
    correction_ratio: float = None  # |mean(φ)| / SE
    hessian_diagnostics: list = None  # Per-fold: {"min_eig": float, "condition": float}


# =============================================================================
# Naive Estimator
# =============================================================================

def naive(
    X: np.ndarray,
    T: np.ndarray,
    Y: np.ndarray,
    family,
    config,
    log_dir: str = None,
    sim_id: int = None,
) -> InferenceResult:
    """Naive plug-in estimator.

    1. Train model on full data
    2. μ̂ = mean(β(X))
    3. SE = std(β) / √n

    This underestimates uncertainty because it ignores regularization bias.
    Expected: ~30-50% coverage instead of 95%.
    """
    n = len(X)

    model = _create_structural_model(
        config=config,
        input_dim=X.shape[1],
        n_params=getattr(family, 'n_params', 2),
    )

    history = train_structural(
        model=model,
        family=family,
        X=X, T=T, Y=Y,
        epochs=config.epochs,
        lr=config.lr,
        batch_size=config.batch_size,
        weight_decay=config.weight_decay,
        early_stopping=getattr(config, 'early_stopping', True),
        patience=getattr(config, 'patience', 10),
        val_split=getattr(config, 'val_split', 0.1),
    )

    # Save training history
    if log_dir and sim_id is not None:
        history.save(f"{log_dir}/training/naive_sim{sim_id}.json")

    model.eval()
    with torch.no_grad():
        theta = model(torch.FloatTensor(X))
        t_tensor = torch.FloatTensor(T)
        target_values = family.h_value(theta, t_tensor).numpy()
        # Extract α̂(X) and β̂(X) for parameter recovery metrics
        alpha_hat = theta[:, 0].numpy()
        beta_hat = theta[:, 1].numpy()

    mu_hat = float(np.mean(target_values))
    se = float(np.std(target_values, ddof=1) / np.sqrt(n))

    return InferenceResult(
        mu_hat=mu_hat, se=se, values=target_values,
        alpha_hat=alpha_hat, beta_hat=beta_hat, histories=[history]
    )


# =============================================================================
# Influence Function Estimator (50-Fold Cross-Fitting)
# =============================================================================

def influence(
    X: np.ndarray,
    T: np.ndarray,
    Y: np.ndarray,
    family,
    config,
    log_dir: str = None,
    sim_id: int = None,
) -> InferenceResult:
    """Influence function estimator with K-fold structural cross-fitting.

    The FLM protocol (NOT DML):
    1. K folds (each model sees (K-1)/K of data)
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
    corrections_all = np.zeros(n)  # Phase 3: φᵢ values
    alpha_hat_all = np.zeros(n)  # For parameter recovery
    beta_hat_all = np.zeros(n)   # For parameter recovery

    n_folds = getattr(config, 'n_folds', 50)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=config.seed)
    histories = []
    hessian_diagnostics = []  # Phase 3: Per-fold stability info

    for fold_idx, (train_idx, eval_idx) in enumerate(kf.split(X)):
        X_tr, T_tr, Y_tr = X[train_idx], T[train_idx], Y[train_idx]
        X_ev, T_ev, Y_ev = X[eval_idx], T[eval_idx], Y[eval_idx]

        # Step A: Train structural model on D_train
        struct_model = _create_structural_model(
            config=config,
            input_dim=X.shape[1],
            n_params=getattr(family, 'n_params', 2),
        )
        history = train_structural(
            model=struct_model,
            family=family,
            X=X_tr, T=T_tr, Y=Y_tr,
            epochs=config.epochs,
            lr=config.lr,
            batch_size=config.batch_size,
            weight_decay=config.weight_decay,
            early_stopping=getattr(config, 'early_stopping', True),
            patience=getattr(config, 'patience', 10),
            val_split=getattr(config, 'val_split', 0.1),
        )
        histories.append(history)

        # Save training history for first few folds
        if log_dir and sim_id is not None and fold_idx < 3:
            history.save(f"{log_dir}/training/influence_sim{sim_id}_fold{fold_idx}.json")

        # Step B: Train nuisance model on same D_train (two-way split)
        # Use larger network and train longer for better E[T|X] estimation
        nuisance_model = NuisanceNet(input_dim=X.shape[1], hidden_dims=[64, 32])
        train_nuisance(
            model=nuisance_model,
            X=X_tr, T=T_tr,
            epochs=100,  # Longer training
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

            # Hessian from training fold (with Phase 3 diagnostics)
            Lambda, min_eig, condition = family.compute_hessian_with_diagnostics(
                theta_tr, T_tr_t, t_mean_tr
            )
            Lambda_inv = torch.linalg.pinv(Lambda)
            hessian_diagnostics.append({"min_eig": min_eig, "condition": condition})

            # Step C: Compute ψ on D_test (eval fold)
            X_ev_t = torch.FloatTensor(X_ev)
            T_ev_t = torch.FloatTensor(T_ev)
            Y_ev_t = torch.FloatTensor(Y_ev)

            theta_ev = struct_model(X_ev_t)
            t_mean_ev, t_var_ev = nuisance_model(X_ev_t)

            # Get both psi and correction term (Phase 3)
            psi, correction = family.influence_score(
                y=Y_ev_t,
                t=T_ev_t,
                theta=theta_ev,
                t_mean=t_mean_ev,
                t_var=t_var_ev,
                lambda_inv=Lambda_inv,
                return_correction=True,
            )

            psi_all[eval_idx] = psi.numpy()
            corrections_all[eval_idx] = correction.numpy()
            # Store α̂ and β̂ for parameter recovery metrics
            alpha_hat_all[eval_idx] = theta_ev[:, 0].numpy()
            beta_hat_all[eval_idx] = theta_ev[:, 1].numpy()

    # FLM paper: influence function SE (variance of ψ scores)
    mu_hat = float(np.mean(psi_all))
    se = float(np.std(psi_all, ddof=1) / np.sqrt(n))

    # Phase 3: Compute correction ratio (should be 0.1-1.0 for valid inference)
    correction_mean = float(np.abs(np.mean(corrections_all)))
    correction_ratio = correction_mean / se if se > 0 else float('nan')

    return InferenceResult(
        mu_hat=mu_hat, se=se, values=psi_all,
        alpha_hat=alpha_hat_all, beta_hat=beta_hat_all, histories=histories,
        corrections=corrections_all,
        correction_ratio=correction_ratio,
        hessian_diagnostics=hessian_diagnostics,
    )


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
    log_dir: str = None,
    sim_id: int = None,
) -> InferenceResult:
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
    histories = []

    # Full model for point estimate
    model_full = _create_structural_model(
        config=config,
        input_dim=X.shape[1],
        n_params=getattr(family, 'n_params', 2),
    )
    history = train_structural(
        model=model_full,
        family=family,
        X=X, T=T, Y=Y,
        epochs=config.epochs,
        lr=config.lr,
        batch_size=config.batch_size,
        weight_decay=config.weight_decay,
        early_stopping=getattr(config, 'early_stopping', True),
        patience=getattr(config, 'patience', 10),
        val_split=getattr(config, 'val_split', 0.1),
    )
    histories.append(history)

    # Save training history
    if log_dir and sim_id is not None:
        history.save(f"{log_dir}/training/bootstrap_sim{sim_id}.json")

    model_full.eval()
    with torch.no_grad():
        theta_full = model_full(torch.FloatTensor(X))
        t_tensor = torch.FloatTensor(T)
        mu_hat = float(family.h_value(theta_full, t_tensor).numpy().mean())

    # Bootstrap
    boot_epochs = max(config.epochs // 2, 30)
    bootstrap_mus = []

    for _ in range(B):
        idx = rng.choice(n, n, replace=True)
        X_b, T_b, Y_b = X[idx], T[idx], Y[idx]

        model_b = _create_structural_model(
            config=config,
            input_dim=X.shape[1],
            n_params=getattr(family, 'n_params', 2),
        )
        history_b = train_structural(
            model=model_b,
            family=family,
            X=X_b, T=T_b, Y=Y_b,
            epochs=boot_epochs,
            lr=config.lr,
            batch_size=config.batch_size,
            weight_decay=config.weight_decay,
        )
        histories.append(history_b)

        model_b.eval()
        with torch.no_grad():
            theta_b = model_b(torch.FloatTensor(X_b))
            t_b_tensor = torch.FloatTensor(T_b)
            mu_b = float(family.h_value(theta_b, t_b_tensor).numpy().mean())

        bootstrap_mus.append(mu_b)

    se = float(np.std(bootstrap_mus, ddof=1))

    return InferenceResult(mu_hat=mu_hat, se=se, values=np.array(bootstrap_mus), histories=histories)


# =============================================================================
# BCa Bootstrap Estimator
# =============================================================================

def bootstrap_bca(
    X: np.ndarray,
    T: np.ndarray,
    Y: np.ndarray,
    family,
    config,
    B: int = 100,
    log_dir: str = None,
    sim_id: int = None,
) -> Tuple[float, float]:
    """BCa (Bias-Corrected and accelerated) Bootstrap.

    Improves on standard bootstrap by:
    1. Bias correction: adjusts for median bias in bootstrap distribution
    2. Acceleration: adjusts for skewness using jackknife

    Returns mu_hat and SE calibrated to give proper 95% CI coverage.
    """
    n = len(X)
    rng = np.random.default_rng(config.seed)

    # Full model for point estimate
    model_full = _create_structural_model(
        config=config,
        input_dim=X.shape[1],
        n_params=getattr(family, 'n_params', 2),
    )
    train_structural(
        model=model_full,
        family=family,
        X=X, T=T, Y=Y,
        epochs=config.epochs,
        lr=config.lr,
        batch_size=config.batch_size,
        weight_decay=config.weight_decay,
        early_stopping=getattr(config, 'early_stopping', True),
        patience=getattr(config, 'patience', 10),
        val_split=getattr(config, 'val_split', 0.1),
    )

    model_full.eval()
    with torch.no_grad():
        theta_full = model_full(torch.FloatTensor(X))
        t_tensor = torch.FloatTensor(T)
        mu_hat = float(family.h_value(theta_full, t_tensor).numpy().mean())

    # Bootstrap replicates
    boot_epochs = max(config.epochs // 2, 30)
    bootstrap_mus = []

    for _ in range(B):
        idx = rng.choice(n, n, replace=True)
        X_b, T_b, Y_b = X[idx], T[idx], Y[idx]

        model_b = _create_structural_model(
            config=config,
            input_dim=X.shape[1],
            n_params=getattr(family, 'n_params', 2),
        )
        train_structural(
            model=model_b,
            family=family,
            X=X_b, T=T_b, Y=Y_b,
            epochs=boot_epochs,
            lr=config.lr,
            batch_size=config.batch_size,
            weight_decay=config.weight_decay,
            early_stopping=getattr(config, 'early_stopping', True),
            patience=getattr(config, 'patience', 10),
            val_split=0,  # No val split for bootstrap
        )

        model_b.eval()
        with torch.no_grad():
            theta_b = model_b(torch.FloatTensor(X_b))
            t_b_tensor = torch.FloatTensor(T_b)
            mu_b = float(family.h_value(theta_b, t_b_tensor).numpy().mean())

        bootstrap_mus.append(mu_b)

    bootstrap_mus = np.array(bootstrap_mus)

    # BCa correction
    from scipy.stats import norm as scipy_norm

    # Bias correction factor (z0)
    prop_less = np.mean(bootstrap_mus < mu_hat)
    z0 = scipy_norm.ppf(max(min(prop_less, 0.999), 0.001))

    # Acceleration factor (a) via jackknife
    # Simplified: use bootstrap variance estimate
    jack_mus = []
    n_jack = min(20, n // 10)  # Subsample for speed
    jack_idx = rng.choice(n, n_jack, replace=False)

    for i in jack_idx:
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        X_j, T_j, Y_j = X[mask], T[mask], Y[mask]

        model_j = _create_structural_model(
            config=config,
            input_dim=X.shape[1],
            n_params=getattr(family, 'n_params', 2),
        )
        train_structural(
            model=model_j,
            family=family,
            X=X_j, T=T_j, Y=Y_j,
            epochs=boot_epochs // 2,
            lr=config.lr,
            batch_size=config.batch_size,
            weight_decay=config.weight_decay,
            early_stopping=False,
            val_split=0,
        )

        model_j.eval()
        with torch.no_grad():
            theta_j = model_j(torch.FloatTensor(X_j))
            t_j_tensor = torch.FloatTensor(T_j)
            mu_j = float(family.h_value(theta_j, t_j_tensor).numpy().mean())
        jack_mus.append(mu_j)

    jack_mus = np.array(jack_mus)
    jack_mean = np.mean(jack_mus)

    # Acceleration
    num = np.sum((jack_mean - jack_mus) ** 3)
    denom = 6 * (np.sum((jack_mean - jack_mus) ** 2) ** 1.5)
    a = num / (denom + 1e-10)

    # BCa confidence interval percentiles
    alpha = 0.05
    z_alpha = scipy_norm.ppf(alpha / 2)
    z_1alpha = scipy_norm.ppf(1 - alpha / 2)

    # Adjusted percentiles
    p_lo = scipy_norm.cdf(z0 + (z0 + z_alpha) / (1 - a * (z0 + z_alpha)))
    p_hi = scipy_norm.cdf(z0 + (z0 + z_1alpha) / (1 - a * (z0 + z_1alpha)))

    # BCa interval
    ci_lo = np.percentile(bootstrap_mus, 100 * p_lo)
    ci_hi = np.percentile(bootstrap_mus, 100 * p_hi)

    # Convert CI width to SE (assuming symmetric for comparison)
    se = (ci_hi - ci_lo) / (2 * 1.96)

    return mu_hat, se


# =============================================================================
# Method Registry
# =============================================================================

METHODS = {
    "naive": naive,
    "influence": influence,
    "bootstrap": bootstrap,
    "bootstrap_bca": bootstrap_bca,
}


def get_method(name: str):
    """Get inference method by name."""
    if name not in METHODS:
        raise ValueError(f"Unknown method: {name}. Available: {list(METHODS.keys())}")
    return METHODS[name]
