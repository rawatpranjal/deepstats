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
from sklearn.neural_network import MLPRegressor

try:
    from lightgbm import LGBMRegressor
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

from .models import StructuralNet, StructuralNetSeparate, NuisanceNet, train_structural, train_nuisance, TrainingHistory


# =============================================================================
# Lambda Estimator (Nonparametric Hessian Regression)
# =============================================================================

class LambdaEstimator:
    """Nonparametric regression for Λ(x) = E[ℓ_θθ | X=x].

    Used in three-way splitting for families where Λ depends on θ.
    Fits separate regression models for each unique element of the Hessian matrix.
    """

    def __init__(self, method: str = 'mlp', n_params: int = 2):
        """Initialize LambdaEstimator.

        Args:
            method: 'mlp' (sklearn MLPRegressor) or 'lgbm' (LightGBM)
            n_params: Number of structural parameters (determines Hessian size)
        """
        self.method = method
        self.n_params = n_params
        self.models = []
        self.n_unique = None  # Number of unique elements in symmetric matrix

        if method == 'lgbm' and not HAS_LIGHTGBM:
            raise ImportError("LightGBM not installed. Use method='mlp' or install lightgbm.")

    def fit(self, X: np.ndarray, hessians: np.ndarray):
        """Fit regression models from X to Hessian elements.

        Args:
            X: (n, d_x) covariate matrix
            hessians: (n, n_params, n_params) per-observation Hessians
        """
        n = X.shape[0]
        self.models = []

        # Extract unique elements (upper triangle of symmetric matrix)
        targets = []
        for i in range(self.n_params):
            for j in range(i, self.n_params):
                targets.append(hessians[:, i, j])

        self.n_unique = len(targets)

        for target in targets:
            if self.method == 'mlp':
                model = MLPRegressor(
                    hidden_layer_sizes=(32, 16),
                    max_iter=200,
                    early_stopping=True,
                    validation_fraction=0.1,
                    random_state=42,
                )
            else:  # lgbm
                model = LGBMRegressor(
                    n_estimators=100,
                    num_leaves=31,
                    verbose=-1,
                    random_state=42,
                )
            model.fit(X, target)
            self.models.append(model)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict Hessians at new covariate values.

        Args:
            X: (n, d_x) covariate matrix

        Returns:
            hessians: (n, n_params, n_params) predicted Hessians
        """
        n = X.shape[0]
        predictions = [m.predict(X) for m in self.models]

        # Reconstruct symmetric matrix
        hessians = np.zeros((n, self.n_params, self.n_params))
        idx = 0
        for i in range(self.n_params):
            for j in range(i, self.n_params):
                hessians[:, i, j] = predictions[idx]
                hessians[:, j, i] = predictions[idx]  # Symmetry
                idx += 1

        return hessians


def batch_pinv(matrices: np.ndarray, ridge: float = 1e-4) -> np.ndarray:
    """Batch pseudo-inverse for array of matrices with ridge regularization.

    Args:
        matrices: (n, d, d) array of square matrices
        ridge: Ridge regularization for stability

    Returns:
        inverses: (n, d, d) array of pseudo-inverses
    """
    n, d, _ = matrices.shape
    inverses = np.zeros_like(matrices)

    # Add ridge regularization
    eye = np.eye(d)
    for i in range(n):
        mat = matrices[i] + ridge * eye
        try:
            inverses[i] = np.linalg.pinv(mat)
        except np.linalg.LinAlgError:
            # Fallback: use larger ridge
            inverses[i] = np.linalg.pinv(mat + 0.01 * eye)

    return inverses


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
        # For multi-param models, use first param as alpha, target functional as beta
        n_params = theta.shape[1]
        alpha_hat = theta[:, 0].numpy()
        if n_params == 2:
            beta_hat = theta[:, 1].numpy()
        else:
            beta_hat = target_values  # Use target functional for multi-param models

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
       - If Λ depends on θ: THREE-WAY split (60% θ, 40% Λ)
       - Otherwise: TWO-WAY split (all train data for θ and Λ)
       - Compute ψᵢ on D_test (fold k)
    3. μ̂ = mean(ψ), SE computed via within-fold variance

    The influence score corrects for regularization bias:
    ψᵢ = Ĥ(θ̂ᵢ) - Ĥ_θᵀ · Λ̂(xᵢ)⁻¹ · ℓ̂_θ,ᵢ

    Expected: ~95% coverage.
    """
    n = len(X)
    psi_all = np.zeros(n)
    corrections_all = np.zeros(n)  # Phase 3: φᵢ values
    alpha_hat_all = np.zeros(n)  # For parameter recovery
    beta_hat_all = np.zeros(n)   # For parameter recovery
    fold_indices = np.zeros(n, dtype=int)  # Track which fold each obs belongs to

    n_folds = getattr(config, 'n_folds', 50)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=config.seed)
    histories = []
    hessian_diagnostics = []  # Phase 3: Per-fold stability info

    # Config options
    use_three_way = family.lambda_depends_on_theta()
    theta_lambda_split = getattr(config, 'theta_lambda_split', 0.6)
    lambda_method = getattr(config, 'lambda_method', 'mlp')
    center_treatment = getattr(config, 'center_treatment', False)
    n_params = getattr(family, 'n_params', 2)

    for fold_idx, (train_idx, eval_idx) in enumerate(kf.split(X)):
        fold_indices[eval_idx] = fold_idx

        # Check for multi-dimensional T
        is_multidim_t = T.ndim > 1 and T.shape[1] > 1

        if use_three_way and not is_multidim_t:
            # ============================================
            # THREE-WAY SPLITTING (for nonlinear families)
            # ============================================
            n_train = len(train_idx)
            split_point = int(theta_lambda_split * n_train)

            # Shuffle train_idx for random split
            rng = np.random.RandomState(config.seed + fold_idx)
            shuffled_train_idx = train_idx.copy()
            rng.shuffle(shuffled_train_idx)

            theta_train_idx = shuffled_train_idx[:split_point]
            lambda_train_idx = shuffled_train_idx[split_point:]

            X_theta, T_theta, Y_theta = X[theta_train_idx], T[theta_train_idx], Y[theta_train_idx]
            X_lambda, T_lambda, Y_lambda = X[lambda_train_idx], T[lambda_train_idx], Y[lambda_train_idx]
            X_ev, T_ev, Y_ev = X[eval_idx], T[eval_idx], Y[eval_idx]

            # Step A: Train structural model on θ-training fold
            struct_model = _create_structural_model(
                config=config,
                input_dim=X.shape[1],
                n_params=n_params,
            )
            history = train_structural(
                model=struct_model,
                family=family,
                X=X_theta, T=T_theta, Y=Y_theta,
                epochs=config.epochs,
                lr=config.lr,
                batch_size=config.batch_size,
                weight_decay=config.weight_decay,
                early_stopping=getattr(config, 'early_stopping', True),
                patience=getattr(config, 'patience', 10),
                val_split=getattr(config, 'val_split', 0.1),
            )
            histories.append(history)

            # Step B: Compute per-observation Hessians on Λ-training fold
            struct_model.eval()
            with torch.no_grad():
                X_lambda_t = torch.FloatTensor(X_lambda)
                T_lambda_t = torch.FloatTensor(T_lambda)
                Y_lambda_t = torch.FloatTensor(Y_lambda)
                theta_lambda_pred = struct_model(X_lambda_t)

                # Per-observation Hessians: (n_lambda, n_params, n_params)
                hessians_lambda = family.hessian_at_point(Y_lambda_t, T_lambda_t, theta_lambda_pred)
                hessians_lambda_np = hessians_lambda.numpy()

            # Step C: Fit nonparametric Λ(x) regression
            lambda_estimator = LambdaEstimator(method=lambda_method, n_params=n_params)
            lambda_estimator.fit(X_lambda, hessians_lambda_np)

            # Step D: Predict Λ̂(x_i) on eval fold and invert per observation
            Lambda_eval_np = lambda_estimator.predict(X_ev)  # (n_eval, n_params, n_params)
            Lambda_inv_eval_np = batch_pinv(Lambda_eval_np)  # (n_eval, n_params, n_params)
            Lambda_inv_eval = torch.FloatTensor(Lambda_inv_eval_np)

            # Record diagnostics (use mean Hessian for summary)
            Lambda_mean = np.mean(Lambda_eval_np, axis=0)
            eigenvalues = np.linalg.eigvalsh(Lambda_mean)
            min_eig = float(eigenvalues.min())
            max_eig = float(eigenvalues.max())
            condition = max_eig / (min_eig + 1e-10)
            hessian_diagnostics.append({"min_eig": min_eig, "condition": condition})

            # Step E: Compute ψ on eval fold with per-observation Λ⁻¹
            with torch.no_grad():
                X_ev_t = torch.FloatTensor(X_ev)
                T_ev_t = torch.FloatTensor(T_ev)
                Y_ev_t = torch.FloatTensor(Y_ev)
                theta_ev = struct_model(X_ev_t)

                # t_mean for influence score (use simple mean if not centering)
                if center_treatment:
                    # Train nuisance model on full train data for centering
                    nuisance_model = NuisanceNet(input_dim=X.shape[1], hidden_dims=[64, 32])
                    train_nuisance(
                        model=nuisance_model,
                        X=X[train_idx], T=T[train_idx],
                        epochs=100, lr=config.lr,
                        batch_size=config.batch_size, weight_decay=config.weight_decay,
                    )
                    nuisance_model.eval()
                    t_mean_ev, t_var_ev = nuisance_model(X_ev_t)
                else:
                    # No centering - use zeros (T_design = [1, T])
                    t_mean_ev = torch.zeros(len(X_ev))
                    t_var_ev = torch.ones(len(X_ev))

                # Get psi with per-observation Λ⁻¹
                psi, correction = family.influence_score(
                    y=Y_ev_t,
                    t=T_ev_t,
                    theta=theta_ev,
                    t_mean=t_mean_ev,
                    t_var=t_var_ev,
                    lambda_inv=Lambda_inv_eval,  # (n_eval, n_params, n_params)
                    return_correction=True,
                    center_treatment=center_treatment,
                )

        else:
            # ============================================
            # TWO-WAY SPLITTING (for linear family or multidim T)
            # ============================================
            X_tr, T_tr, Y_tr = X[train_idx], T[train_idx], Y[train_idx]
            X_ev, T_ev, Y_ev = X[eval_idx], T[eval_idx], Y[eval_idx]

            # Step A: Train structural model on D_train
            struct_model = _create_structural_model(
                config=config,
                input_dim=X.shape[1],
                n_params=n_params,
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

            # Step B: Train nuisance model (if centering or for diagnostics)
            if not is_multidim_t:
                nuisance_model = NuisanceNet(input_dim=X.shape[1], hidden_dims=[64, 32])
                train_nuisance(
                    model=nuisance_model,
                    X=X_tr, T=T_tr,
                    epochs=100,
                    lr=config.lr,
                    batch_size=config.batch_size,
                    weight_decay=config.weight_decay,
                )

            # Compute aggregate Λ from D_train
            struct_model.eval()
            if not is_multidim_t:
                nuisance_model.eval()

            with torch.no_grad():
                X_tr_t = torch.FloatTensor(X_tr)
                T_tr_t = torch.FloatTensor(T_tr)
                theta_tr = struct_model(X_tr_t)

                if is_multidim_t:
                    t_mean_tr = T_tr_t.mean(dim=0)
                elif center_treatment:
                    t_mean_tr, _ = nuisance_model(X_tr_t)
                else:
                    t_mean_tr = torch.zeros(len(X_tr))

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

                if is_multidim_t:
                    t_mean_ev = T_ev_t.mean(dim=0).expand(len(X_ev), -1)
                    t_var_ev = T_ev_t.var(dim=0).expand(len(X_ev), -1)
                elif center_treatment:
                    t_mean_ev, t_var_ev = nuisance_model(X_ev_t)
                else:
                    t_mean_ev = torch.zeros(len(X_ev))
                    t_var_ev = torch.ones(len(X_ev))

                # Get psi with aggregate Λ⁻¹
                psi, correction = family.influence_score(
                    y=Y_ev_t,
                    t=T_ev_t,
                    theta=theta_ev,
                    t_mean=t_mean_ev,
                    t_var=t_var_ev,
                    lambda_inv=Lambda_inv,  # (n_params, n_params)
                    return_correction=True,
                    center_treatment=center_treatment,
                )

        # Store results for this fold
        psi_all[eval_idx] = psi.numpy()
        corrections_all[eval_idx] = correction.numpy()

        # Store α̂ and β̂ for parameter recovery metrics
        n_params_actual = theta_ev.shape[1]
        alpha_hat_all[eval_idx] = theta_ev[:, 0].numpy()
        if n_params_actual == 2:
            beta_hat_all[eval_idx] = theta_ev[:, 1].numpy()
        else:
            beta_hat_all[eval_idx] = family.h_value(theta_ev, T_ev_t).numpy()

        # Save training history for first few folds
        if log_dir and sim_id is not None and fold_idx < 3:
            history.save(f"{log_dir}/training/influence_sim{sim_id}_fold{fold_idx}.json")

    # Within-fold variance formula (paper formula)
    mu_hat = float(np.mean(psi_all))
    se = _compute_within_fold_se(psi_all, fold_indices, n_folds)

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


def _compute_within_fold_se(psi_all: np.ndarray, fold_indices: np.ndarray, K: int) -> float:
    """Compute SE using within-fold variance formula from the paper.

    Paper formula: Ψ̂ = (1/K) Σ_k (1/|I_k|) Σ_{i∈I_k} (ψ_ik - μ̂_k)²
    SE = √(Ψ̂/n)
    """
    n = len(psi_all)
    variance_sum = 0.0

    for k in range(K):
        psi_k = psi_all[fold_indices == k]
        if len(psi_k) > 0:
            mu_k = psi_k.mean()
            variance_k = ((psi_k - mu_k) ** 2).mean()
            variance_sum += variance_k

    Psi_hat = variance_sum / K
    return float(np.sqrt(Psi_hat / n))


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
# Dual-Target Tobit Estimators (Single Training → Both Estimands)
# =============================================================================

def naive_tobit_both(
    X: np.ndarray,
    T: np.ndarray,
    Y: np.ndarray,
    config,
    log_dir: str = None,
    sim_id: int = None,
) -> dict:
    """Naive estimator for BOTH Tobit targets from single training.

    Trains one model, computes both:
    - Latent: E[β]
    - Observed: E[β·Φ(z)]

    Returns dict with 'latent' and 'observed' InferenceResult objects.
    """
    from .families import TobitFamily

    # Use latent family for training (loss is identical for both targets)
    family_train = TobitFamily(target="latent")
    family_latent = TobitFamily(target="latent")
    family_observed = TobitFamily(target="observed")

    n = len(X)

    model = _create_structural_model(
        config=config,
        input_dim=X.shape[1],
        n_params=3,  # Tobit has 3 params
    )

    history = train_structural(
        model=model,
        family=family_train,
        X=X, T=T, Y=Y,
        epochs=config.epochs,
        lr=config.lr,
        batch_size=config.batch_size,
        weight_decay=config.weight_decay,
        early_stopping=getattr(config, 'early_stopping', True),
        patience=getattr(config, 'patience', 10),
        val_split=getattr(config, 'val_split', 0.1),
    )

    if log_dir and sim_id is not None:
        history.save(f"{log_dir}/training/naive_sim{sim_id}.json")

    model.eval()
    with torch.no_grad():
        theta = model(torch.FloatTensor(X))
        t_tensor = torch.FloatTensor(T)

        # Compute both targets from same theta
        h_latent = family_latent.h_value(theta, t_tensor).numpy()
        h_observed = family_observed.h_value(theta, t_tensor).numpy()

        alpha_hat = theta[:, 0].numpy()
        beta_hat = theta[:, 1].numpy()

    # Latent target
    mu_latent = float(np.mean(h_latent))
    se_latent = float(np.std(h_latent, ddof=1) / np.sqrt(n))

    # Observed target
    mu_observed = float(np.mean(h_observed))
    se_observed = float(np.std(h_observed, ddof=1) / np.sqrt(n))

    return {
        "latent": InferenceResult(
            mu_hat=mu_latent, se=se_latent, values=h_latent,
            alpha_hat=alpha_hat, beta_hat=beta_hat, histories=[history]
        ),
        "observed": InferenceResult(
            mu_hat=mu_observed, se=se_observed, values=h_observed,
            alpha_hat=alpha_hat, beta_hat=beta_hat, histories=[history]
        ),
    }


def influence_tobit_both(
    X: np.ndarray,
    T: np.ndarray,
    Y: np.ndarray,
    config,
    log_dir: str = None,
    sim_id: int = None,
) -> dict:
    """Influence function estimator for BOTH Tobit targets from single training.

    Trains K models once, computes influence scores for both:
    - Latent: E[β]
    - Observed: E[β·Φ(z)]

    Returns dict with 'latent' and 'observed' InferenceResult objects.
    """
    from .families import TobitFamily

    family_train = TobitFamily(target="latent")  # Loss is identical
    family_latent = TobitFamily(target="latent")
    family_observed = TobitFamily(target="observed")

    n = len(X)

    # Arrays for both targets
    psi_latent = np.zeros(n)
    psi_observed = np.zeros(n)
    corrections_latent = np.zeros(n)
    corrections_observed = np.zeros(n)
    alpha_hat_all = np.zeros(n)
    beta_hat_all = np.zeros(n)

    n_folds = getattr(config, 'n_folds', 50)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=config.seed)
    histories = []
    hessian_diagnostics = []

    for fold_idx, (train_idx, eval_idx) in enumerate(kf.split(X)):
        X_tr, T_tr, Y_tr = X[train_idx], T[train_idx], Y[train_idx]
        X_ev, T_ev, Y_ev = X[eval_idx], T[eval_idx], Y[eval_idx]

        # Step A: Train structural model on D_train (ONCE)
        struct_model = _create_structural_model(
            config=config,
            input_dim=X.shape[1],
            n_params=3,  # Tobit has 3 params
        )
        history = train_structural(
            model=struct_model,
            family=family_train,
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

        if log_dir and sim_id is not None and fold_idx < 3:
            history.save(f"{log_dir}/training/influence_sim{sim_id}_fold{fold_idx}.json")

        # Step B: Train nuisance model
        nuisance_model = NuisanceNet(input_dim=X.shape[1], hidden_dims=[64, 32])
        train_nuisance(
            model=nuisance_model,
            X=X_tr, T=T_tr,
            epochs=100,
            lr=config.lr,
            batch_size=config.batch_size,
            weight_decay=config.weight_decay,
        )

        struct_model.eval()
        nuisance_model.eval()

        with torch.no_grad():
            X_tr_t = torch.FloatTensor(X_tr)
            T_tr_t = torch.FloatTensor(T_tr)
            theta_tr = struct_model(X_tr_t)
            t_mean_tr, _ = nuisance_model(X_tr_t)

            # Compute Hessian (same for both targets - depends on loss, not H)
            Lambda, min_eig, condition = family_train.compute_hessian_with_diagnostics(
                theta_tr, T_tr_t, t_mean_tr
            )
            Lambda_inv = torch.linalg.pinv(Lambda)
            hessian_diagnostics.append({"min_eig": min_eig, "condition": condition})

            # Step C: Compute predictions on D_test
            X_ev_t = torch.FloatTensor(X_ev)
            T_ev_t = torch.FloatTensor(T_ev)
            Y_ev_t = torch.FloatTensor(Y_ev)

            theta_ev = struct_model(X_ev_t)
            t_mean_ev, t_var_ev = nuisance_model(X_ev_t)

            # Compute influence scores for BOTH targets (using same theta_ev)
            psi_lat, corr_lat = family_latent.influence_score(
                y=Y_ev_t, t=T_ev_t, theta=theta_ev,
                t_mean=t_mean_ev, t_var=t_var_ev, lambda_inv=Lambda_inv,
                return_correction=True,
            )
            psi_obs, corr_obs = family_observed.influence_score(
                y=Y_ev_t, t=T_ev_t, theta=theta_ev,
                t_mean=t_mean_ev, t_var=t_var_ev, lambda_inv=Lambda_inv,
                return_correction=True,
            )

            psi_latent[eval_idx] = psi_lat.numpy()
            psi_observed[eval_idx] = psi_obs.numpy()
            corrections_latent[eval_idx] = corr_lat.numpy()
            corrections_observed[eval_idx] = corr_obs.numpy()
            alpha_hat_all[eval_idx] = theta_ev[:, 0].numpy()
            beta_hat_all[eval_idx] = theta_ev[:, 1].numpy()

    # Compute results for latent target
    mu_latent = float(np.mean(psi_latent))
    se_latent = float(np.std(psi_latent, ddof=1) / np.sqrt(n))
    corr_ratio_latent = float(np.abs(np.mean(corrections_latent))) / se_latent if se_latent > 0 else float('nan')

    # Compute results for observed target
    mu_observed = float(np.mean(psi_observed))
    se_observed = float(np.std(psi_observed, ddof=1) / np.sqrt(n))
    corr_ratio_observed = float(np.abs(np.mean(corrections_observed))) / se_observed if se_observed > 0 else float('nan')

    return {
        "latent": InferenceResult(
            mu_hat=mu_latent, se=se_latent, values=psi_latent,
            alpha_hat=alpha_hat_all, beta_hat=beta_hat_all, histories=histories,
            corrections=corrections_latent, correction_ratio=corr_ratio_latent,
            hessian_diagnostics=hessian_diagnostics,
        ),
        "observed": InferenceResult(
            mu_hat=mu_observed, se=se_observed, values=psi_observed,
            alpha_hat=alpha_hat_all, beta_hat=beta_hat_all, histories=histories,
            corrections=corrections_observed, correction_ratio=corr_ratio_observed,
            hessian_diagnostics=hessian_diagnostics,
        ),
    }


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
