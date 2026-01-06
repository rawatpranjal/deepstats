#!/usr/bin/env python
"""Monte Carlo validation for influence function inference.

Compares naive, influence, and bootstrap methods across 8 models.

Usage:
    python run_mc.py --M 50 --N 1000 --models linear poisson --methods naive influence
    python run_mc.py --M 10 --N 500 --models linear --methods naive influence bootstrap
"""

import argparse
import time
from dataclasses import dataclass, field, asdict
from typing import List

import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed

from .dgp import get_dgp, DGPS
from .families import get_family, FAMILIES
from .inference import naive, influence, bootstrap, METHODS
from .metrics import compute_metrics, print_table
from .logging import create_full_report, save_report, format_human_readable


# =============================================================================
# Config
# =============================================================================

# Network depth presets
NETWORK_PRESETS = {
    "default": [64, 32],
    "deep": [128, 64, 32],
    "deeper": [256, 128, 64, 32],
}


@dataclass
class Config:
    """Simulation configuration."""
    M: int = 50                   # Monte Carlo simulations
    N: int = 1000                 # Sample size
    n_folds: int = 50             # Cross-fitting folds (98% train)
    epochs: int = 100             # Training epochs
    lr: float = 0.01              # Learning rate
    batch_size: int = 64          # Batch size
    hidden_dims: List[int] = field(default_factory=lambda: [64, 32])
    dropout: float = 0.1          # Dropout
    weight_decay: float = 1e-4    # L2 regularization
    seed: int = 42
    # Early stopping
    early_stopping: bool = True
    patience: int = 10
    val_split: float = 0.1
    # Architecture
    separate_nets: bool = False   # Use separate networks for α and β
    # Target
    target: str = "beta"          # Target: 'beta' or 'ame' (for logit)
    # Logging
    log_dir: str = None


# =============================================================================
# Run One Simulation
# =============================================================================

def run_one_sim(
    sim_id: int,
    model_name: str,
    method_names: List[str],
    config: Config,
) -> List[dict]:
    """Run one MC simulation for a model."""
    dgp = get_dgp(model_name, seed=config.seed + sim_id * 1000)
    # Pass target to logit/tobit families for AME/observed support
    if model_name == "logit":
        family_kwargs = {"target": config.target}
    elif model_name == "tobit":
        # Map beta -> latent, ame -> observed for tobit
        tobit_target = "observed" if config.target == "ame" else "latent"
        family_kwargs = {"target": tobit_target}
    else:
        family_kwargs = {}
    family = get_family(model_name, **family_kwargs)
    data = dgp.generate(config.N)
    # For tobit with observed target, recompute mu_true
    if model_name == "tobit" and config.target == "ame":
        mu_true = dgp.compute_true_mu(target="observed")
    else:
        mu_true = data.mu_true

    from .inference import bootstrap_bca
    method_funcs = {"naive": naive, "influence": influence, "bootstrap": bootstrap, "bootstrap_bca": bootstrap_bca}
    results = []

    # Determine log_dir for this model
    log_dir = f"{config.log_dir}/{model_name}" if config.log_dir else None

    for method_name in method_names:
        try:
            result = method_funcs[method_name](
                data.X, data.T, data.Y, family, config,
                log_dir=log_dir,
                sim_id=sim_id,
            )
            mu_hat = result.mu_hat
            se = result.se
            bias = mu_hat - mu_true
            ci_lo = mu_hat - 1.96 * se
            ci_hi = mu_hat + 1.96 * se
            covered = ci_lo <= mu_true <= ci_hi

            # Compute overfitting metrics from histories
            overfit_rate = 0.0
            mean_best_epoch = 0.0
            if result.histories:
                statuses = [h.status for h in result.histories]
                overfit_rate = sum(s == "overfit" for s in statuses) / len(statuses)
                mean_best_epoch = sum(h.best_epoch for h in result.histories) / len(result.histories)

            # Compute parameter recovery metrics (Phase 1)
            rmse_alpha = np.nan
            rmse_beta = np.nan
            corr_alpha = np.nan
            corr_beta = np.nan
            if result.alpha_hat is not None and result.beta_hat is not None:
                rmse_alpha = float(np.sqrt(np.mean((result.alpha_hat - data.alpha_true)**2)))
                rmse_beta = float(np.sqrt(np.mean((result.beta_hat - data.beta_true)**2)))
                corr_alpha = float(np.corrcoef(result.alpha_hat, data.alpha_true)[0, 1])
                corr_beta = float(np.corrcoef(result.beta_hat, data.beta_true)[0, 1])

            # Phase 3: Training diagnostics
            final_grad_norm = np.nan
            final_beta_std = np.nan
            final_train_loss = np.nan
            final_val_loss = np.nan
            train_val_gap = np.nan
            if result.histories and result.histories[0].grad_norm:
                # Average final gradient norm across folds
                grad_norms = [h.grad_norm[-1] if h.grad_norm else np.nan for h in result.histories]
                final_grad_norm = float(np.nanmean(grad_norms))
            if result.histories and result.histories[0].beta_std:
                # Average final beta std across folds
                beta_stds = [h.beta_std[-1] if h.beta_std else np.nan for h in result.histories]
                final_beta_std = float(np.nanmean(beta_stds))
            if result.histories:
                # Average final train/val loss across folds (at best_epoch)
                train_losses = []
                val_losses = []
                for h in result.histories:
                    if h.train_loss and h.val_loss:
                        best_idx = min(h.best_epoch, len(h.train_loss) - 1)
                        train_losses.append(h.train_loss[best_idx])
                        val_losses.append(h.val_loss[best_idx])
                if train_losses and val_losses:
                    final_train_loss = float(np.nanmean(train_losses))
                    final_val_loss = float(np.nanmean(val_losses))
                    train_val_gap = final_val_loss - final_train_loss

            # Phase 3: Correction diagnostics (influence method only)
            correction_ratio = np.nan
            correction_mean = np.nan
            correction_std = np.nan
            min_hessian_eig = np.nan
            hessian_condition = np.nan
            if result.corrections is not None:
                correction_mean = float(np.mean(result.corrections))
                correction_std = float(np.std(result.corrections))
            if result.correction_ratio is not None:
                correction_ratio = result.correction_ratio
            if result.hessian_diagnostics:
                min_hessian_eig = float(np.mean([h["min_eig"] for h in result.hessian_diagnostics]))
                hessian_condition = float(np.mean([h["condition"] for h in result.hessian_diagnostics]))

            # Oracle loss and smoothness ratio (NEW diagnostic metrics)
            oracle_loss = np.nan
            excess_loss_ratio = np.nan
            smoothness_ratio = np.nan

            # Oracle loss: loss with TRUE parameters (theoretical minimum)
            # Y = alpha* + beta* * T + epsilon, so oracle loss = var(epsilon) = 1.0 for linear DGP
            oracle_loss = 1.0  # For linear Gaussian DGP with sigma=1

            # Excess loss ratio: (val_loss - oracle_loss) / oracle_loss
            if not np.isnan(final_val_loss) and oracle_loss > 0:
                excess_loss_ratio = (final_val_loss - oracle_loss) / oracle_loss

            # Smoothness ratio: std(grad(beta_hat)) / std(grad(beta_true))
            # Measures regularization shrinkage - ratio < 1 means predicted beta is flatter
            if result.beta_hat is not None and data.beta_true is not None:
                try:
                    # Sort by first covariate for meaningful gradient
                    sort_idx = np.argsort(data.X[:, 0])
                    beta_hat_sorted = result.beta_hat[sort_idx]
                    beta_true_sorted = data.beta_true[sort_idx]

                    beta_hat_grad = np.gradient(beta_hat_sorted)
                    beta_true_grad = np.gradient(beta_true_sorted)

                    std_hat = np.std(beta_hat_grad)
                    std_true = np.std(beta_true_grad)

                    if std_true > 1e-10:
                        smoothness_ratio = float(std_hat / std_true)
                except Exception:
                    pass

            results.append({
                "sim_id": sim_id,
                "model": model_name,
                "method": method_name,
                "mu_hat": mu_hat,
                "se": se,
                "mu_true": mu_true,
                "bias": bias,
                "covered": covered,
                "rmse_alpha": rmse_alpha,
                "rmse_beta": rmse_beta,
                "corr_alpha": corr_alpha,
                "corr_beta": corr_beta,
                "overfit_rate": overfit_rate,
                "mean_best_epoch": mean_best_epoch,
                # Phase 3: Training diagnostics
                "final_grad_norm": final_grad_norm,
                "final_beta_std": final_beta_std,
                "final_train_loss": final_train_loss,
                "final_val_loss": final_val_loss,
                "train_val_gap": train_val_gap,
                # Phase 3: Correction diagnostics
                "correction_ratio": correction_ratio,
                "correction_mean": correction_mean,
                "correction_std": correction_std,
                "min_hessian_eig": min_hessian_eig,
                "hessian_condition": hessian_condition,
                # NEW: Oracle loss and smoothness diagnostics
                "oracle_loss": oracle_loss,
                "excess_loss_ratio": excess_loss_ratio,
                "smoothness_ratio": smoothness_ratio,
            })
        except Exception as e:
            print(f"Error in {model_name}/{method_name} sim {sim_id}: {e}")
            results.append({
                "sim_id": sim_id,
                "model": model_name,
                "method": method_name,
                "mu_hat": np.nan,
                "se": np.nan,
                "mu_true": mu_true,
                "bias": np.nan,
                "covered": False,
                "rmse_alpha": np.nan,
                "rmse_beta": np.nan,
                "corr_alpha": np.nan,
                "corr_beta": np.nan,
                "overfit_rate": np.nan,
                "mean_best_epoch": np.nan,
                "final_grad_norm": np.nan,
                "final_beta_std": np.nan,
                "final_train_loss": np.nan,
                "final_val_loss": np.nan,
                "train_val_gap": np.nan,
                "correction_ratio": np.nan,
                "correction_mean": np.nan,
                "correction_std": np.nan,
                "min_hessian_eig": np.nan,
                "hessian_condition": np.nan,
                "oracle_loss": np.nan,
                "excess_loss_ratio": np.nan,
                "smoothness_ratio": np.nan,
            })

    return results


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Monte Carlo validation for influence function inference"
    )
    parser.add_argument("--M", type=int, default=50, help="Number of MC sims")
    parser.add_argument("--N", type=int, default=1000, help="Sample size")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--n-folds", type=int, default=50, help="Cross-fitting folds")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["linear", "poisson", "logit"],
        choices=list(DGPS.keys()),
        help="Models to run",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["naive", "influence"],
        choices=list(METHODS.keys()),
        help="Inference methods",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default="mc_results.csv", help="Output CSV")
    parser.add_argument("--n-jobs", type=int, default=1, help="Parallel jobs (-1 for all cores)")
    parser.add_argument(
        "--network",
        type=str,
        default="default",
        choices=list(NETWORK_PRESETS.keys()),
        help="Network depth preset",
    )
    parser.add_argument("--log-dir", type=str, default="logs", help="Directory for training logs")
    parser.add_argument("--no-early-stop", action="store_true", help="Disable early stopping")
    parser.add_argument("--separate-nets", action="store_true", help="Use separate networks for α and β")
    parser.add_argument(
        "--target",
        type=str,
        default="beta",
        choices=["beta", "ame"],
        help="Target: beta (latent effect) or ame (observed effect). "
             "For logit: beta=log-odds, ame=marginal effect. "
             "For tobit: beta=E[beta], ame=E[beta*Phi(z)]",
    )

    args = parser.parse_args()

    # Get hidden_dims from preset
    hidden_dims = NETWORK_PRESETS[args.network]

    config = Config(
        M=args.M,
        N=args.N,
        epochs=args.epochs,
        n_folds=args.n_folds,
        seed=args.seed,
        hidden_dims=hidden_dims,
        early_stopping=not args.no_early_stop,
        separate_nets=args.separate_nets,
        target=args.target,
        log_dir=args.log_dir,
    )

    start_time = time.time()

    print("=" * 60)
    print("Influence Function Validation")
    print("=" * 60)
    print(f"M={config.M}, N={config.N}, epochs={config.epochs}, folds={config.n_folds}")
    print(f"Models: {args.models}")
    print(f"Methods: {args.methods}")
    print(f"Network: {args.network} {config.hidden_dims}")
    print(f"Separate nets: {config.separate_nets}")
    print(f"Target: {config.target}")
    print(f"Early stopping: {config.early_stopping}")
    print(f"Parallel jobs: {args.n_jobs}")
    print(f"Log dir: {config.log_dir}")
    print("=" * 60)

    all_results = []

    for model_name in args.models:
        print(f"\n=== {model_name.upper()} ===")
        dgp = get_dgp(model_name, seed=config.seed)
        # Handle tobit target for mu_true computation
        if model_name == "tobit":
            tobit_target = "observed" if config.target == "ame" else "latent"
            mu_true = dgp.compute_true_mu(target=tobit_target)
            print(f"True μ* ({tobit_target}) = {mu_true:.6f}")
        else:
            mu_true = dgp.compute_true_mu()
            print(f"True μ* = {mu_true:.6f}")

        if args.n_jobs == 1:
            # Sequential execution
            for sim_id in tqdm(range(config.M), desc=model_name):
                results = run_one_sim(sim_id, model_name, args.methods, config)
                all_results.extend(results)
        else:
            # Parallel execution
            results_list = Parallel(n_jobs=args.n_jobs, verbose=10)(
                delayed(run_one_sim)(sim_id, model_name, args.methods, config)
                for sim_id in range(config.M)
            )
            for results in results_list:
                all_results.extend(results)

    # Save raw results
    df = pd.DataFrame(all_results)
    df.to_csv(args.output, index=False)
    print(f"\nRaw results saved to: {args.output}")

    # Compute and print metrics
    metrics = compute_metrics(df)
    metrics.to_csv(args.output.replace(".csv", ".metrics.csv"), index=False)
    print(f"Metrics saved to: {args.output.replace('.csv', '.metrics.csv')}")

    print("\n")
    print_table(metrics)

    # Create comprehensive log report
    end_time = time.time()
    elapsed = end_time - start_time

    # Get mu_true from first model
    dgp = get_dgp(args.models[0], seed=config.seed)
    mu_true = dgp.compute_true_mu()

    # DGP specification (hardcoded for linear - could be made dynamic)
    dgp_spec = {
        "alpha_star": "sin(2piX1) + X2^3 - 2cos(piX3) + exp(X4/3)*I(X4>0) + 0.5*X5*X6",
        "beta_star": "cos(2piX1)*sin(piX2) + 0.8*tanh(3X3) - 0.5*X4^2 + 0.3*X5*I(X6>0)",
        "mu_true": float(mu_true),
    }

    timing = {
        "total_seconds": elapsed,
        "per_sim_seconds": elapsed / config.M if config.M > 0 else 0,
        "start_time": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(start_time)),
        "end_time": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(end_time)),
    }

    # Create and save report
    report_json = create_full_report(
        config=asdict(config),
        dgp_spec=dgp_spec,
        raw_df=df,
        metrics_df=metrics,
        models=args.models,
        methods=args.methods,
        timing=timing,
    )

    log_path = save_report(report_json, config.log_dir)
    print(f"\nComprehensive report saved to: {log_path}")

    # Also save human-readable version
    human_report = format_human_readable(report_json)
    human_path = log_path.replace(".log", "_readable.txt")
    with open(human_path, "w") as f:
        f.write(human_report)
    print(f"Human-readable report saved to: {human_path}")

    print(f"\nTotal time: {elapsed:.1f}s ({elapsed/config.M:.2f}s per simulation)")


if __name__ == "__main__":
    main()
