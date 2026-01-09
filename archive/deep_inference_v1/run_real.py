"""Run inference comparison on real datasets.

Compares classical econometrics vs deep learning methods.

Usage:
    python -m deep_inference.run_real --dataset bank_marketing --n-folds 10
    python -m deep_inference.run_real --dataset fremtpl2freq --n-folds 10 --subsample 50000
"""

import argparse
import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Optional

import numpy as np

from .real_data import load_dataset, RealDataResult
from .classical import classical_logit, classical_poisson, ClassicalResult
from .families import get_family
from .inference import naive, influence, InferenceResult


@dataclass
class Config:
    """Configuration for real data inference."""
    # Data
    dataset: str = 'bank_marketing'
    subsample: Optional[int] = None

    # Model
    hidden_dims: tuple = (64, 32)
    dropout: float = 0.1

    # Training
    epochs: int = 100
    lr: float = 0.01
    batch_size: int = 64
    weight_decay: float = 1e-4
    early_stopping: bool = True
    patience: int = 10
    val_split: float = 0.1

    # Inference
    n_folds: int = 10
    seed: int = 42

    # Output
    log_dir: str = 'logs'


def run_real_inference(
    data: RealDataResult,
    family_name: str,
    config: Config,
    methods: list = ['classical', 'naive', 'influence'],
) -> dict:
    """Run inference comparison on real data.

    Args:
        data: RealDataResult from load_dataset()
        family_name: 'logit' or 'poisson'
        config: Configuration object
        methods: List of methods to run

    Returns:
        Dictionary with results for each method
    """
    results = {}
    family = get_family(family_name)

    X, T, Y = data.X, data.T, data.Y
    offset = data.offset

    # 1. Classical baseline
    if 'classical' in methods:
        print("  Running classical estimator...")
        if family_name == 'logit':
            classical_result = classical_logit(X, T, Y)
        elif family_name == 'poisson':
            classical_result = classical_poisson(X, T, Y, offset=offset)
        else:
            raise ValueError(f"Classical estimator not implemented for {family_name}")

        results['classical'] = {
            'ate': classical_result.ate,
            'se': classical_result.se,
            'ci_lower': classical_result.ci_lower,
            'ci_upper': classical_result.ci_upper,
            'pvalue': classical_result.pvalue,
            'converged': classical_result.converged,
            'summary': classical_result.model_summary,
        }

    # 2. Deep Naive
    if 'naive' in methods:
        print("  Running deep naive estimator...")
        naive_result = naive(X, T, Y, family, config)

        # Compute heterogeneity metrics
        beta_std = float(np.std(naive_result.beta_hat))

        results['naive'] = {
            'ate': naive_result.mu_hat,
            'se': naive_result.se,
            'ci_lower': naive_result.mu_hat - 1.96 * naive_result.se,
            'ci_upper': naive_result.mu_hat + 1.96 * naive_result.se,
            'beta_std': beta_std,
            'training': {
                'val_loss': naive_result.histories[0].val_loss if naive_result.histories else None,
                'best_epoch': naive_result.histories[0].best_epoch if naive_result.histories else None,
            },
        }

    # 3. Deep Influence (FLM)
    if 'influence' in methods:
        print("  Running deep influence estimator...")
        influence_result = influence(X, T, Y, family, config)

        # Compute heterogeneity metrics
        beta_std = float(np.std(influence_result.beta_hat))

        # Hessian diagnostics
        min_eigs = [d['min_eig'] for d in influence_result.hessian_diagnostics]
        conditions = [d['condition'] for d in influence_result.hessian_diagnostics]

        results['influence'] = {
            'ate': influence_result.mu_hat,
            'se': influence_result.se,
            'ci_lower': influence_result.mu_hat - 1.96 * influence_result.se,
            'ci_upper': influence_result.mu_hat + 1.96 * influence_result.se,
            'beta_std': beta_std,
            'correction_ratio': influence_result.correction_ratio,
            'hessian_diagnostics': {
                'min_eig_mean': float(np.mean(min_eigs)),
                'min_eig_min': float(np.min(min_eigs)),
                'condition_mean': float(np.mean(conditions)),
            },
        }

    return results


def print_results(
    data: RealDataResult,
    results: dict,
    family_name: str,
):
    """Print formatted results table."""
    print()
    print("=" * 80)
    print(f"REAL DATA INFERENCE: {data.metadata['name']} ({family_name.upper()})")
    print("=" * 80)
    print(f"Treatment: {data.metadata['treatment']}")
    print(f"Outcome: {data.metadata['outcome']}")
    print(f"N = {data.metadata['n_obs']:,}")
    print(f"Features = {data.metadata['n_features']}")
    print()

    # Method comparison table
    print(f"{'METHOD':<15} | {'ATE (β)':<10} | {'SE':<10} | {'95% CI':<22} | {'Notes'}")
    print("-" * 80)

    for method, res in results.items():
        ate = res['ate']
        se = res['se']
        ci = f"[{res['ci_lower']:.4f}, {res['ci_upper']:.4f}]"

        if method == 'classical':
            notes = "Homogeneous"
        elif method == 'naive':
            notes = f"β_std={res.get('beta_std', 0):.3f}"
        elif method == 'influence':
            notes = f"β_std={res.get('beta_std', 0):.3f}, R_corr={res.get('correction_ratio', 0):.2f}"
        else:
            notes = ""

        print(f"{method:<15} | {ate:>10.4f} | {se:>10.4f} | {ci:<22} | {notes}")

    print("-" * 80)

    # Heterogeneity analysis
    if 'influence' in results:
        print()
        print("HETEROGENEITY ANALYSIS:")
        beta_std = results['influence'].get('beta_std', 0)
        corr_ratio = results['influence'].get('correction_ratio', 0)
        print(f"  β(X) std dev: {beta_std:.4f}", end="")
        if beta_std > 0.1:
            print(" (substantial heterogeneity)")
        elif beta_std > 0.01:
            print(" (moderate heterogeneity)")
        else:
            print(" (minimal heterogeneity)")

        print(f"  Correction ratio: {corr_ratio:.2f}", end="")
        if corr_ratio > 0.5:
            print(f" (influence SE {(corr_ratio-1)*100:.0f}% {'larger' if corr_ratio > 1 else 'smaller'} than naive)")
        else:
            print(" (small correction)")

    # SE comparison
    if 'naive' in results and 'influence' in results:
        se_naive = results['naive']['se']
        se_influence = results['influence']['se']
        se_ratio = se_influence / se_naive if se_naive > 0 else float('nan')
        print(f"  SE ratio (influence/naive): {se_ratio:.2f}")

    print("=" * 80)


def save_results(
    data: RealDataResult,
    results: dict,
    config: Config,
    family_name: str,
):
    """Save results to JSON log file."""
    os.makedirs(config.log_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(config.log_dir, f"real_{config.dataset}_{timestamp}.log")

    report = {
        'meta': {
            'generated': datetime.now().isoformat(),
            'version': '1.0',
            'framework': 'FLM',
        },
        'config': {
            'dataset': config.dataset,
            'family': family_name,
            'n_folds': config.n_folds,
            'epochs': config.epochs,
            'hidden_dims': list(config.hidden_dims),
            'subsample': config.subsample,
        },
        'data': data.metadata,
        'results': results,
    }

    with open(log_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\nResults saved to: {log_path}")
    return log_path


def main():
    parser = argparse.ArgumentParser(description="Run inference on real datasets")
    parser.add_argument('--dataset', type=str, default='bank_marketing',
                        choices=['bank_marketing', 'fremtpl2freq', 'credit_default'],
                        help='Dataset to analyze')
    parser.add_argument('--family', type=str, default=None,
                        help='Family (auto-detected if not specified)')
    parser.add_argument('--n-folds', type=int, default=10,
                        help='Number of cross-fitting folds')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Training epochs')
    parser.add_argument('--subsample', type=int, default=None,
                        help='Subsample size for large datasets')
    parser.add_argument('--methods', nargs='+', default=['classical', 'naive', 'influence'],
                        help='Methods to run')
    parser.add_argument('--log-dir', type=str, default='logs',
                        help='Directory for log files')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    args = parser.parse_args()

    # Create config
    config = Config(
        dataset=args.dataset,
        subsample=args.subsample,
        n_folds=args.n_folds,
        epochs=args.epochs,
        log_dir=args.log_dir,
        seed=args.seed,
    )

    # Auto-detect family
    if args.family:
        family_name = args.family
    elif args.dataset in ['bank_marketing', 'credit_default']:
        family_name = 'logit'
    elif args.dataset == 'fremtpl2freq':
        family_name = 'poisson'
    else:
        raise ValueError(f"Cannot auto-detect family for {args.dataset}")

    print("=" * 80)
    print("REAL DATA INFERENCE")
    print("=" * 80)
    print(f"Dataset: {args.dataset}")
    print(f"Family: {family_name}")
    print(f"Methods: {args.methods}")
    print(f"Folds: {args.n_folds}, Epochs: {args.epochs}")
    if args.subsample:
        print(f"Subsample: {args.subsample:,}")
    print("=" * 80)
    print()

    # Load data
    print("Loading data...")
    load_kwargs = {}
    if args.subsample:
        load_kwargs['subsample'] = args.subsample
        load_kwargs['seed'] = args.seed

    data = load_dataset(args.dataset, **load_kwargs)
    print(f"  Loaded {data.metadata['n_obs']:,} observations, {data.metadata['n_features']} features")
    print(f"  Treatment rate: {data.metadata.get('treatment_rate', 0):.2%}")
    print()

    # Run inference
    print("Running inference...")
    results = run_real_inference(data, family_name, config, methods=args.methods)

    # Print and save results
    print_results(data, results, family_name)
    save_results(data, results, config, family_name)


if __name__ == '__main__':
    main()
