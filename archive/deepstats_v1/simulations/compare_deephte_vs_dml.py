"""Compare DeepHTE vs EconML LinearDML on high-dimensional patterns.

This script runs Monte Carlo simulations to compare DeepHTE's neural network
approach against LinearDML's linear CATE assumption on tough high-dimensional
data generating processes.

Key patterns tested:
- mixed: Combination of interactions, thresholds, and periodic effects
- sparse_nonlinear: Complex effects from few variables among many

Expected results:
- LinearDML should struggle with nonlinear heterogeneity
- DeepHTE should capture complex patterns with sufficient capacity

Usage:
    python simulations/compare_deephte_vs_dml.py [--n_reps 50] [--output results.csv]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import deepstats as ds
from deepstats.simulations import make_tough_highdim_scenario


def build_formula(p: int) -> str:
    """Build formula string for p covariates."""
    covs = " + ".join([f"X{i+1}" for i in range(p)])
    return f"Y ~ a({covs}) + b({covs}) * T"


def run_deephte(data: pd.DataFrame, p: int, seed: int) -> dict:
    """Run DeepHTE on data."""
    model = ds.DeepHTE(
        formula=build_formula(p),
        backbone="mlp",
        hidden_dims=[64, 32],
        epochs=200,
        verbose=0,
        random_state=seed,
    )
    result = model.fit(data)
    return {
        "ate": result.ate,
        "ate_se": result.ate_se,
        "ite": result.ite,
    }


def run_lineardml(data: pd.DataFrame, seed: int) -> dict | None:
    """Run LinearDML on data."""
    try:
        from deepstats.comparison import EconMLWrapper

        dml = EconMLWrapper(cv=3, random_state=seed)
        result = dml.fit(data)
        return {
            "ate": result.ate,
            "ate_se": result.ate_se,
            "ite": result.ite,
        }
    except ImportError:
        return None


def run_comparison(
    n_reps: int = 50,
    patterns: list[str] | None = None,
    n: int = 2000,
    p: int = 50,
    verbose: bool = True,
) -> pd.DataFrame:
    """Run Monte Carlo comparison.

    Parameters
    ----------
    n_reps : int
        Number of repetitions per pattern.
    patterns : list[str]
        DGP patterns to test.
    n : int
        Sample size per repetition.
    p : int
        Number of covariates.
    verbose : bool
        Print progress.

    Returns
    -------
    pd.DataFrame
        Results with columns: method, pattern, rep, ate_bias, ite_rmse, ite_corr
    """
    if patterns is None:
        patterns = ["mixed", "sparse_nonlinear"]

    results = []
    dml_available = True

    for pattern in patterns:
        if verbose:
            print(f"\n{'='*60}")
            print(f"Pattern: {pattern}")
            print(f"{'='*60}")

        for rep in range(n_reps):
            if verbose and (rep + 1) % 10 == 0:
                print(f"  Rep {rep + 1}/{n_reps}")

            # Generate tough high-dim data
            data = make_tough_highdim_scenario(
                seed=rep, n=n, p=p, pattern=pattern
            )
            true_ate = data.true_ate
            true_ite = data.true_ite

            # DeepHTE
            deep_result = run_deephte(data.data, p, seed=rep)
            results.append({
                "method": "DeepHTE",
                "pattern": pattern,
                "rep": rep,
                "ate_est": deep_result["ate"],
                "ate_se": deep_result["ate_se"],
                "ate_bias": deep_result["ate"] - true_ate,
                "ite_rmse": np.sqrt(np.mean((deep_result["ite"] - true_ite)**2)),
                "ite_corr": np.corrcoef(deep_result["ite"], true_ite)[0, 1],
                "true_ate": true_ate,
            })

            # LinearDML
            if dml_available:
                dml_result = run_lineardml(data.data, seed=rep)
                if dml_result is None:
                    if verbose:
                        print("  (econml not installed, skipping LinearDML)")
                    dml_available = False
                else:
                    results.append({
                        "method": "LinearDML",
                        "pattern": pattern,
                        "rep": rep,
                        "ate_est": dml_result["ate"],
                        "ate_se": dml_result["ate_se"],
                        "ate_bias": dml_result["ate"] - true_ate,
                        "ite_rmse": np.sqrt(np.mean((dml_result["ite"] - true_ite)**2)),
                        "ite_corr": np.corrcoef(dml_result["ite"], true_ite)[0, 1],
                        "true_ate": true_ate,
                    })

    return pd.DataFrame(results)


def summarize_results(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize results by method and pattern."""
    summary = df.groupby(["pattern", "method"]).agg({
        "ate_bias": ["mean", "std"],
        "ite_rmse": ["mean", "std"],
        "ite_corr": ["mean", "std"],
    }).round(4)

    # Flatten column names
    summary.columns = ["_".join(col).strip() for col in summary.columns.values]
    return summary.reset_index()


def print_results_table(df: pd.DataFrame) -> None:
    """Print results as formatted table."""
    summary = summarize_results(df)

    print("\n" + "=" * 80)
    print("SIMULATION RESULTS: DeepHTE vs LinearDML")
    print("=" * 80)
    print(f"\n{'Pattern':<20} {'Method':<12} {'ATE Bias':<12} {'ITE RMSE':<12} {'ITE Corr':<12}")
    print("-" * 80)

    for _, row in summary.iterrows():
        pattern = row["pattern"]
        method = row["method"]
        ate_bias = f"{row['ate_bias_mean']:.3f} ({row['ate_bias_std']:.3f})"
        ite_rmse = f"{row['ite_rmse_mean']:.3f} ({row['ite_rmse_std']:.3f})"
        ite_corr = f"{row['ite_corr_mean']:.3f} ({row['ite_corr_std']:.3f})"
        print(f"{pattern:<20} {method:<12} {ate_bias:<12} {ite_rmse:<12} {ite_corr:<12}")

    print("=" * 80)
    print("Note: Values show mean (std) across simulation repetitions")
    print("ATE Bias: Closer to 0 is better")
    print("ITE RMSE: Lower is better")
    print("ITE Corr: Higher is better (captures heterogeneity)")


def main():
    parser = argparse.ArgumentParser(
        description="Compare DeepHTE vs LinearDML on high-dim patterns"
    )
    parser.add_argument(
        "--n_reps", type=int, default=50,
        help="Number of simulation repetitions (default: 50)"
    )
    parser.add_argument(
        "--n", type=int, default=2000,
        help="Sample size per simulation (default: 2000)"
    )
    parser.add_argument(
        "--p", type=int, default=50,
        help="Number of covariates (default: 50)"
    )
    parser.add_argument(
        "--output", type=str, default="simulation_results.csv",
        help="Output CSV file (default: simulation_results.csv)"
    )
    parser.add_argument(
        "--patterns", type=str, nargs="+",
        default=["mixed", "sparse_nonlinear"],
        help="DGP patterns to test"
    )

    args = parser.parse_args()

    print(f"Running comparison simulation")
    print(f"  n_reps: {args.n_reps}")
    print(f"  n: {args.n}")
    print(f"  p: {args.p}")
    print(f"  patterns: {args.patterns}")
    print(f"  output: {args.output}")

    # Run simulation
    results = run_comparison(
        n_reps=args.n_reps,
        patterns=args.patterns,
        n=args.n,
        p=args.p,
        verbose=True,
    )

    # Save results
    output_path = Path(args.output)
    results.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")

    # Print summary
    print_results_table(results)


if __name__ == "__main__":
    main()
