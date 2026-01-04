"""
Tune DeepHTE hyperparameters on a simple dataset.

Tests different configurations to find the best ITE RMSE performance.

Usage:
    python simulations/tune_deephte.py
"""

import numpy as np
import pandas as pd
from itertools import product
import time
import sys
sys.path.insert(0, "src")

import deepstats as ds
from deepstats.comparison import CausalForestWrapper, EconMLWrapper


def generate_balanced_data(n=2000, p=10, seed=42):
    """Generate balanced scenario with moderate heterogeneity."""
    np.random.seed(seed)

    # Covariates
    X = np.random.randn(n, p)

    # Treatment (randomized)
    T = np.random.binomial(1, 0.5, n)

    # True effects: moderate nonlinearity
    a_true = 0.5 * X[:, 0] + 0.3 * X[:, 1] + 0.2 * X[:, 0] * X[:, 1]
    b_true = 2.0 + 0.5 * X[:, 0] + 0.3 * X[:, 1]**2 - 0.2 * X[:, 2]

    # Outcome
    noise = np.random.randn(n) * 0.5
    Y = a_true + b_true * T + noise

    # Create DataFrame
    data = pd.DataFrame(X, columns=[f"X{i+1}" for i in range(p)])
    data["T"] = T
    data["Y"] = Y

    return data, b_true, b_true.mean()


def evaluate_deephte(data, true_ite, config, seed=42):
    """Evaluate DeepHTE with given config."""
    formula = "Y ~ a(" + " + ".join([f"X{i+1}" for i in range(10)]) + ") + b(" + " + ".join([f"X{i+1}" for i in range(10)]) + ") * T"

    try:
        model = ds.DeepHTE(
            formula=formula,
            epochs=config["epochs"],
            hidden_dims=config["hidden_dims"],
            lr=config["lr"],
            dropout=config["dropout"],
            weight_decay=config["weight_decay"],
            random_state=seed,
            verbose=0,
        )

        start = time.time()
        result = model.fit(data)
        fit_time = time.time() - start

        # Compute ITE metrics
        pred_ite = result.ite
        ite_rmse = np.sqrt(np.mean((pred_ite - true_ite)**2))
        ite_corr = np.corrcoef(pred_ite, true_ite)[0, 1]
        ate_bias = abs(result.ate - true_ite.mean())

        return {
            "ite_rmse": ite_rmse,
            "ite_corr": ite_corr,
            "ate_bias": ate_bias,
            "ate": result.ate,
            "fit_time": fit_time,
            "success": True,
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def evaluate_baselines(data, true_ite, seed=42):
    """Evaluate CausalForest and LinearDML baselines."""
    results = {}

    # CausalForest
    try:
        cf = CausalForestWrapper(n_estimators=100, random_state=seed)
        start = time.time()
        cf_result = cf.fit(data)
        cf_time = time.time() - start

        cf_ite = cf_result.ite
        results["causal_forest"] = {
            "ite_rmse": np.sqrt(np.mean((cf_ite - true_ite)**2)),
            "ite_corr": np.corrcoef(cf_ite, true_ite)[0, 1],
            "ate_bias": abs(cf_result.ate - true_ite.mean()),
            "fit_time": cf_time,
        }
    except Exception as e:
        results["causal_forest"] = {"error": str(e)}

    # LinearDML
    try:
        ldml = EconMLWrapper(random_state=seed)
        start = time.time()
        ldml_result = ldml.fit(data)
        ldml_time = time.time() - start

        ldml_ite = ldml_result.ite
        results["linear_dml"] = {
            "ite_rmse": np.sqrt(np.mean((ldml_ite - true_ite)**2)),
            "ite_corr": np.corrcoef(ldml_ite, true_ite)[0, 1],
            "ate_bias": abs(ldml_result.ate - true_ite.mean()),
            "fit_time": ldml_time,
        }
    except Exception as e:
        results["linear_dml"] = {"error": str(e)}

    return results


def main():
    print("=" * 80)
    print("DeepHTE Hyperparameter Tuning")
    print("=" * 80)

    # Generate data
    print("\nGenerating balanced dataset (n=2000, p=10)...")
    data, true_ite, true_ate = generate_balanced_data(n=2000, p=10, seed=42)
    print(f"True ATE: {true_ate:.3f}")
    print(f"True ITE range: [{true_ite.min():.2f}, {true_ite.max():.2f}]")

    # Evaluate baselines first
    print("\n" + "-" * 80)
    print("BASELINES")
    print("-" * 80)

    baselines = evaluate_baselines(data, true_ite, seed=42)

    for name, res in baselines.items():
        if "error" not in res:
            print(f"\n{name}:")
            print(f"  ITE RMSE: {res['ite_rmse']:.4f}")
            print(f"  ITE Corr: {res['ite_corr']:.4f}")
            print(f"  ATE Bias: {res['ate_bias']:.4f}")
            print(f"  Time: {res['fit_time']:.1f}s")

    # Define search space (reduced for speed)
    param_grid = {
        "epochs": [200, 500],
        "lr": [0.001, 0.005, 0.01],
        "hidden_dims": [[64, 32], [128, 64], [256, 128]],
        "dropout": [0.0, 0.1],
        "weight_decay": [1e-5, 1e-4],
    }

    # Generate all combinations
    keys = list(param_grid.keys())
    combinations = list(product(*param_grid.values()))
    n_configs = len(combinations)

    print(f"\n" + "-" * 80)
    print(f"GRID SEARCH ({n_configs} configurations)")
    print("-" * 80)

    results = []
    best_rmse = float("inf")
    best_config = None

    for i, values in enumerate(combinations):
        config = dict(zip(keys, values))

        print(f"\n[{i+1}/{n_configs}] Testing: epochs={config['epochs']}, lr={config['lr']}, "
              f"hidden={config['hidden_dims']}, dropout={config['dropout']}, wd={config['weight_decay']}")

        res = evaluate_deephte(data, true_ite, config, seed=42)

        if res["success"]:
            results.append({**config, **res})
            print(f"  -> ITE RMSE: {res['ite_rmse']:.4f}, ITE Corr: {res['ite_corr']:.4f}, Time: {res['fit_time']:.1f}s")

            if res["ite_rmse"] < best_rmse:
                best_rmse = res["ite_rmse"]
                best_config = config.copy()
                best_result = res.copy()
                print("  -> NEW BEST!")
        else:
            print(f"  -> FAILED: {res['error']}")

    # Summary
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)

    print("\n### BEST DeepHTE CONFIG ###")
    print(f"  epochs: {best_config['epochs']}")
    print(f"  lr: {best_config['lr']}")
    print(f"  hidden_dims: {best_config['hidden_dims']}")
    print(f"  dropout: {best_config['dropout']}")
    print(f"  weight_decay: {best_config['weight_decay']}")
    print(f"\n  ITE RMSE: {best_result['ite_rmse']:.4f}")
    print(f"  ITE Corr: {best_result['ite_corr']:.4f}")
    print(f"  ATE Bias: {best_result['ate_bias']:.4f}")

    print("\n### COMPARISON ###")
    print(f"\n{'Method':<20} {'ITE RMSE':>12} {'ITE Corr':>12} {'Winner':>10}")
    print("-" * 55)

    methods = [
        ("DeepHTE (tuned)", best_result["ite_rmse"], best_result["ite_corr"]),
    ]

    for name, res in baselines.items():
        if "error" not in res:
            methods.append((name, res["ite_rmse"], res["ite_corr"]))

    # Find winner
    best_method = min(methods, key=lambda x: x[1])

    for name, rmse, corr in methods:
        winner = "<-- BEST" if rmse == best_method[1] else ""
        print(f"{name:<20} {rmse:>12.4f} {corr:>12.4f} {winner:>10}")

    # Save results
    if results:
        df = pd.DataFrame(results)
        df = df.sort_values("ite_rmse")
        df.to_csv("simulations/results/tuning_results.csv", index=False)
        print(f"\nResults saved to simulations/results/tuning_results.csv")

        print("\n### TOP 5 CONFIGS ###")
        print(df[["epochs", "lr", "hidden_dims", "dropout", "weight_decay", "ite_rmse", "ite_corr"]].head().to_string(index=False))


if __name__ == "__main__":
    main()
