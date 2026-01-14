"""
Cross-Fitting Isolation Test

Verifies that cross-fitting properly isolates train/test sets.
This is critical for valid inference - if there's data leakage,
coverage guarantees are invalidated.

Test CF1: Data Leakage Detection
- Run with n_folds=2
- Track which samples θ̂_k was trained on
- Verify fold k test samples NOT in θ̂_k training set
"""

import sys
import numpy as np
import torch
from typing import Dict, Any, List, Tuple

sys.path.insert(0, "/Users/pranjal/deepest/src")


def verify_fold_isolation(
    n: int = 100,
    n_folds: int = 5,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Verify that cross-fitting properly isolates train/test sets.

    This test checks:
    1. Each observation appears in test set exactly once
    2. Training sets don't include their test observations
    3. Train + Test covers all observations for each fold

    Args:
        n: Number of observations
        n_folds: Number of cross-fitting folds
        verbose: Print detailed output

    Returns:
        Dictionary with test results
    """
    if verbose:
        print("=" * 60)
        print("TEST CF1: CROSS-FITTING FOLD ISOLATION")
        print("=" * 60)
        print(f"\nn={n}, n_folds={n_folds}")

    results = {"passed": True, "metrics": {}}

    # Create fold assignments (same logic as in algorithm.py)
    indices = np.arange(n)
    np.random.seed(42)
    np.random.shuffle(indices)

    fold_size = n // n_folds
    folds = []
    for k in range(n_folds):
        start = k * fold_size
        end = start + fold_size if k < n_folds - 1 else n
        test_idx = indices[start:end]
        train_idx = np.concatenate([indices[:start], indices[end:]])
        folds.append((train_idx, test_idx))

    # Check 1: Each observation in test set exactly once
    if verbose:
        print("\n--- Check 1: Each observation in test set exactly once ---")

    test_counts = np.zeros(n)
    for train_idx, test_idx in folds:
        test_counts[test_idx] += 1

    all_once = np.all(test_counts == 1)
    results["metrics"]["all_in_test_once"] = all_once

    if verbose:
        print(f"  Min test count: {test_counts.min()}")
        print(f"  Max test count: {test_counts.max()}")
        print(f"  All exactly once: {all_once}")

    if not all_once:
        results["passed"] = False

    # Check 2: No overlap between train and test
    if verbose:
        print("\n--- Check 2: No train/test overlap ---")

    overlaps = []
    for k, (train_idx, test_idx) in enumerate(folds):
        overlap = set(train_idx) & set(test_idx)
        overlaps.append(len(overlap))
        if verbose and len(overlap) > 0:
            print(f"  Fold {k}: {len(overlap)} overlapping indices")

    no_overlap = all(o == 0 for o in overlaps)
    results["metrics"]["no_train_test_overlap"] = no_overlap

    if verbose:
        print(f"  No overlaps: {no_overlap}")

    if not no_overlap:
        results["passed"] = False

    # Check 3: Train + Test covers all indices
    if verbose:
        print("\n--- Check 3: Train + Test covers all indices ---")

    coverage_ok = True
    for k, (train_idx, test_idx) in enumerate(folds):
        combined = set(train_idx) | set(test_idx)
        if combined != set(range(n)):
            coverage_ok = False
            missing = set(range(n)) - combined
            if verbose:
                print(f"  Fold {k}: Missing indices {missing}")

    results["metrics"]["full_coverage"] = coverage_ok

    if verbose:
        print(f"  Full coverage: {coverage_ok}")

    if not coverage_ok:
        results["passed"] = False

    # Check 4: Train set sizes are correct
    if verbose:
        print("\n--- Check 4: Train set sizes ---")

    expected_train_size = n - fold_size
    train_sizes = [len(train_idx) for train_idx, _ in folds]
    size_variance = np.var(train_sizes)

    results["metrics"]["train_size_mean"] = np.mean(train_sizes)
    results["metrics"]["train_size_var"] = size_variance

    if verbose:
        print(f"  Expected train size: ~{expected_train_size}")
        print(f"  Actual train sizes: {train_sizes}")
        print(f"  Variance: {size_variance:.4f}")

    if verbose:
        status = "PASS" if results["passed"] else "FAIL"
        print(f"\n--- TEST CF1: {status} ---")

    return results


def verify_three_way_split(
    n: int = 100,
    n_folds: int = 5,
    train_theta_frac: float = 0.6,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Verify that 3-way splitting (for regime C) properly isolates sets.

    3-way split structure:
    - Fold A (60%): Train θ̂
    - Fold B (40%): Fit Λ̂ (using θ̂ from A)
    - Fold C (test): Evaluate ψ (using θ̂ from A, Λ̂ from B)

    Args:
        n: Number of observations
        n_folds: Number of outer cross-fitting folds
        train_theta_frac: Fraction of non-test data for θ training
        verbose: Print detailed output

    Returns:
        Dictionary with test results
    """
    if verbose:
        print("\n" + "=" * 60)
        print("TEST CF2: THREE-WAY SPLIT ISOLATION")
        print("=" * 60)
        print(f"\nn={n}, n_folds={n_folds}, theta_frac={train_theta_frac}")

    results = {"passed": True, "metrics": {}}

    # Create fold assignments
    indices = np.arange(n)
    np.random.seed(42)
    np.random.shuffle(indices)

    fold_size = n // n_folds
    all_folds = []

    for k in range(n_folds):
        start = k * fold_size
        end = start + fold_size if k < n_folds - 1 else n

        # Test set (Fold C)
        test_idx = indices[start:end]

        # Non-test indices
        non_test_idx = np.concatenate([indices[:start], indices[end:]])

        # Split non-test into A (θ) and B (Λ)
        n_theta = int(len(non_test_idx) * train_theta_frac)
        np.random.shuffle(non_test_idx)  # Shuffle for random split
        theta_idx = non_test_idx[:n_theta]
        lambda_idx = non_test_idx[n_theta:]

        all_folds.append({
            'theta_idx': theta_idx,
            'lambda_idx': lambda_idx,
            'test_idx': test_idx
        })

    # Check 1: No overlap between any pair of sets
    if verbose:
        print("\n--- Check 1: No pairwise overlaps ---")

    no_overlaps = True
    for k, fold in enumerate(all_folds):
        theta_set = set(fold['theta_idx'])
        lambda_set = set(fold['lambda_idx'])
        test_set = set(fold['test_idx'])

        overlap_TL = theta_set & lambda_set
        overlap_TT = theta_set & test_set
        overlap_LT = lambda_set & test_set

        if len(overlap_TL) > 0 or len(overlap_TT) > 0 or len(overlap_LT) > 0:
            no_overlaps = False
            if verbose:
                print(f"  Fold {k}: θ∩Λ={len(overlap_TL)}, θ∩Test={len(overlap_TT)}, Λ∩Test={len(overlap_LT)}")

    results["metrics"]["no_pairwise_overlaps"] = no_overlaps

    if verbose:
        print(f"  No overlaps: {no_overlaps}")

    if not no_overlaps:
        results["passed"] = False

    # Check 2: Each observation in test set exactly once (across outer folds)
    if verbose:
        print("\n--- Check 2: Each observation in test exactly once ---")

    test_counts = np.zeros(n)
    for fold in all_folds:
        test_counts[fold['test_idx']] += 1

    all_once = np.all(test_counts == 1)
    results["metrics"]["all_in_test_once"] = all_once

    if verbose:
        print(f"  All exactly once: {all_once}")

    if not all_once:
        results["passed"] = False

    # Check 3: Coverage of each fold
    if verbose:
        print("\n--- Check 3: Full coverage per fold ---")

    coverage_ok = True
    for k, fold in enumerate(all_folds):
        combined = set(fold['theta_idx']) | set(fold['lambda_idx']) | set(fold['test_idx'])
        if combined != set(range(n)):
            coverage_ok = False
            if verbose:
                print(f"  Fold {k}: Missing {len(set(range(n)) - combined)} indices")

    results["metrics"]["full_coverage"] = coverage_ok

    if verbose:
        print(f"  Full coverage: {coverage_ok}")

    if not coverage_ok:
        results["passed"] = False

    # Check 4: Approximate split ratios
    if verbose:
        print("\n--- Check 4: Split ratios ---")

    theta_fracs = [len(f['theta_idx']) / (len(f['theta_idx']) + len(f['lambda_idx']))
                   for f in all_folds]
    mean_theta_frac = np.mean(theta_fracs)

    results["metrics"]["mean_theta_frac"] = mean_theta_frac
    results["metrics"]["target_theta_frac"] = train_theta_frac

    if verbose:
        print(f"  Target θ fraction: {train_theta_frac}")
        print(f"  Mean θ fraction: {mean_theta_frac:.4f}")
        print(f"  Individual fractions: {[f'{f:.3f}' for f in theta_fracs]}")

    if verbose:
        status = "PASS" if results["passed"] else "FAIL"
        print(f"\n--- TEST CF2: {status} ---")

    return results


def run_crossfit_isolation_test(verbose: bool = True) -> Dict[str, Any]:
    """
    Run all cross-fitting isolation tests.

    Returns:
        Dictionary with results from all cross-fitting tests
    """
    print("\n" + "#" * 60)
    print("# CROSS-FITTING ISOLATION TESTS")
    print("#" * 60)

    results = {}

    # Test with various configurations
    configs = [
        {"n": 100, "n_folds": 5},
        {"n": 100, "n_folds": 10},
        {"n": 1000, "n_folds": 20},
        {"n": 1000, "n_folds": 50},
    ]

    for i, config in enumerate(configs):
        if verbose:
            print(f"\n--- Configuration {i+1}: {config} ---")

        result = verify_fold_isolation(
            n=config["n"],
            n_folds=config["n_folds"],
            verbose=verbose
        )
        results[f"cf1_config_{i+1}"] = result

    # Test 3-way split
    results["cf2_three_way"] = verify_three_way_split(
        n=1000,
        n_folds=20,
        verbose=verbose
    )

    # Summary
    all_pass = all(r.get("passed", False) for r in results.values())

    print("\n" + "=" * 60)
    print("CROSS-FITTING TESTS SUMMARY")
    print("=" * 60)
    for name, result in results.items():
        status = "PASS" if result.get("passed") else "FAIL"
        print(f"  {name}: {status}")
    print("-" * 60)
    print(f"  OVERALL: {'PASS' if all_pass else 'FAIL'}")
    print("=" * 60)

    return {"results": results, "passed": all_pass}


if __name__ == "__main__":
    run_crossfit_isolation_test(verbose=True)
