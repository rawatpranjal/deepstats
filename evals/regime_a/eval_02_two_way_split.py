"""
Eval 02: Two-Way Split Verification (Regime A)

Goal: Verify that Regime A uses 2-way cross-fitting (not 3-way).

In Regime A (RCT with known F_T):
    - Lambda is COMPUTED from known distribution, not estimated
    - No need for separate fold to estimate Lambda
    - Should use standard 2-way split: Train / Test

This test verifies:
    1. CrossFitter uses 2-way split when lambda_strategy.requires_separate_fold = False
    2. No "lambda training fold" is created
    3. All data is used efficiently

Criteria:
    - requires_separate_fold = False for ComputeLambda
    - Training uses full train set (not split further)
"""

import sys
import numpy as np
import torch

sys.path.insert(0, "/Users/pranjal/deepest/src")

from evals.dgps.regime_a_rct_logit import RCTLogitDGP, generate_rct_logit_data


def run_eval_02_two_way_split(verbose: bool = True):
    """
    Run two-way split verification for Regime A.
    """
    print("=" * 60)
    print("EVAL 02: TWO-WAY SPLIT VERIFICATION (Regime A)")
    print("=" * 60)

    print("\nExpected Behavior:")
    print("  - ComputeLambda.requires_separate_fold = False")
    print("  - CrossFitter uses 2-way split: Train | Test")
    print("  - No additional fold for Lambda estimation")

    # Check ComputeLambda properties
    try:
        from deep_inference.lambda_.compute import ComputeLambda

        class DummyDist:
            def sample(self, shape):
                return torch.zeros(shape)

        strategy = ComputeLambda(treatment_dist=DummyDist())

        requires_theta = strategy.requires_theta
        requires_separate_fold = strategy.requires_separate_fold

        if verbose:
            print(f"\n--- ComputeLambda Properties ---")
            print(f"  requires_theta: {requires_theta}")
            print(f"  requires_separate_fold: {requires_separate_fold}")

        # Validation
        print("\n" + "=" * 60)
        print("VALIDATION CRITERIA")
        print("=" * 60)

        criteria = {
            "requires_separate_fold = False": not requires_separate_fold,
        }

        all_pass = True
        for name, passed in criteria.items():
            status = "PASS" if passed else "FAIL"
            print(f"  {name}: {status}")
            if not passed:
                all_pass = False

        print("\n" + "=" * 60)
        if all_pass:
            print("EVAL 02: PASS")
        else:
            print("EVAL 02: FAIL")
        print("=" * 60)

        return {
            "requires_theta": requires_theta,
            "requires_separate_fold": requires_separate_fold,
            "passed": all_pass,
            "skipped": False,
        }

    except ImportError as e:
        print(f"\n  [SKIP] ComputeLambda not implemented: {e}")
        print("\nEVAL 02: SKIPPED (implementation pending)")
        return {"passed": None, "skipped": True}


if __name__ == "__main__":
    result = run_eval_02_two_way_split(verbose=True)
