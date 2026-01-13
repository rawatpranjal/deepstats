"""
Eval 01: θ Recovery (Regime B)

Goal: Verify network can recover nonlinear α*(x) = x² and linear β*(x) = 2x-1.

This is a critical test because:
    - α*(x) = x² requires the network to learn nonlinear features
    - A linear network would FAIL (can only fit α = a₀ + a₁x)
    - β*(x) = 2x - 1 is linear, easier to fit

Criteria:
    - Corr(α̂, α*) > 0.9 (despite nonlinear form)
    - Corr(β̂, β*) > 0.9
    - RMSE reasonable (depends on noise level)
"""

import sys
import numpy as np
import torch

sys.path.insert(0, "/Users/pranjal/deepest/src")

from evals.dgps.regime_b_linear import LinearDGP, generate_linear_data


def run_eval_01_theta_recovery(n: int = 1000, verbose: bool = True):
    """
    Run θ recovery evaluation for Regime B.
    """
    print("=" * 60)
    print("EVAL 01: θ RECOVERY (Regime B)")
    print("=" * 60)

    dgp = LinearDGP()

    print("\nDGP: Linear with x² intercept")
    print(f"  α*(x) = x²")
    print(f"  β*(x) = 2x - 1")
    print(f"  X ~ Uniform({dgp.X_low}, {dgp.X_high})")
    print(f"  T = {dgp.T_confound}X + N(0, {dgp.T_noise_std}²)")

    # Generate data
    Y, T, X, theta_true, mu_true = generate_linear_data(n=n, seed=42, dgp=dgp)

    print(f"\n--- Data Generated (n={n}) ---")
    print(f"  Y: mean={Y.mean():.4f}, std={Y.std():.4f}")
    print(f"  T: mean={T.mean():.4f}, std={T.std():.4f}")

    # Try to run inference
    try:
        from deep_inference import inference

        result = inference(
            Y=Y.numpy(),
            T=T.numpy(),
            X=X.numpy(),
            model="linear",
            target="average_slope",
            n_folds=10,
            epochs=50,
            hidden_dims=[32, 16],
            lr=0.01,
            verbose=False,
        )

        # Get theta predictions
        theta_hat = result.theta_hat  # (n, 2)

        alpha_true = theta_true[:, 0].numpy()
        beta_true = theta_true[:, 1].numpy()
        alpha_hat = theta_hat[:, 0]
        beta_hat = theta_hat[:, 1]

        # Compute metrics
        corr_alpha = np.corrcoef(alpha_true, alpha_hat)[0, 1]
        corr_beta = np.corrcoef(beta_true, beta_hat)[0, 1]
        rmse_alpha = np.sqrt(np.mean((alpha_hat - alpha_true) ** 2))
        rmse_beta = np.sqrt(np.mean((beta_hat - beta_true) ** 2))

        if verbose:
            print(f"\n--- θ Recovery ---")
            print(f"  α* (true): mean={alpha_true.mean():.4f}, std={alpha_true.std():.4f}")
            print(f"  α̂ (pred): mean={alpha_hat.mean():.4f}, std={alpha_hat.std():.4f}")
            print(f"  Corr(α̂, α*) = {corr_alpha:.4f}")
            print(f"  RMSE(α̂, α*) = {rmse_alpha:.4f}")
            print()
            print(f"  β* (true): mean={beta_true.mean():.4f}, std={beta_true.std():.4f}")
            print(f"  β̂ (pred): mean={beta_hat.mean():.4f}, std={beta_hat.std():.4f}")
            print(f"  Corr(β̂, β*) = {corr_beta:.4f}")
            print(f"  RMSE(β̂, β*) = {rmse_beta:.4f}")

        # Validation
        print("\n" + "=" * 60)
        print("VALIDATION CRITERIA")
        print("=" * 60)

        criteria = {
            "Corr(α̂, α*) > 0.85": corr_alpha > 0.85,
            "Corr(β̂, β*) > 0.85": corr_beta > 0.85,
        }

        all_pass = True
        for name, passed in criteria.items():
            status = "PASS" if passed else "FAIL"
            print(f"  {name}: {status}")
            if not passed:
                all_pass = False

        print("\n" + "=" * 60)
        if all_pass:
            print("EVAL 01: PASS")
        else:
            print("EVAL 01: FAIL")
        print("=" * 60)

        return {
            "corr_alpha": corr_alpha,
            "corr_beta": corr_beta,
            "rmse_alpha": rmse_alpha,
            "rmse_beta": rmse_beta,
            "passed": all_pass,
            "skipped": False,
        }

    except ImportError as e:
        print(f"\n  [SKIP] inference() not available: {e}")
        print("\nEVAL 01: SKIPPED (implementation pending)")
        return {"passed": None, "skipped": True}


if __name__ == "__main__":
    result = run_eval_01_theta_recovery(n=1000, verbose=True)
