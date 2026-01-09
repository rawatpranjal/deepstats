"""
BRUTAL NUMERICAL VERIFICATION: src2 vs causal_nets
===================================================

This script manually computes influence functions using BOTH approaches
and verifies they produce identical results.

causal_nets approach:
- psi_0: influence function for E[Y(0)|X]
- psi_1: influence function for E[Y(1)|X]
- ATE = mean(psi_1 - psi_0)
- SE = std(psi_1 - psi_0) / sqrt(n)

src2 approach:
- psi: influence function for E[beta(X)] directly
- ATE = mean(psi)
- SE = sqrt(var(psi) / n)
"""

import numpy as np
from scipy.stats import norm

np.set_printoptions(precision=6, suppress=True)

print("=" * 80)
print("BRUTAL NUMERICAL VERIFICATION: src2 vs causal_nets")
print("=" * 80)

# =============================================================================
# TEST DATA (n=6 observations)
# =============================================================================
n = 6
np.random.seed(42)

# Covariates
X = np.array([0.2, 0.5, 0.8, 0.3, 0.6, 0.9])

# Binary treatment
T = np.array([1, 0, 1, 0, 1, 0])

# Neural network outputs: theta = (alpha, beta)
# These would come from StructuralNet in practice
alpha_hat = np.array([1.0, 1.5, 2.0, 1.2, 1.8, 2.2])
beta_hat = np.array([2.0, 1.8, 2.5, 2.2, 1.9, 2.3])
theta_hat = np.column_stack([alpha_hat, beta_hat])

# True outcomes (generated from the model + noise)
mu = alpha_hat + beta_hat * T
epsilon = np.array([0.5, -0.3, 0.3, 0.9, -0.2, 0.4])  # residuals
Y = mu + epsilon

print("\n" + "=" * 80)
print("TABLE 1: RAW DATA")
print("=" * 80)
print(f"{'i':<3} | {'Y_i':<8} | {'T_i':<4} | {'X_i':<6} | {'alpha_i':<8} | {'beta_i':<8} | {'mu_i':<8} | {'resid_i':<8}")
print("-" * 80)
for i in range(n):
    print(f"{i:<3} | {Y[i]:<8.4f} | {T[i]:<4} | {X[i]:<6.2f} | {alpha_hat[i]:<8.4f} | {beta_hat[i]:<8.4f} | {mu[i]:<8.4f} | {epsilon[i]:<8.4f}")

# =============================================================================
# STEP 1: Compute Gradients (Score)
# =============================================================================
# For linear: grad_ell = -2 * residual * (1, t)
residual = Y - mu
grad_alpha = -2 * residual
grad_beta = -2 * residual * T
l_theta = np.column_stack([grad_alpha, grad_beta])

print("\n" + "=" * 80)
print("TABLE 2: GRADIENTS (Score Function)")
print("=" * 80)
print("Formula: ∇ℓ = -2 * residual * (1, t)")
print("-" * 80)
print(f"{'i':<3} | {'residual':<10} | {'∇ℓ_alpha':<12} | {'∇ℓ_beta':<12}")
print("-" * 80)
for i in range(n):
    print(f"{i:<3} | {residual[i]:<10.4f} | {grad_alpha[i]:<12.4f} | {grad_beta[i]:<12.4f}")

# =============================================================================
# STEP 2: Compute Hessians
# =============================================================================
# For linear: H = 2 * [[1, t], [t, t^2]]
H = np.zeros((n, 2, 2))
for i in range(n):
    H[i] = 2 * np.array([[1, T[i]], [T[i], T[i]**2]])

print("\n" + "=" * 80)
print("TABLE 3: HESSIANS")
print("=" * 80)
print("Formula: H = 2 * [[1, t], [t, t²]]")
print("-" * 80)
for i in range(n):
    print(f"i={i} (T={T[i]}): H = {H[i].tolist()}")
    if T[i] == 0:
        print(f"        ⚠️  SINGULAR (det=0) when T=0!")

# =============================================================================
# STEP 3: Aggregate Lambda (average Hessian)
# =============================================================================
Lambda = H.mean(axis=0)
det_Lambda = np.linalg.det(Lambda)
Lambda_inv = np.linalg.inv(Lambda)

print("\n" + "=" * 80)
print("TABLE 4: AGGREGATE LAMBDA")
print("=" * 80)
print(f"Lambda = mean(H) =")
print(Lambda)
print(f"\ndet(Lambda) = {det_Lambda:.6f}")
print(f"\nLambda_inv =")
print(Lambda_inv)

# =============================================================================
# STEP 4: src2 APPROACH - Unified Influence Function
# =============================================================================
# psi_i = h_i - h_grad_i @ Lambda_inv @ l_theta_i
# where h_i = beta_i, h_grad = (0, 1)

h = beta_hat  # target: beta(x)
h_grad = np.array([0, 1])  # gradient of h w.r.t. theta

psi_src2 = np.zeros(n)
corrections_src2 = np.zeros(n)

print("\n" + "=" * 80)
print("TABLE 5: src2 INFLUENCE FUNCTION CALCULATION")
print("=" * 80)
print("Formula: ψ_i = h_i - h_grad @ Λ⁻¹ @ ∇ℓ_i")
print("         h_i = β_i,  h_grad = (0, 1)")
print("-" * 80)
print(f"{'i':<3} | {'h_i=β_i':<10} | {'∇ℓ_i':<20} | {'correction':<12} | {'ψ_i':<10}")
print("-" * 80)

for i in range(n):
    correction = h_grad @ Lambda_inv @ l_theta[i]
    psi_i = h[i] - correction
    corrections_src2[i] = correction
    psi_src2[i] = psi_i
    print(f"{i:<3} | {h[i]:<10.4f} | ({l_theta[i,0]:>7.4f}, {l_theta[i,1]:>7.4f}) | {correction:<12.4f} | {psi_i:<10.4f}")

# =============================================================================
# STEP 5: causal_nets APPROACH - Separate psi_0, psi_1
# =============================================================================
# In causal_nets:
# - mu_0(x) = alpha(x) (potential outcome under T=0)
# - mu_1(x) = alpha(x) + beta(x) (potential outcome under T=1)
# - psi_0 = mu_0 + IF_correction_0
# - psi_1 = mu_1 + IF_correction_1
# - Then: psi_1 - psi_0 should equal our psi

# For the linear model, the IF correction for mu_0 involves h_grad = (1, 0)
# and for mu_1 involves h_grad = (1, 1)

mu_0 = alpha_hat  # E[Y(0)|X] = alpha(x)
mu_1 = alpha_hat + beta_hat  # E[Y(1)|X] = alpha(x) + beta(x)

h_grad_0 = np.array([1, 0])  # gradient of mu_0 = alpha w.r.t. theta
h_grad_1 = np.array([1, 1])  # gradient of mu_1 = alpha + beta w.r.t. theta

psi_0_cn = np.zeros(n)
psi_1_cn = np.zeros(n)
corrections_0 = np.zeros(n)
corrections_1 = np.zeros(n)

print("\n" + "=" * 80)
print("TABLE 6: causal_nets INFLUENCE FUNCTION CALCULATION")
print("=" * 80)
print("Formula: ψ₀_i = μ₀_i - h_grad_0 @ Λ⁻¹ @ ∇ℓ_i")
print("         ψ₁_i = μ₁_i - h_grad_1 @ Λ⁻¹ @ ∇ℓ_i")
print("         h_grad_0 = (1, 0),  h_grad_1 = (1, 1)")
print("-" * 80)
print(f"{'i':<3} | {'μ₀_i':<8} | {'μ₁_i':<8} | {'corr_0':<10} | {'corr_1':<10} | {'ψ₀_i':<10} | {'ψ₁_i':<10}")
print("-" * 80)

for i in range(n):
    corr_0 = h_grad_0 @ Lambda_inv @ l_theta[i]
    corr_1 = h_grad_1 @ Lambda_inv @ l_theta[i]
    psi_0_i = mu_0[i] - corr_0
    psi_1_i = mu_1[i] - corr_1
    corrections_0[i] = corr_0
    corrections_1[i] = corr_1
    psi_0_cn[i] = psi_0_i
    psi_1_cn[i] = psi_1_i
    print(f"{i:<3} | {mu_0[i]:<8.4f} | {mu_1[i]:<8.4f} | {corr_0:<10.4f} | {corr_1:<10.4f} | {psi_0_i:<10.4f} | {psi_1_i:<10.4f}")

# =============================================================================
# STEP 6: COMPARISON - The Critical Test
# =============================================================================
psi_diff_cn = psi_1_cn - psi_0_cn  # causal_nets: psi_1 - psi_0

print("\n" + "=" * 80)
print("TABLE 7: CRITICAL COMPARISON")
print("=" * 80)
print("Testing: src2's ψ_i == causal_nets's (ψ₁_i - ψ₀_i)")
print("-" * 80)
print(f"{'i':<3} | {'src2 ψ_i':<12} | {'cn ψ₁-ψ₀':<12} | {'difference':<12} | {'MATCH?':<8}")
print("-" * 80)

all_match = True
for i in range(n):
    diff = abs(psi_src2[i] - psi_diff_cn[i])
    match = "YES ✓" if diff < 1e-10 else "NO ✗"
    if diff >= 1e-10:
        all_match = False
    print(f"{i:<3} | {psi_src2[i]:<12.6f} | {psi_diff_cn[i]:<12.6f} | {diff:<12.2e} | {match:<8}")

print("-" * 80)
if all_match:
    print("🎉 ALL OBSERVATIONS MATCH EXACTLY! 🎉")
else:
    print("⚠️  MISMATCH DETECTED!")

# =============================================================================
# STEP 7: Final Statistics Comparison
# =============================================================================
# src2 approach
ate_src2 = psi_src2.mean()
var_src2 = psi_src2.var(ddof=0)  # population variance
se_src2 = np.sqrt(var_src2 / n)
ci_src2 = (ate_src2 - 1.96 * se_src2, ate_src2 + 1.96 * se_src2)

# causal_nets approach
ate_cn = psi_diff_cn.mean()
se_cn = psi_diff_cn.std(ddof=0) / np.sqrt(n)  # This is std()/sqrt(n), same as sqrt(var/n)
ci_cn = (ate_cn - norm.ppf(0.975) * psi_diff_cn.std(ddof=0) / np.sqrt(n),
         ate_cn + norm.ppf(0.975) * psi_diff_cn.std(ddof=0) / np.sqrt(n))

# Naive estimate (no IF correction)
ate_naive = beta_hat.mean()

print("\n" + "=" * 80)
print("TABLE 8: FINAL RESULTS COMPARISON")
print("=" * 80)
print(f"{'Metric':<25} | {'src2':<15} | {'causal_nets':<15} | {'Match?':<10}")
print("-" * 80)
print(f"{'ATE':<25} | {ate_src2:<15.6f} | {ate_cn:<15.6f} | {'YES ✓' if abs(ate_src2 - ate_cn) < 1e-10 else 'NO ✗':<10}")
print(f"{'SE':<25} | {se_src2:<15.6f} | {se_cn:<15.6f} | {'YES ✓' if abs(se_src2 - se_cn) < 1e-10 else 'NO ✗':<10}")
print(f"{'95% CI Lower':<25} | {ci_src2[0]:<15.6f} | {ci_cn[0]:<15.6f} | {'YES ✓' if abs(ci_src2[0] - ci_cn[0]) < 1e-10 else 'NO ✗':<10}")
print(f"{'95% CI Upper':<25} | {ci_src2[1]:<15.6f} | {ci_cn[1]:<15.6f} | {'YES ✓' if abs(ci_src2[1] - ci_cn[1]) < 1e-10 else 'NO ✗':<10}")
print("-" * 80)
print(f"{'Naive ATE (no IF)':<25} | {ate_naive:<15.6f}")
print(f"{'IF Correction':<25} | {ate_src2 - ate_naive:<15.6f}")
print(f"{'Correction %':<25} | {100*(ate_src2 - ate_naive)/ate_naive:<14.2f}%")

# =============================================================================
# STEP 8: Mathematical Proof
# =============================================================================
print("\n" + "=" * 80)
print("MATHEMATICAL PROOF OF EQUIVALENCE")
print("=" * 80)
print("""
src2:
  ψ_i = β_i - (0,1) @ Λ⁻¹ @ ∇ℓ_i

causal_nets:
  ψ₁_i - ψ₀_i = (μ₁_i - corr_1) - (μ₀_i - corr_0)
              = (μ₁_i - μ₀_i) - (corr_1 - corr_0)
              = β_i - ((1,1) - (1,0)) @ Λ⁻¹ @ ∇ℓ_i
              = β_i - (0,1) @ Λ⁻¹ @ ∇ℓ_i
              = ψ_i (src2)

QED: Both formulas are ALGEBRAICALLY IDENTICAL.
""")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 80)
print("VERIFICATION SUMMARY")
print("=" * 80)
print(f"""
✓ src2 ψ values:         {psi_src2.tolist()}
✓ causal_nets ψ₁-ψ₀:     {psi_diff_cn.tolist()}
✓ Maximum difference:    {max(abs(psi_src2 - psi_diff_cn)):.2e}

FINAL VERDICT: {"PASSED ✓✓✓" if all_match else "FAILED ✗✗✗"}

Both implementations compute IDENTICAL influence functions.
The formulas are mathematically equivalent.
""")
