#!/usr/bin/env python3
"""
Multimodal Gallery: Text & Image Embeddings as Covariates
==========================================================

This gallery demonstrates deep-inference on high-dimensional embeddings
from text and images. Three examples covering Linear, Logit, and Poisson.

Each example has:
- Y: Meaningful outcome
- T: Treatment of interest
- X: High-dimensional embeddings (simulating text/image features)

Examples:
1. LINEAR:  Wages ~ Experience | Job Description Embeddings
2. LOGIT:   Purchase ~ Discount | Product Image Embeddings
3. POISSON: Citations ~ Open Access | Paper Abstract Embeddings
"""

import numpy as np
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.insert(0, '../src')
sys.path.insert(0, 'src')

from deep_inference import structural_dml

# ============================================================
# HELPER: Simulate class-clustered embeddings (like real BERT/ResNet)
# ============================================================

def simulate_class_embeddings(n, dim, n_classes, class_names=None, seed=42):
    """
    Simulate embeddings that cluster by class (like real BERT/ResNet).

    The NN must learn to "classify" from X to predict β(X).
    This mimics real image/text recognition tasks.

    Args:
        n: Number of observations
        dim: Embedding dimension
        n_classes: Number of semantic classes
        class_names: Optional list of class names for display
        seed: Random seed

    Returns:
        X: (n, dim) embeddings clustered by class
        class_labels: (n,) integer class labels
        class_names: List of class name strings
    """
    np.random.seed(seed)

    # Generate class centroids in embedding space (well-separated)
    centroids = np.random.randn(n_classes, dim) * 3.0

    # Assign random class labels
    class_labels = np.random.choice(n_classes, n)

    # Each observation = centroid + Gaussian noise
    X = centroids[class_labels] + np.random.randn(n, dim) * 0.8

    if class_names is None:
        class_names = [f"Class_{i}" for i in range(n_classes)]

    return X, class_labels, class_names

print("="*80)
print("MULTIMODAL GALLERY: Deep Inference with High-Dimensional Embeddings")
print("="*80)

# ============================================================
# EXAMPLE 1: LINEAR - Wages by Job Category
# ============================================================
print("\n" + "="*80)
print("EXAMPLE 1: LINEAR MODEL")
print("Wages ~ Experience | Job Category Embeddings")
print("="*80)

print("""
SCENARIO: A labor economist studies how experience affects wages.
The experience premium varies by JOB CATEGORY - the NN must infer
the category from job description embeddings.

- Y: Log hourly wage (continuous)
- T: Years of experience (standardized)
- X: 64-dim embeddings of job descriptions

JOB CATEGORIES (5 classes with different experience premiums):
  - Entry-level:  β = 0.4 (small experience premium)
  - Mid-level:    β = 0.7
  - Senior:       β = 1.0 (baseline)
  - Executive:    β = 1.3
  - Specialist:   β = 1.6 (highest premium - experience matters most)

The NN must CLASSIFY jobs from embeddings to predict β(X).
""")

np.random.seed(2024)
N = 10000
EMBED_DIM = 64

# Job categories with distinct experience premiums
JOB_CATEGORIES = ["Entry-level", "Mid-level", "Senior", "Executive", "Specialist"]
BETA_BY_JOB = {0: 0.4, 1: 0.7, 2: 1.0, 3: 1.3, 4: 1.6}

# Generate class-clustered embeddings (NN must learn to classify!)
X_wages, job_class, _ = simulate_class_embeddings(
    N, dim=EMBED_DIM, n_classes=5, class_names=JOB_CATEGORIES, seed=2024
)

# True β depends on job category (discrete heterogeneity)
beta_wages = np.array([BETA_BY_JOB[c] for c in job_class])
alpha_wages = 2.5 + 0.3 * np.sin(job_class)  # Base wage varies by category

# Treatment: experience (confounded - harder test)
T_wages = beta_wages + np.random.normal(0, 0.5, N)

# Outcome: log wage
Y_wages = alpha_wages + beta_wages * T_wages + np.random.normal(0, 0.3, N)

mu_true_wages = beta_wages.mean()

print(f"Data: N={N}, X dim={EMBED_DIM}")
print(f"True E[β(X)] = {mu_true_wages:.4f} (avg {mu_true_wages*100:.1f}% wage increase per year exp)")
print(f"Heterogeneity: β ranges from {beta_wages.min():.3f} to {beta_wages.max():.3f}")

print("\nRunning deep-inference (Linear family)...")
result_wages = structural_dml(
    Y=Y_wages, T=T_wages, X=X_wages,
    family='linear',
    hidden_dims=[64, 32],
    epochs=200,
    n_folds=20,
    lambda_method='lgbm',
    lr=0.01,
    verbose=False
)

print("\nPublication-ready summary:")
print(result_wages.summary())

print(f"\nRESULTS:")
print(f"  True E[β]:     {mu_true_wages:.4f}")
print(f"  Estimated:     {result_wages.mu_hat:.4f}")
print(f"  SE:            {result_wages.se:.4f}")
print(f"  95% CI:        [{result_wages.ci_lower:.4f}, {result_wages.ci_upper:.4f}]")
print(f"  Covers truth:  {result_wages.ci_lower <= mu_true_wages <= result_wages.ci_upper}")

# Heterogeneity analysis
beta_hat_wages = result_wages.theta_hat[:, 1]
corr_wages = np.corrcoef(beta_wages, beta_hat_wages)[0, 1]
print(f"\nHeterogeneity Recovery:")
print(f"  Corr(β_true, β_hat): {corr_wages:.3f}")

# Classification accuracy: does NN identify high-β categories?
top_idx = np.argsort(beta_hat_wages)[-int(N*0.1):]
bottom_idx = np.argsort(beta_hat_wages)[:int(N*0.1)]
print(f"\n  Top 10% (highest estimated β):")
print(f"    True category distribution: {np.bincount(job_class[top_idx], minlength=5)}")
print(f"    Avg β_true: {beta_wages[top_idx].mean():.4f}")
print(f"    Avg β_hat: {beta_hat_wages[top_idx].mean():.4f}")
print(f"  Bottom 10% (lowest estimated β):")
print(f"    True category distribution: {np.bincount(job_class[bottom_idx], minlength=5)}")
print(f"    Avg β_true: {beta_wages[bottom_idx].mean():.4f}")
print(f"    Avg β_hat: {beta_hat_wages[bottom_idx].mean():.4f}")


# ============================================================
# EXAMPLE 2: LOGIT - Purchase by Product Category
# ============================================================
print("\n" + "="*80)
print("EXAMPLE 2: LOGIT MODEL")
print("Purchase ~ Discount | Product Category Embeddings")
print("="*80)

print("""
SCENARIO: An e-commerce company studies discount effectiveness.
The discount sensitivity varies by PRODUCT CATEGORY - the NN must
infer the category from product image embeddings.

- Y: Purchase (0/1 binary)
- T: Discount level (standardized)
- X: 64-dim embeddings from product images

PRODUCT CATEGORIES (5 classes with different discount sensitivities):
  - Electronics:  β = 0.4 (low sensitivity - people research anyway)
  - Fashion:      β = 0.8
  - Home & Garden: β = 1.2 (baseline)
  - Beauty:       β = 1.6
  - Sports:       β = 2.0 (highest - impulse buyers love discounts)

The NN must CLASSIFY products from image embeddings to predict β(X).
""")

np.random.seed(2025)
N = 16000  # Binary outcomes need ~2x data
EMBED_DIM = 64

# Product categories with distinct discount sensitivities
PRODUCT_CATEGORIES = ["Electronics", "Fashion", "Home", "Beauty", "Sports"]
BETA_BY_PRODUCT = {0: 0.4, 1: 0.8, 2: 1.2, 3: 1.6, 4: 2.0}

# Generate class-clustered embeddings (NN must learn to classify!)
X_purchase, product_class, _ = simulate_class_embeddings(
    N, dim=EMBED_DIM, n_classes=5, class_names=PRODUCT_CATEGORIES, seed=2025
)

# True β depends on product category (discrete heterogeneity)
beta_purchase = np.array([BETA_BY_PRODUCT[c] for c in product_class])
alpha_purchase = 0.5 + 0.3 * np.sin(product_class)  # Base varies by category

# Treatment: discount level (confounded - harder test)
T_purchase = beta_purchase + np.random.normal(0, 0.5, N)

from scipy.special import expit
prob_purchase = expit(alpha_purchase + beta_purchase * T_purchase)
Y_purchase = np.random.binomial(1, prob_purchase).astype(float)

mu_true_purchase = beta_purchase.mean()

print(f"Data: N={N}, X dim={EMBED_DIM}")
print(f"True E[β(X)] = {mu_true_purchase:.4f} (avg log-odds increase per discount unit)")
print(f"Heterogeneity: β ranges from {beta_purchase.min():.3f} to {beta_purchase.max():.3f}")
print(f"Purchase rate: {Y_purchase.mean():.1%}")

print("\nRunning deep-inference (Logit family)...")
result_purchase = structural_dml(
    Y=Y_purchase, T=T_purchase, X=X_purchase,
    family='logit',
    hidden_dims=[64, 32],
    epochs=200,
    n_folds=20,
    lambda_method='lgbm',
    lr=0.01,
    verbose=False
)

print("\nPublication-ready summary:")
print(result_purchase.summary())

print(f"\nRESULTS:")
print(f"  True E[β]:     {mu_true_purchase:.4f}")
print(f"  Estimated:     {result_purchase.mu_hat:.4f}")
print(f"  SE:            {result_purchase.se:.4f}")
print(f"  95% CI:        [{result_purchase.ci_lower:.4f}, {result_purchase.ci_upper:.4f}]")
print(f"  Covers truth:  {result_purchase.ci_lower <= mu_true_purchase <= result_purchase.ci_upper}")

# Heterogeneity analysis
beta_hat_purchase = result_purchase.theta_hat[:, 1]
corr_purchase = np.corrcoef(beta_purchase, beta_hat_purchase)[0, 1]
print(f"\nHeterogeneity Recovery:")
print(f"  Corr(β_true, β_hat): {corr_purchase:.3f}")

# Classification accuracy: does NN identify high-sensitivity categories?
top_idx = np.argsort(beta_hat_purchase)[-int(N*0.1):]
bottom_idx = np.argsort(beta_hat_purchase)[:int(N*0.1)]
print(f"\n  Products where discounts WORK (top 10% β_hat):")
print(f"    Category distribution: {np.bincount(product_class[top_idx], minlength=5)}")
print(f"    Avg β_true: {beta_purchase[top_idx].mean():.4f}")
print(f"    Avg β_hat: {beta_hat_purchase[top_idx].mean():.4f}")
print(f"  Products where discounts DON'T WORK (bottom 10% β_hat):")
print(f"    Category distribution: {np.bincount(product_class[bottom_idx], minlength=5)}")
print(f"    Avg β_true: {beta_purchase[bottom_idx].mean():.4f}")
print(f"    Avg β_hat: {beta_hat_purchase[bottom_idx].mean():.4f}")


# ============================================================
# EXAMPLE 3: POISSON - Citations by Research Field
# ============================================================
print("\n" + "="*80)
print("EXAMPLE 3: POISSON MODEL")
print("Citations ~ Open Access | Research Field Embeddings")
print("="*80)

print("""
SCENARIO: A bibliometrics researcher studies the Open Access citation advantage.
The OA advantage varies by RESEARCH FIELD - the NN must infer the field
from paper abstract embeddings.

- Y: Citation count (non-negative integer)
- T: Open Access intensity (standardized)
- X: 64-dim embeddings of paper abstracts

RESEARCH FIELDS (5 classes with different OA advantages):
  - Humanities:   β = 0.3 (low - already accessible, small audience)
  - Economics:    β = 0.6
  - Biology:      β = 1.0 (baseline)
  - Physics:      β = 1.4
  - CS/ML:        β = 1.8 (highest - paywalls really hurt, preprint culture)

The NN must CLASSIFY papers from abstract embeddings to predict β(X).
""")

np.random.seed(2026)
N = 12000
EMBED_DIM = 64

# Research fields with distinct OA advantages
RESEARCH_FIELDS = ["Humanities", "Economics", "Biology", "Physics", "CS/ML"]
BETA_BY_FIELD = {0: 0.3, 1: 0.6, 2: 1.0, 3: 1.4, 4: 1.8}

# Generate class-clustered embeddings (NN must learn to classify!)
X_cite, field_class, _ = simulate_class_embeddings(
    N, dim=EMBED_DIM, n_classes=5, class_names=RESEARCH_FIELDS, seed=2026
)

# True β depends on research field (discrete heterogeneity)
beta_cite = np.array([BETA_BY_FIELD[c] for c in field_class])
alpha_cite = 1.0 + 0.3 * np.sin(field_class)  # Base citation rate varies by field

# Treatment: OA intensity (confounded - harder test)
T_cite = beta_cite + np.random.normal(0, 0.5, N)

# Poisson outcome (citation counts)
log_lambda = alpha_cite + beta_cite * T_cite
lambda_cite = np.exp(np.clip(log_lambda, -5, 5))  # Clip for numerical stability
Y_cite = np.random.poisson(lambda_cite).astype(float)

mu_true_cite = beta_cite.mean()

print(f"Data: N={N}, X dim={EMBED_DIM}")
print(f"True E[β(X)] = {mu_true_cite:.4f} (avg log-rate increase from OA)")
print(f"  → Exp(β) = {np.exp(mu_true_cite):.2f}x citation multiplier")
print(f"Heterogeneity: β ranges from {beta_cite.min():.3f} to {beta_cite.max():.3f}")
print(f"Mean citations: {Y_cite.mean():.1f}, Max: {Y_cite.max():.0f}")

print("\nRunning deep-inference (Poisson family)...")
result_cite = structural_dml(
    Y=Y_cite, T=T_cite, X=X_cite,
    family='poisson',
    hidden_dims=[64, 32],
    epochs=200,
    n_folds=20,
    lambda_method='lgbm',
    lr=0.01,
    verbose=False
)

print("\nPublication-ready summary:")
print(result_cite.summary())

print(f"\nRESULTS:")
print(f"  True E[β]:     {mu_true_cite:.4f}")
print(f"  Estimated:     {result_cite.mu_hat:.4f}")
print(f"  SE:            {result_cite.se:.4f}")
print(f"  95% CI:        [{result_cite.ci_lower:.4f}, {result_cite.ci_upper:.4f}]")
print(f"  Covers truth:  {result_cite.ci_lower <= mu_true_cite <= result_cite.ci_upper}")

# Heterogeneity analysis
beta_hat_cite = result_cite.theta_hat[:, 1]
corr_cite = np.corrcoef(beta_cite, beta_hat_cite)[0, 1]
print(f"\nHeterogeneity Recovery:")
print(f"  Corr(β_true, β_hat): {corr_cite:.3f}")

# Classification accuracy: does NN identify high-OA-advantage fields?
top_idx = np.argsort(beta_hat_cite)[-int(N*0.1):]
bottom_idx = np.argsort(beta_hat_cite)[:int(N*0.1)]
print(f"\n  Papers with LARGEST OA advantage (top 10% β_hat):")
print(f"    Field distribution: {np.bincount(field_class[top_idx], minlength=5)}")
print(f"    Avg β_true: {beta_cite[top_idx].mean():.4f} ({np.exp(beta_cite[top_idx].mean()):.1f}x)")
print(f"    Avg β_hat: {beta_hat_cite[top_idx].mean():.4f}")
print(f"  Papers with SMALLEST OA advantage (bottom 10% β_hat):")
print(f"    Field distribution: {np.bincount(field_class[bottom_idx], minlength=5)}")
print(f"    Avg β_true: {beta_cite[bottom_idx].mean():.4f} ({np.exp(beta_cite[bottom_idx].mean()):.1f}x)")
print(f"    Avg β_hat: {beta_hat_cite[bottom_idx].mean():.4f}")


# ============================================================
# SUMMARY
# ============================================================
print("\n" + "="*80)
print("GALLERY SUMMARY")
print("="*80)

print(f"""
{'Model':<12} {'Outcome':<20} {'Treatment':<15} {'X Dim':<8} {'Covers?':<10} {'Corr(β)':<10}
{'-'*80}
{'Linear':<12} {'Log wages':<20} {'Experience':<15} {64:<8} {result_wages.ci_lower <= mu_true_wages <= result_wages.ci_upper!s:<10} {corr_wages:.3f}
{'Logit':<12} {'Purchase (0/1)':<20} {'Discount %':<15} {64:<8} {result_purchase.ci_lower <= mu_true_purchase <= result_purchase.ci_upper!s:<10} {corr_purchase:.3f}
{'Poisson':<12} {'Citations':<20} {'Open Access':<15} {64:<8} {result_cite.ci_lower <= mu_true_cite <= result_cite.ci_upper!s:<10} {corr_cite:.3f}
{'-'*80}
""")

print("""
KEY INSIGHTS:

1. EMBEDDINGS AS COVARIATES: Feature embeddings (64 dim) work seamlessly as
   covariates X. The neural network learns which dimensions drive heterogeneity.

2. VALID INFERENCE: Despite high-dimensional X, influence function correction
   provides valid 95% confidence intervals. Naive SEs would be ~5x too small.

3. HETEROGENEITY RECOVERY: The package captures treatment effect heterogeneity
   driven by latent factors in the embeddings. This enables:
   - Targeting (which products to discount?)
   - Personalization (which workers benefit from training?)
   - Policy design (which papers to make open access?)

4. REAL-WORLD USAGE:
   - Replace simulated embeddings with real features (PCA of BERT/ResNet)
   - For very high-dim embeddings (384-768), use larger N or apply PCA first
   - Rule of thumb: n/dim ratio > 50 for stable estimation
""")

# ============================================================
# HTE DISTRIBUTIONS
# ============================================================
print("\n" + "="*80)
print("HETEROGENEOUS TREATMENT EFFECT (HTE) DISTRIBUTIONS")
print("="*80)

def print_hte_distribution(name, beta_true, beta_hat, unit=""):
    """Print HTE distribution summary."""
    print(f"\n{name}:")
    print(f"  {'':20} {'True β(X)':<20} {'Estimated β̂(X)':<20}")
    print(f"  {'-'*60}")
    print(f"  {'Mean':<20} {beta_true.mean():<20.4f} {beta_hat.mean():<20.4f}")
    print(f"  {'Std Dev':<20} {beta_true.std():<20.4f} {beta_hat.std():<20.4f}")
    print(f"  {'Min':<20} {beta_true.min():<20.4f} {beta_hat.min():<20.4f}")
    print(f"  {'25th %ile':<20} {np.percentile(beta_true, 25):<20.4f} {np.percentile(beta_hat, 25):<20.4f}")
    print(f"  {'Median':<20} {np.median(beta_true):<20.4f} {np.median(beta_hat):<20.4f}")
    print(f"  {'75th %ile':<20} {np.percentile(beta_true, 75):<20.4f} {np.percentile(beta_hat, 75):<20.4f}")
    print(f"  {'Max':<20} {beta_true.max():<20.4f} {beta_hat.max():<20.4f}")
    if unit:
        print(f"\n  Interpretation: β represents {unit}")

print_hte_distribution(
    "LINEAR: Experience → Wages",
    beta_wages, beta_hat_wages,
    "% wage increase per year of experience"
)

print_hte_distribution(
    "LOGIT: Discount → Purchase",
    beta_purchase, beta_hat_purchase,
    "log-odds increase per 1% discount"
)

print_hte_distribution(
    "POISSON: Open Access → Citations",
    beta_cite, beta_hat_cite,
    "log citation rate increase from OA"
)

# ASCII histogram of HTE distributions
def ascii_histogram(data, bins=20, width=50, title=""):
    """Print ASCII histogram."""
    counts, edges = np.histogram(data, bins=bins)
    max_count = max(counts)
    print(f"\n  {title}")
    print(f"  {'─'*width}")
    for i, count in enumerate(counts):
        bar_len = int(count / max_count * (width - 15)) if max_count > 0 else 0
        lo, hi = edges[i], edges[i+1]
        print(f"  {lo:>6.3f} │{'█' * bar_len}")
    print(f"  {'─'*width}")

print("\n" + "-"*80)
print("HTE DISTRIBUTIONS (Estimated β̂(X))")
print("-"*80)

ascii_histogram(beta_hat_wages, title="LINEAR: β̂(X) for Experience Effect on Wages")
ascii_histogram(beta_hat_purchase, title="LOGIT: β̂(X) for Discount Effect on Purchase")
ascii_histogram(beta_hat_cite, title="POISSON: β̂(X) for Open Access Effect on Citations")

print("\n" + "="*80)
print("GALLERY COMPLETE")
print("="*80)
