# Multimodal Tutorial: Text & Image Embeddings

This tutorial demonstrates `deep-inference` with **high-dimensional embeddings** as covariates X. Modern deep learning features (BERT, ResNet, CLIP) can capture rich heterogeneity in treatment effects.

---

## Why Embeddings?

Traditional econometrics uses tabular covariates (age, income, education). But rich data sources—job descriptions, product images, research abstracts—contain information that drives treatment effect heterogeneity.

**This package handles high-dimensional X seamlessly:**
- Feature embeddings (64+ dimensions)
- PCA-reduced text/image embeddings from BERT, ResNet, CLIP
- The neural network learns which dimensions drive heterogeneity in $\beta(X)$

**Note:** For very high-dimensional embeddings (384-768+), use PCA to reduce to ~64 dimensions, or ensure n/dim ratio > 50 for stable estimation.

---

## Gallery of Examples

We demonstrate three model families with realistic scenarios:

| Model | Outcome (Y) | Treatment (T) | Covariates (X) |
|-------|-------------|---------------|----------------|
| **Linear** | Log wages | Years of experience | Job embeddings (64-dim) |
| **Logit** | Purchase (0/1) | Discount % | Product embeddings (64-dim) |
| **Poisson** | Citation count | Open Access (0/1) | Abstract embeddings (64-dim) |

---

## Example 1: Linear — Wages with Job Embeddings

**Scenario:** A labor economist studies how experience affects wages. The effect may vary by job type—captured via job description embeddings.

$$Y_i = \alpha(X_i) + \beta(X_i) \cdot T_i + \varepsilon_i$$

where:
- $Y$: Log hourly wage
- $T$: Years of experience
- $X$: 64-dim embedding of job description (e.g., PCA of BERT features)

**Hypothesis:** Complex jobs have steeper experience gradients.

```python
import numpy as np
from deep_inference import structural_dml

# X: Job description embeddings (PCA-reduced from SentenceTransformer)
# T: Years of experience
# Y: Log hourly wage

result = structural_dml(
    Y=Y, T=T, X=X_embeddings,
    family='linear',
    hidden_dims=[128, 64],
    epochs=150,
    n_folds=50
)

print(result.summary())

# Analyze heterogeneity
beta_hat = result.theta_hat[:, 1]  # Individual-level effects
print(f"Effect range: [{beta_hat.min():.3f}, {beta_hat.max():.3f}]")
```

**HTE Distribution:**
```
                     True β(X)         Estimated β̂(X)
Mean                 0.030             0.043
Std Dev              0.019             0.040
Min                 -0.043            -0.217
Median               0.030             0.044
Max                  0.103             0.166
```

---

## Example 2: Logit — Purchases with Image Embeddings

**Scenario:** An e-commerce company studies discount effectiveness. Does a 10% discount work better for some products than others?

$$P(Y_i = 1) = \sigma(\alpha(X_i) + \beta(X_i) \cdot T_i)$$

where:
- $Y$: Purchase indicator (0/1)
- $T$: Discount percentage
- $X$: 64-dim embedding of product image (e.g., PCA of ResNet features)

**Hypothesis:** "Premium-looking" products may be hurt by discounts (quality signaling), while "value" products benefit.

```python
result = structural_dml(
    Y=Y_purchase, T=T_discount, X=X_image_embeddings,
    family='logit',
    hidden_dims=[128, 64],
    epochs=150,
    n_folds=50
)

# Who should get discounts?
beta_hat = result.theta_hat[:, 1]
discount_sensitive = beta_hat > np.median(beta_hat)
print(f"Products to discount: {discount_sensitive.sum()} / {len(beta_hat)}")
```

**Policy Insight:**
- Products where discounts **work** (high $\hat{\beta}$): Value-oriented products
- Products where discounts **hurt** (low $\hat{\beta}$): Premium products

---

## Example 3: Poisson — Citations with Abstract Embeddings

**Scenario:** A bibliometrics researcher studies the Open Access (OA) citation advantage. Which papers benefit most from OA?

$$Y_i \sim \text{Poisson}(\exp(\alpha(X_i) + \beta(X_i) \cdot T_i))$$

where:
- $Y$: Citation count
- $T$: Open Access indicator (0/1)
- $X$: 64-dim embedding of paper abstract (e.g., PCA of SciBERT features)

**Hypothesis:** Technical papers behind paywalls benefit more from OA than already-accessible papers.

```python
result = structural_dml(
    Y=Y_citations, T=T_open_access, X=X_abstract_embeddings,
    family='poisson',
    hidden_dims=[128, 64],
    epochs=150,
    n_folds=50
)

# Citation multiplier from OA
print(result.summary())
print(f"\nCitation multiplier: {np.exp(result.mu_hat):.2f}x")

# Which papers benefit most?
beta_hat = result.theta_hat[:, 1]
top_beneficiaries = np.argsort(beta_hat)[-100:]  # Top 100
```

**HTE Distribution:**
```
                     True β(X)         Estimated β̂(X)
Mean                 0.297             0.407
Std Dev              0.415             0.350
Min                 -1.254            -1.532
Median               0.298             0.383
Max                  1.790             1.681

Interpretation: exp(0.3) = 1.35x citation multiplier from OA
```

---

## Using Real Embeddings

Replace simulated embeddings with real ones. **For best results, use PCA to reduce high-dimensional embeddings to ~64 dimensions.**

### Text Embeddings (Sentence-Transformers)

```python
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA

model = SentenceTransformer('all-MiniLM-L6-v2')
X_raw = model.encode(texts)  # (N, 384) numpy array

# Reduce dimensions for stable estimation
pca = PCA(n_components=64)
X = pca.fit_transform(X_raw)  # (N, 64)

result = structural_dml(Y, T, X, family='linear', epochs=150, n_folds=50)
```

### Image Embeddings (torchvision)

```python
import torch
from torchvision import models
from sklearn.decomposition import PCA

resnet = models.resnet50(pretrained=True)
resnet = torch.nn.Sequential(*list(resnet.children())[:-1])

X_raw = resnet(images).squeeze().numpy()  # (N, 2048)

# Reduce dimensions
pca = PCA(n_components=64)
X = pca.fit_transform(X_raw)  # (N, 64)

result = structural_dml(Y, T, X, family='logit', epochs=150, n_folds=50)
```

### Multimodal (CLIP)

```python
import clip
from sklearn.decomposition import PCA

model, preprocess = clip.load("ViT-B/32")
X_images = model.encode_image(images)
X_texts = model.encode_text(texts)
X_raw = torch.cat([X_images, X_texts], dim=1).numpy()  # (N, 1024)

# Reduce dimensions
pca = PCA(n_components=64)
X = pca.fit_transform(X_raw)  # (N, 64)

result = structural_dml(Y, T, X, family='poisson', epochs=150, n_folds=50)
```

---

## Key Takeaways

1. **High-dimensional X works:** 64+ dim embeddings are handled seamlessly
2. **Heterogeneity captured:** The model learns which embedding dimensions drive $\beta(X)$ (correlations 0.4-0.9)
3. **Valid inference:** Influence function correction provides valid CIs for most model families
4. **Policy-relevant:** Identify *who* benefits from treatment for targeting
5. **Practical guidance:** For very high-dim embeddings, use PCA to reduce to ~64 dims, ensure n/dim > 50

---

## Run the Full Gallery

```bash
python tutorials/06_multimodal_gallery.py
```

This runs all three examples with simulated embeddings and prints:
- Point estimates and confidence intervals
- HTE distribution tables
- ASCII histograms of $\hat{\beta}(X)$
- Policy insights for each scenario
