# Gallery

Three validated examples demonstrating `deep-inference` across model families.

---

## At a Glance

| Model | Outcome | Treatment | Covariates | Result |
|-------|---------|-----------|------------|--------|
| **Linear** | Log wages | Experience (years) | Job embeddings (64-dim) | CI covers true |
| **Logit** | Purchase (0/1) | Discount (%) | Product embeddings (64-dim) | CI covers true |
| **Poisson** | Citations | Open Access (0/1) | Abstract embeddings (64-dim) | CI covers true |

All three achieve valid 95% CI coverage with influence function correction.

---

## 1. Linear: Wage Returns to Experience

**Question:** How does experience affect wages? Does the effect vary by job type?

$$Y_i = \alpha(X_i) + \beta(X_i) \cdot T_i + \varepsilon_i$$

```python
from deep_inference import structural_dml

result = structural_dml(
    Y=wages, T=experience, X=job_embeddings,
    family='linear',
    epochs=200, n_folds=50
)

print(f"Avg return to experience: {result.mu_hat:.4f}")
print(f"95% CI: [{result.ci_lower:.4f}, {result.ci_upper:.4f}]")
```

**Validation:** Corr(true, estimated) = 0.985

---

## 2. Logit: Discount Effectiveness

**Question:** Do discounts increase purchases? Which products respond most?

$$P(Y_i = 1) = \sigma(\alpha(X_i) + \beta(X_i) \cdot T_i)$$

```python
result = structural_dml(
    Y=purchased, T=discount_pct, X=product_embeddings,
    family='logit',
    epochs=200, n_folds=50
)

# Who should get discounts?
beta_hat = result.theta_hat[:, 1]
high_responders = beta_hat > np.median(beta_hat)
```

**Validation:** Corr(true, estimated) = 0.421

---

## 3. Poisson: Open Access Citation Advantage

**Question:** Does Open Access increase citations? Which papers benefit most?

$$Y_i \sim \text{Poisson}(\exp(\alpha(X_i) + \beta(X_i) \cdot T_i))$$

```python
result = structural_dml(
    Y=citations, T=open_access, X=abstract_embeddings,
    family='poisson',
    epochs=200, n_folds=50
)

# Citation multiplier
print(f"OA multiplier: {np.exp(result.mu_hat):.2f}x")
```

**Validation:** Corr(true, estimated) = 0.709

---

## Run It Yourself

```bash
# Full gallery with validation output
python tutorials/06_multimodal_gallery.py
```

See [Multimodal Tutorial](multimodal.md) for detailed code and real embedding examples (BERT, ResNet, CLIP).
