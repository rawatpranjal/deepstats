# Structural DML Algorithm

A general-purpose algorithm for structural deep learning with valid inference, based on the Farrell-Liang-Misra framework.

---

## The Setup

**You have:**
- A parametric structural model defined by a loss function $\ell(y, t, \theta)$
- Observed data $\{(y_i, t_i, x_i)\}_{i=1}^n$
- A target quantity of interest $H(x, \theta; \tilde{t})$

**You want:**
- Estimates of heterogeneous parameters $\theta^*(x)$
- Valid inference on $\mu^* = E[H(X, \theta^*(X); \tilde{t})]$

---

## Algorithm Overview

```
INPUT:
  - Loss function l(y, t, theta) defining your structural model
  - Target function H(x, theta; t_tilde) defining your parameter of interest
  - Data {(y_i, t_i, x_i)}
  - Number of folds K (recommend K >= 20)

OUTPUT:
  - Point estimate mu_hat
  - Standard error SE(mu_hat)
  - 95% confidence interval
```

---

## Phase 0: Model Specification

Define your structural model components:

| Component | Description | Example (Logit) |
|-----------|-------------|-----------------|
| $\ell(y, t, \theta)$ | Loss function | Binary cross-entropy |
| $H(x, \theta; \tilde{t})$ | Target functional | $\beta(x)$ or AME |
| $\theta(x)$ | Parameters | $[\alpha(x), \beta(x)]$ |

---

## Phase 1: Data Splitting

Split data into K disjoint folds $I_1, I_2, \ldots, I_K$ of equal size.

**For three-way splitting** (required when $\Lambda$ depends on $\theta$):
- Further split each $I_k^c$ into two parts: $I_k^a$ and $I_k^b$

**When is three-way needed?**
- Two-way: $\Lambda(x)$ doesn't depend on $\theta(x)$ (linear models)
- Three-way: $\Lambda(x)$ depends on $\theta(x)$ (logit, probit, Tobit, etc.)

---

## Phase 2: First-Stage Estimation

For each fold $k = 1, \ldots, K$:

### Step 2a: Estimate Parameter Functions

Train structural DNN on training fold:

$$\hat{\theta}_k(\cdot) = \arg\min_{\theta \in \mathcal{F}_{dnn}} \sum_{i \in I_k^a} \ell(y_i, t_i, \theta(x_i))$$

### Step 2b: Compute Hessians

For each data point $i \in I_k^b$:
- Compute $\ell_{\theta\theta}(y_i, t_i, \hat{\theta}_k(x_i))$ via automatic differentiation

### Step 2c: Estimate Conditional Hessian

Regress Hessian values on $x$ to get $\hat{\Lambda}_k(x)$:

$$\hat{\Lambda}_k(\cdot) = \text{NonparametricRegression}\left(\{(x_i, \ell_{\theta\theta}(y_i, t_i, \hat{\theta}_k(x_i)))\}_{i \in I_k^b}\right)$$

---

## Phase 3: Compute Influence Function Values

For each fold $k$ and each observation $i \in I_k$ (held-out):

### Step 3a: Compute Derivatives

Using automatic differentiation:
- $\ell_\theta(y_i, t_i, \hat{\theta}_k(x_i))$ - gradient of loss w.r.t. $\theta$
- $H_\theta(x_i, \hat{\theta}_k(x_i); \tilde{t})$ - Jacobian of target w.r.t. $\theta$

### Step 3b: Evaluate Influence Function

$$\psi_{ik} = H(x_i, \hat{\theta}_k(x_i); \tilde{t}) - H_\theta(x_i, \hat{\theta}_k(x_i); \tilde{t}) \cdot \hat{\Lambda}_k(x_i)^{-1} \cdot \ell_\theta(y_i, t_i, \hat{\theta}_k(x_i))$$

---

## Phase 4: Aggregate and Compute Standard Errors

### Point Estimate

$$\hat{\mu} = \frac{1}{n} \sum_k \sum_{i \in I_k} \psi_{ik}$$

### Variance Estimate

$$\hat{\Psi} = \frac{1}{K} \sum_k \frac{1}{|I_k|} \sum_{i \in I_k} (\psi_{ik} - \hat{\mu}_k)^2$$

where $\hat{\mu}_k = \frac{1}{|I_k|} \sum_{i \in I_k} \psi_{ik}$

### Standard Error and CI

$$SE(\hat{\mu}) = \sqrt{\hat{\Psi}/n}$$

$$CI_{95\%} = [\hat{\mu} - 1.96 \cdot SE(\hat{\mu}), \hat{\mu} + 1.96 \cdot SE(\hat{\mu})]$$

---

## Neural Network Architecture

### The Parameter Layer

The key architectural insight is the **parameter layer**:

```
Standard NN (for prediction):
    Input -> Hidden -> Hidden -> ... -> Output y_hat
    Loss = (y - y_hat)^2

Structural NN (for parameter estimation):
    Input x -> Hidden -> Hidden -> ... -> Parameter layer theta_hat(x)
                                               |
    Treatment t ---------------------------> Model layer: l(y, t, theta_hat(x))
    Outcome y ------------------------------>
```

### PyTorch Implementation

```python
class StructuralDNN(nn.Module):
    def __init__(self, dim_x, dim_theta, hidden_sizes):
        super().__init__()

        # Hidden layers for learning theta(x)
        layers = []
        prev_size = dim_x
        for h in hidden_sizes:
            layers.append(nn.Linear(prev_size, h))
            layers.append(nn.ReLU())
            prev_size = h
        self.hidden = nn.Sequential(*layers)

        # Parameter layer: outputs theta(x)
        self.param_layer = nn.Linear(prev_size, dim_theta)

    def forward(self, x):
        """Returns parameter vector theta(x)"""
        h = self.hidden(x)
        theta = self.param_layer(h)
        return theta

    def structural_loss(self, y, t, x):
        theta = self.forward(x)
        return self.loss_fn(y, t, theta)
```

---

## Computing Derivatives with Autodiff

```python
def compute_influence_components(model, y, t, x, H_func):
    """
    Compute all components needed for the influence function
    using automatic differentiation.
    """
    x.requires_grad_(True)

    # Get theta_hat(x)
    theta = model(x)

    # Compute l_theta: gradient of loss w.r.t. theta
    loss = model.loss_fn(y, t, theta)
    ell_theta = torch.autograd.grad(loss, theta, create_graph=True)[0]

    # Compute l_theta_theta: Hessian of loss w.r.t. theta
    ell_theta_theta = []
    for j in range(len(theta)):
        grad_j = torch.autograd.grad(ell_theta[j], theta, retain_graph=True)[0]
        ell_theta_theta.append(grad_j)
    ell_theta_theta = torch.stack(ell_theta_theta)

    # Compute H and H_theta
    H_val = H_func(x, theta)
    H_theta = torch.autograd.grad(H_val, theta)[0]

    return theta, ell_theta, ell_theta_theta, H_val, H_theta
```

---

## Estimating $\Lambda(x)$

The conditional Hessian $\Lambda(x) = E[\ell_{\theta\theta}(Y, T, \theta(X)) | X = x]$ requires nonparametric regression:

```python
def estimate_Lambda(x_train, hessian_values, x_eval):
    """
    Regress Hessian values on x to get Lambda_hat(x).
    Can use any nonparametric method: neural net, random forest, kernel, etc.
    """
    # Option 1: Another neural network
    Lambda_model = train_regression_nn(x_train, hessian_values)
    return Lambda_model(x_eval)

    # Option 2: Random forest (often more stable)
    Lambda_model = RandomForestRegressor().fit(x_train, hessian_values)
    return Lambda_model.predict(x_eval)
```

---

## Model-Specific Examples

### Example 1: Heterogeneous Logit

```
Model: P[Y=1|X=x, T=t] = G(alpha(x) + beta(x)*t)
       where G(u) = 1/(1+exp(-u))

Loss: l(y, t, theta(x)) = -y*log(G(theta'*t_1))
                         - (1-y)*log(1-G(theta'*t_1))
      where theta = (alpha, beta)', t_1 = (1, t)'

Derivatives:
  l_theta = -(y - G(theta'*t_1)) * t_1
  l_theta_theta = G(theta'*t_1) * (1-G(theta'*t_1)) * t_1 * t_1'

Target (Average Marginal Effect):
  H(x, theta; t_tilde) = G(theta'*t_tilde) * (1-G(theta'*t_tilde)) * beta
```

### Example 2: Heterogeneous Linear Model

```
Model: E[Y|X=x, T=t] = alpha(x) + beta(x)*t

Loss: l(y, t, theta(x)) = (y - alpha(x) - beta(x)*t)^2

Derivatives:
  l_theta = -2*(y - alpha - beta*t) * (1, t)'
  l_theta_theta = 2 * [[1, t], [t, t^2]]

Lambda(x) = 2 * E[T_1 * T_1' | X=x]

Target (Average Treatment Effect):
  H(x, theta; t_tilde) = beta(x)
  H_theta = (0, 1)
```

### Example 3: Heterogeneous Tobit

```
Model: Y = max(0, Y*), Y*|X,T ~ N(alpha(x) + beta(x)*t, sigma(x)^2)

Loss: l = -1{y=0} * log(Phi(-theta_1't_1/theta_2))
        - 1{y>0} * [log(theta_2) + (1/2)*((y-theta_1't_1)*theta_2)^2]
      where theta_1 = (alpha,beta)/sigma, theta_2 = 1/sigma

Derivatives: Complex but computed automatically

Target: E[beta(X)] or E[dE[Y|X,T]/dT]
```

### Example 4: Optimal Personalized Pricing

```
First stage: Same as logit model above

Target: Optimal price r_opt(x) solving:
  max_r  Pi(r, x) = P[Y=1|x,r] * (r - c)

H is implicitly defined by FOC: dPi/dr = 0

Use envelope theorem: At optimum, dPi/dr = 0, so
  H_theta = dr_opt/d_theta via implicit function theorem
  Or: just use numerical differentiation
```

---

## Practical Recommendations

### Hyperparameter Guidelines

| Parameter | Recommendation | Notes |
|-----------|---------------|-------|
| K (folds) | 20-50 | More folds = more stable, but slower |
| Hidden layers | 2-3 | Deeper rarely helps for heterogeneity |
| Width | 10-50 nodes | Depends on sample size |
| Activation | ReLU | Standard choice |
| Optimizer | Adam | Learning rate ~0.001 |
| Epochs | Early stopping | Monitor validation loss |

### When Three-Way Splitting is Needed

**Two-way splitting suffices when:**
- $\Lambda(x)$ doesn't depend on $\theta(x)$ (linear models)
- T is randomly assigned and independent of X

**Three-way splitting required when:**
- $\Lambda(x) = E[\ell_{\theta\theta}|X]$ depends on $\theta(x)$ through $G'(\theta' \cdot t)$ terms
- Most nonlinear models (logit, probit, Tobit, etc.)

### Diagnosing Problems

**If confidence intervals are huge:**
- Check overlap: $\Lambda(x)$ may be nearly singular for some $x$
- Check positivity in propensity scores
- Consider trimming extreme regions

**If point estimates are unstable across runs:**
- Increase K (more folds)
- Use short-stacking (ensemble multiple methods)
- Simplify network architecture

**If bias correction term dominates:**
- First stage may be poorly estimated
- Consider more data or simpler model
- Check that structural model is appropriate

---

## Complete Pseudocode Summary

```
ALGORITHM: StructuralDML

INPUT: Data (Y, T, X), loss l, target H, folds K

1. SPLIT data into K folds

2. FOR k = 1 to K:
     a. Train structural DNN on fold k's complement:
        theta_k(.) = argmin sum_{i not in I_k} l(y_i, t_i, theta(x_i))

     b. Compute Hessians via autodiff:
        {l_theta_theta(y_i, t_i, theta_k(x_i))} for i not in I_k

     c. Estimate Lambda_k(.) by regressing Hessians on X

     d. FOR i in I_k (held-out):
          Compute psi_ik = H - H_theta * Lambda_k^{-1} * l_theta

3. AGGREGATE:
     mu_hat = mean(psi_ik)
     SE = sqrt(var(psi_ik)/n)

OUTPUT: mu_hat, SE, 95% CI = [mu_hat +/- 1.96*SE]
```

---

## Key Innovations

This algorithm is remarkably general. Any structural model with a smooth loss function and any smooth target functional can be handled.

The key contributions from Farrell-Liang-Misra:

1. **The architecture** that directs neural network flexibility toward the parameters
2. **The generic influence function formula** using ordinary derivatives
3. **The insight that automatic differentiation** makes this all computationally tractable
