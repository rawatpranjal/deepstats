# Deep Learning for Individual Heterogeneity

This page explains the theoretical framework from Farrell, Liang, and Misra (2021, 2025) that underlies `deepstats`.

## The Problem This Paper Solves

### The Context

Economists use **structural models** to understand markets, decisions, and policy effects. These models encode economic theory - for example, that demand falls when prices rise.

The problem: these models typically assume everyone responds the same way. A standard demand model estimates "a 1% price increase reduces demand by 2%" - ignoring that different customers react differently.

Meanwhile, **machine learning** finds complex patterns but is a black box. A neural network predicts behavior accurately but doesn't tell you *why* people buy, or what happens with price changes you've never tried.

### The Core Tension

The paper demonstrates this with data from a South African lending experiment:

| Approach | Fits Data | Optimal Price |
|----------|-----------|---------------|
| Random Forest | Yes | "Charge infinity" (extrapolates flat) |
| Neural Network | Yes | Bizarre, nonsensical |
| Structural Logit | Yes | Reasonable, economically sound |

**Key insight**: ML cannot learn economic structure from data alone. No amount of data teaches ML that demand curves slope downward or that prices can't be infinite.

But pure structural models miss heterogeneity.

---

## The Solution: Enriched Structural Models

### The Key Idea

Take your existing economic model and **make its parameters flexible functions of observables**.

**Standard model:**
$$P[\text{purchase}] = G(\alpha + \beta \cdot r)$$

**Enriched model:**
$$P[\text{purchase}] = G(\alpha(X) + \beta(X) \cdot r)$$

The structure (logit, downward-sloping demand) is preserved. But now different customers have different intercepts $\alpha(X)$ and price sensitivities $\beta(X)$.

### What Each Part Provides

**Structure provides:**
- Interpretable parameters ($\beta$ is price sensitivity, not just a weight)
- Sensible extrapolation (demand goes to zero as prices go to infinity)
- Ability to optimize (find profit-maximizing prices)
- Counterfactual analysis

**ML provides:**
- Rich, flexible heterogeneity patterns
- Handles many covariates with complex interactions
- No need to pre-specify functional forms

**Neither alone suffices.**

---

## Core Framework

### Starting Point: Parametric Structural Model

$$\theta^\star = \arg\min_{\theta \in \Theta} \mathbb{E}[\ell(Y, T, \theta)]$$

### Enriched Model: Parameters as Functions

$$\theta^\star(\cdot) = \arg\min_{\theta \in \mathcal{F}} \mathbb{E}[\ell(Y, T, \theta(X))]$$

### Target: Second-Stage Parameter

$$\mu^\star = \mathbb{E}[H(X, \theta^\star(X), \tilde{t})]$$

Examples of $H$:
- Average treatment effect: $H = \beta(X)$
- Average marginal effect: $H = G'(\theta' \tilde{t}) \cdot \beta(X)$
- Optimal personalized price: $H = \arg\max_r \Pi(r, X)$

---

## Technical Results

### Result 1: Estimation Converges (Theorem 1)

The structured DNN estimates parameter functions at the optimal nonparametric rate:

$$\|\hat{\theta}_k - \theta^\star_k\|^2_{L_2(X)} = O\left(n^{-\frac{p}{p+d_c}} \log^8 n\right)$$

Where:
- $p$ = smoothness of true functions
- $d_c$ = dimension of continuous covariates

**Key advantage**: The dimension of treatment $T$ doesn't matter - its relationship to outcomes is given by structure, not learned.

### Result 2: Valid Inference (Theorem 2)

The **influence function** enables valid inference despite ML first stage:

$$\psi(y, t, x, \theta, \Lambda) = H(x, \theta(x); \tilde{t}) - H_\theta(x, \theta(x); \tilde{t}) \Lambda(x)^{-1} \ell_\theta(y, t, \theta(x))$$

Where:
- $\ell_\theta$ is the gradient of loss w.r.t. $\theta$
- $\Lambda(x) = \mathbb{E}[\ell_{\theta\theta}(Y, T, \theta(x)) \mid X = x]$ is the conditional Hessian
- $H_\theta$ is the Jacobian of $H$ w.r.t. $\theta$

**DML estimator:**
$$\hat{\mu} = \frac{1}{K} \sum_{k=1}^{K} \frac{1}{|I_k|} \sum_{i \in I_k} \psi(y_i, t_i, \hat{\theta}_k(x_i), \hat{\Lambda}_k(x_i))$$

**Asymptotic normality:** Under rate conditions $\|\hat{\theta} - \theta^\star\|_{L_2} = o_P(n^{-1/4})$:

$$\sqrt{n}(\hat{\mu} - \mu^\star) \xrightarrow{d} N(0, \Psi)$$

---

## Why the Influence Function Works

### Double Robustness

The key property:
$$E[\psi - \mu^*] = O(\|\hat{\theta} - \theta^*\|^2)$$

First-order errors in $\hat{\theta}$ don't affect $\hat{\mu}$. This is why we can use regularized neural networks and still get valid inference.

### Ordinary Derivatives Suffice

A crucial insight: because we explicitly enrich a parametric model, **ordinary derivatives characterize the influence function** - no functional derivatives needed.

This means:
1. No new derivation needed for each model
2. Automatic differentiation computes everything
3. Applies to any $(ℓ, H)$ combination

---

## Application: Binary Choice with Heterogeneity

### Model

$$P[Y=1 \mid X=x, R=r] = G(\theta_1^\star(x_d, x_a) + \theta_2^\star(x_d) \cdot r)$$

where $G(u) = 1/(1 + e^{-u})$ is the logit function.

### Average Marginal Effect

$$\text{AME}(\tilde{r}) = \mathbb{E}[G(\theta^\star(X)'\tilde{r}_1)(1 - G(\theta^\star(X)'\tilde{r}_1))\theta_2^\star(X)]$$

### Optimal Personalized Pricing

Solve for $r_{\text{opt}}$ via:
$$\frac{d\Pi(r)}{dr} = 0$$

where expected profits are:
$$\Pi(r) = L\left[P(r)(M(1-D(r))r - D(r)) + (1-P(r))Mr_0\right]$$

---

## Connection to Double Machine Learning

### How DML Works

DML (Chernozhukov et al., 2018) solves: how do you get valid inference when ML estimates nuisance functions?

Two ingredients:
1. **Sample splitting** (cross-fitting)
2. **Neyman orthogonal scores**: moment conditions insensitive to nuisance estimation errors

### This Paper Extends DML

| Aspect | Standard DML | Farrell-Liang-Misra |
|--------|-------------|---------------------|
| **First stage** | Nuisance functions | Structural parameters $\theta^*(X)$ |
| **First stage interpretation** | Statistical objects | Economic objects with meaning |
| **Second stage** | Simple parameters (ATE) | Rich functionals (optimal prices, profits) |
| **Score derivation** | Case-by-case | Generic formula via ordinary derivatives |
| **Structure imposed** | Minimal | Full economic model |

### The Technical Bridge

DML requires Neyman orthogonal scores:
$$\frac{\partial}{\partial \eta} \mathbb{E}[\psi(W; \mu, \eta)] \bigg|_{\eta = \eta^*} = 0$$

Theorem 2 shows the influence function formula is Neyman orthogonal for **any** structural loss $\ell$ and target $H$.

---

## Practical Implications

### For Applied Researchers

1. **Keep your favorite models**: Logit, probit, Tobit, IV - add flexible heterogeneity while keeping interpretation
2. **Inference is straightforward**: Generic influence function formula
3. **Automatic differentiation helps**: Modern ML software computes all derivatives

### For Decision-Makers

1. **Personalization becomes possible**: Optimize differently for different customer types
2. **Counterfactuals remain valid**: Ask "what if" questions ML alone cannot answer
3. **Uncertainty is quantified**: Confidence intervals for personalized strategies

---

## Summary

The framework articulates a philosophy: **ML and economic structure are complements**.

- Use ML for finding complex patterns in high-dimensional data
- Use structure for encoding theory, enabling optimization, ensuring sensible counterfactuals

Combine them thoughtfully to get the best of both worlds.
