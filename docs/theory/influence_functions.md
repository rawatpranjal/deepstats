# Deep Learning for Individual Heterogeneity

This paper develops a framework for embedding deep neural networks into structural economic models to capture rich heterogeneity while preserving interpretability.

---

## Core Framework

**Starting point:** A parametric structural model

$$\theta^\star = \arg\min_{\theta \in \Theta} \mathbb{E}[\ell(Y, T, \theta)]$$

**Enriched model:** Parameters become functions of observables $X$

$$\theta^\star(\cdot) = \arg\min_{\theta \in \mathcal{F}} \mathbb{E}[\ell(Y, T, \theta(X))]$$

**Second-stage parameter of interest:**

$$\mu^\star = \mathbb{E}[H(X, \theta^\star(X), \tilde{t})]$$

---

## Estimation via Structured DNNs

The parameter functions are estimated by:

$$\hat{\theta}(\cdot) = \arg\min_{\theta \in \mathcal{F}_{dnn}} \frac{1}{n} \sum_{i=1}^{n} \ell(y_i, t_i, \theta(x_i))$$

**Convergence rate (Theorem 1):** For smooth parameter functions with $p$ derivatives and $d_c$ continuous covariates:

$$\|\hat{\theta}_k - \theta^\star_k\|^2_{L_2(X)} = O\left(n^{-\frac{p}{p+d_c}} \log^8 n\right)$$

> **Theorem 1 (FLM 2021):** "Under smoothness assumptions, the neural network estimator achieves the minimax optimal rate:
> $$\|\hat{\theta}_k - \theta^\star_k\|^2_{L_2} = O_P\left(n^{-\frac{p}{p+d_c}} \log^8 n\right)$$
> where p is the smoothness of θ*(·) and d_c is the dimension of continuous covariates."

---

## Inference via Influence Functions

**Influence function (Theorem 2):**

$$\psi(y, t, x, \theta, \Lambda) = H(x, \theta(x); \tilde{t}) - H_\theta(x, \theta(x); \tilde{t}) \Lambda(x)^{-1} \ell_\theta(y, t, \theta(x))$$

where:
- $\ell_\theta$ is the gradient of the loss w.r.t. $\theta$
- $\Lambda(x) = \mathbb{E}[\ell_{\theta\theta}(Y, T, \theta(x)) \mid X = x]$ is the conditional Hessian
- $H_\theta$ is the Jacobian of $H$ w.r.t. $\theta$

**Cross-fitted estimator:**

$$\hat{\mu} = \frac{1}{K} \sum_{k=1}^{K} \frac{1}{|I_k|} \sum_{i \in I_k} \psi(y_i, t_i, \hat{\theta}_k(x_i), \hat{\Lambda}_k(x_i))$$

**Asymptotic normality:** Under rate conditions $\|\hat{\theta} - \theta^\star\|_{L_2} = o_P(n^{-1/4})$:

$$\sqrt{n}(\hat{\mu} - \mu^\star) \xrightarrow{d} N(0, \Psi)$$

> **Theorem 2 (FLM 2021):** "Under rate conditions $\|\hat{\theta} - \theta^\star\|_{L_2} = o_P(n^{-1/4})$, the cross-fitted estimator satisfies:
> $$\sqrt{n}(\hat{\mu} - \mu^\star) \xrightarrow{d} N(0, \Psi)$$
> where $\Psi = E[\psi_0(W)^2]$ and $\psi_0$ is the efficient influence function."

> **Neyman Orthogonality (FLM 2021):** "The influence function ψ satisfies Neyman orthogonality, meaning first-order errors in nuisance estimation have no first-order effect on the target estimator. This is why the bias scales as O(δ²) rather than O(δ)."

**Critical Rate Condition:** The rate $n^{-1/4}$ threshold comes from the product rate requirement:
> "The product of nuisance estimation errors must satisfy $\|\hat{\theta} - \theta^\star\| \cdot \|\hat{\Lambda} - \Lambda^\star\| = o_P(n^{-1/2})$"
> — FLM (2021), Theorem 2 conditions

---

## Application: Binary Choice with Heterogeneity

**Model:**

$$P[Y=1 \mid X=x, R=r] = G(\theta_1^\star(x_d, x_a) + \theta_2^\star(x_d) r)$$

where $G(u) = 1/(1 + e^{-u})$ is the logit function.

**Average marginal effect:**

$$\text{AME}(\tilde{r}) = \mathbb{E}[G(\theta^\star(X)'\tilde{r}_1)(1 - G(\theta^\star(X)'\tilde{r}_1))\theta_2^\star(X)]$$

**Optimal personalized pricing:** Solve for $r_{opt}$ via:

$$\frac{d\Pi(r)}{dr} = 0$$

where expected profits are:

$$\Pi(r) = L\left[P(r)(M(1-D(r))r - D(r)) + (1-P(r))Mr_0\right]$$

---

## Key Insight

**Machine learning and economic structure are complements, not substitutes.**

- **ML alone** fits data well but extrapolates nonsensically and can't answer causal questions
- **Structure alone** provides interpretability but misses heterogeneity
- **Combined**: ML learns heterogeneity patterns $\theta(X)$ while structure ensures valid economics
