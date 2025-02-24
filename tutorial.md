## Deep Learning for Parameter Functions (Core Math)

$Notation:$  
- $Y \in \mathbb{R}^{d_Y}$ is the outcome variable.  
- $T \in \mathbb{R}^{d_T}$ is the policy/treatment vector of interest.  
- $X \in \mathbb{R}^{d_X}$ are extra observed characteristics driving heterogeneity (potentially high-dimensional).  
- We have data $\{(y_i,t_i,x_i)\}_{i=1}^n$.  

We enrich a conventional parametric model by allowing its parameters to vary with $X$. Formally, we replace a fixed parameter $\theta$ by a parameter function $\theta_0(x) \in \mathbb{R}^{d_\theta}$ that solves

$$
\theta_0(\cdot) 
= 
\underset{\theta(\cdot)}{\arg\min} 
\;\mathbb{E}\Bigl[\;\ell\bigl(Y,\;T,\;\theta(X)\bigr)\Bigr].
$$

Here $\ell(\cdot)$ is the per-observation loss—often a log-likelihood or other classical M-estimation objective. Because $\theta_0(x)$ directly generalizes the usual parameter vector, these functions retain interpretability: each coordinate of $\theta_0(x)$ corresponds to a structural or econometric parameter that is now individualized by $x$.  

### Second-Step Parameter of Interest

In many applications, one does not just want $\theta_0(x)$ itself, but a finite-dimensional summary

$$
\mu_0 
= 
\mathbb{E}\Bigl[\;H\bigl(X,\;\theta_0(X)\bigr)\Bigr].
$$

The function $H$ can encode, for instance, average slopes, average marginal effects, welfare measures, or other policy-relevant summaries. Once we have a high-quality estimator of $\theta_0(x)$, we compute $\mu_0$ by "plug-in," but that alone generally requires a bias correction for valid inference.  

### Deep Neural Network Estimation

Modern deep learning can approximate complex high-dimensional functions effectively. To maintain the structural form, we set up a "structured net":  
- Hidden layers produce $\hat{\theta}(x)$ in a "parameter layer."  
- A final "model layer" enforces global structure (like a dot product with $T$ if the model is linear in treatment, etc.).  

This yields an estimator

$$
\hat{\theta}
=
\underset{\theta(\cdot)\;\in\;F_{\mathrm{DNN}}}{\arg\min}
\frac{1}{n}
\sum_{i=1}^n
\ell\bigl(y_i,\;t_i,\;\theta(x_i)\bigr),
$$

where $F_{\mathrm{DNN}}$ is an appropriately chosen class of ReLU networks. Under standard smoothness, one obtains $L^2$ convergence rates that depend only on the dimension of the continuous part of $X$, not on $d_T$. (Intuitively, $T$ is handled "structurally" by the final layer.)  

### Inference via Orthogonal Score

We aim for valid inference on

$$
\mu_0 
=
\mathbb{E}\Bigl[\;H\bigl(X,\;\theta_0(X)\bigr)\Bigr].
$$

The key tool is an orthogonal (or "influence") function that corrects for first-stage estimation of $\theta_0(x)$. Let

- $\ell_\theta(w,\theta(x))$ be the gradient of $\ell$ wrt $\theta$,  
- $\Lambda(x) = \mathbb{E}[\;\ell_{\theta\theta}(Y,T,\theta_0(x))\mid X=x\,]$ the conditional Hessian of $\ell$ wrt $\theta$.  

Also let $H_\theta(x,\theta)$ be the Jacobian of $H$ wrt $\theta$. Then a Neyman-orthogonal score for $\mu_0$ is

$$
\psi\bigl(w,\;\theta,\;\Lambda\bigr)
=
H\bigl(x,\;\theta(x)\bigr)
-
H_\theta\bigl(x,\;\theta(x)\bigr)\,\Lambda(x)^{-1}\,\ell_\theta\bigl(w,\;\theta(x)\bigr).
$$

Because $\ell_\theta$ and $\Lambda$ follow from the structural specification $\ell(\cdot)$, this formula covers many classical models (linear, logistic, Tobit, etc.). One simply "plugs in" $(\hat{\theta},\hat{\Lambda})$ for data-based estimates.  

Why do we do this? We want an expression $\psi(w;\dots)$ whose expectation is zero when evaluated at the true parameters, and whose first-order dependence on errors in $\hat{\theta}$ is canceled out (the "orthogonality"). This implies that $\frac{1}{n}\sum \psi(w_i,\hat{\theta},\hat{\Lambda})$ is approximately unbiased for $\mu_0$, and that standard error formulas become valid under relatively mild conditions.  

### Asymptotic Normality

One obtains

$$
\sqrt{n}\,\bigl(\;\hat{\mu}\;-\;\mu_0\bigr)
\;\;\overset{d}{\longrightarrow}\;\;
\mathcal{N}(0,\;\Psi),
$$

with a consistent estimator of $\Psi$ given by the sample variance of the influence values $\psi(w_i,\hat{\theta},\hat{\Lambda})$. Here $\hat{\mu}$ is the average of $\psi_i$ plus a sample-splitting or cross-fitting scheme to mitigate overfitting bias.  

### Comments on Why These Steps Matter

- Allowing $\theta_0$ to depend on $x$ captures rich heterogeneity while preserving interpretability (we still talk about "coefficients" or "parameters," but now as functions).  
- Using deep learning in a structured way ensures that the model's global parametric form in $(Y,T)$ is respected, so the dimension of $T$ does not blow up the complexity.  
- The influence-function approach (Neyman orthogonality) is a standard semiparametric technique ensuring valid asymptotic distributions after a complex or high-dimensional first stage.  
- The matrix $\Lambda(x)$ and its inverse reflect the "local curvature" of the loss. One requires $\Lambda(x)$ to be nonsingular to identify parameters (analogous to invertible information matrices in parametric MLE).  

## Special Illustrations

1) Deep Linear Regression  
   - Mean restriction:  
     $
     \mathbb{E}[Y\mid X=x,\,T=t]=\theta_0(x)^\prime\,t.
     $  
   - Negative log-likelihood or squared-error yields a standard gradient $\ell_\theta=w$-dependent.  

2) Deep Logit Regression  
   - Binary $Y\in\{0,1\}$ with logistic link:  
     $
     \sigma\bigl[\theta_0(x)^\prime\,t\bigr],
     $
     and negative log-likelihood.  

3) Other structured QMLE: Tobit, Multinomial, Fractional outcomes, etc.  

All these fall under the same template once one writes the per-observation loss $\ell$ in terms of $\theta(x)$.

## Takeaway

By rewriting classical parametric models (logistic, linear, multinomial, etc.) so that their parameters become flexible functions $\theta_0(x)$, one obtains interpretable heterogeneity with minimal overhead. Deep nets approximate these functions effectively, and an orthogonal score formula (as above) then gives valid inference on any smooth functional of $\theta_0$. This merges the rich flexibility of modern machine learning with the clarity of structural models.
