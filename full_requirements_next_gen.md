# Deep Inference: Next Generation Requirements

**Version**: 1.0
**Date**: January 2026
**Authors**: Engineering Team
**Status**: Draft

---

## 1. Executive Summary

### 1.1 What This Document Is

This document specifies incremental improvements to the `deep-inference` package. The goal is to extend the existing Protocol Orchestra architecture with new causal estimands, targets, and estimation methods while maintaining backward compatibility with the production-ready `structural_dml()` API.

### 1.2 Why These Changes

The Farrell-Liang-Misra (FLM) framework provides valid inference for heterogeneous structural parameters learned by neural networks. The current implementation covers the core theory but lacks:

1. **Causal estimands** commonly requested by practitioners (ATE, ATT, CATE)
2. **Doubly robust estimation** for improved robustness
3. **IV/2SLS** for endogenous treatments
4. **Economic targets** (elasticity, WTP, welfare)
5. **Unified API** between legacy and new code paths

### 1.3 Scope Boundaries

| IN SCOPE | OUT OF SCOPE |
|----------|--------------|
| IV/2SLS with moment conditions | Berry models (demand systems) |
| Doubly Robust ATE (IPW + OR) | Vector treatments (d_T > 1) |
| Causal estimands: ATE, ATT, CATE | Multi-equation systems |
| New targets: Elasticity, WTP, Welfare | Bayesian uncertainty / posteriors |
| API unification | GPU/distributed training |
| Centralized t_tilde handling | Production deployment tooling |
| Diagnostic framework | |
| Type-safe tensor shapes | |

### 1.4 Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Coverage | 90-99% (95% nominal) | eval_06 + new evals |
| SE Ratio | 0.7-1.5 | Empirical / Mean estimated |
| Backward compatibility | 100% | Existing tests pass |
| API migration | Warnings only | No breaking changes in v0.3 |
| New eval pass rate | 100% | All new evals green |

---

## 2. Current State Assessment

### 2.1 What Works Well

**Evidence from evals:**

| Component | Eval | Status | Evidence |
|-----------|------|--------|----------|
| Parameter recovery | eval_01 | PASS | θ̂(x) correlates >0.95 with θ*(x) |
| Autodiff | eval_02 | PASS | Score/Hessian match calculus formulas |
| Lambda estimation | eval_03 | PASS | Λ̂(x) with ridge achieves 96% coverage |
| Target Jacobian | eval_04 | PASS | H_θ autodiff matches chain rule |
| Psi assembly | eval_05 | PASS | ψ package matches Oracle |
| Frequentist coverage | eval_06 | PASS | 95% coverage with M=50 |

**Production-ready components:**

```
src/deep_inference/
├── models/base.py          # StructuralModel protocol ✓
├── targets/base.py         # Target protocol ✓
├── lambda_/base.py         # LambdaStrategy protocol ✓
├── engine/crossfit.py      # 2-way/3-way cross-fitting ✓
├── engine/assembler.py     # Influence function assembly ✓
├── engine/variance.py      # SE/CI computation ✓
└── families/               # 8 GLM families ✓
```

### 2.2 Technical Debt

**TD-1: Dual API confusion**
- `structural_dml()` (legacy) and `inference()` (new) have overlapping functionality
- Different parameter names (`family` vs `model`, `target` vs `target_fn`)
- Result types differ (`DMLResult` vs `InferenceResult`)

**TD-2: t_tilde handling scattered**
- Hardcoded in some evals, parameter in others
- Default behavior inconsistent (some use mean(T), some use 0.0)
- No central place documenting what t_tilde means for each target

**TD-3: Diagnostics are ad-hoc**
- Each eval prints different metrics
- No structured diagnostic object
- Users can't easily check inference quality

**TD-4: No type safety for tensor shapes**
- Easy to pass (n,) when (n, 1) expected
- Silent broadcasting bugs
- No IDE autocomplete for tensor dimensions

### 2.3 Missing Paper Components

From FLM (2021, 2025) not yet implemented:

| Paper Section | Component | Status |
|---------------|-----------|--------|
| Section 4.2 | Doubly robust estimation | NOT IMPLEMENTED |
| Section 5 | IV/2SLS with moments | NOT IMPLEMENTED |
| Appendix B | General moment conditions | NOT IMPLEMENTED |
| Example 3 | Price elasticity target | NOT IMPLEMENTED |
| Example 5 | Consumer welfare | NOT IMPLEMENTED |

---

## 3. Requirements by Priority

### P0: Foundation (Must Have)

**P0-1: API Unification**
- Deprecate `structural_dml()` with warning pointing to `inference()`
- Add `family=` alias to `inference()` for backward compatibility
- Unify result types into single `InferenceResult`
- Migration path: warnings for 2 minor versions, removal in next major

**P0-2: Centralized t_tilde Handling**
- Create `TreatmentEvaluator` class
- Document t_tilde semantics per target type
- Default: `t_tilde=None` → target-specific default (0.0 for AME, mean(T) for beta)
- Validate t_tilde is in observed range with warning

**P0-3: Diagnostic Framework**
- Create `InferenceDiagnostics` dataclass
- Include: coverage proxy, SE calibration, Lambda condition, theta recovery
- Auto-populated by `inference()`
- `.summary()` method for human-readable report

### P1: Causal Estimands (Should Have)

**P1-1: Doubly Robust ATE**
- Implement `DoublyRobustATE` class using IPW + outcome regression
- Require propensity model specification
- Cross-fit both propensity and outcome models
- Key formula: `ψ_DR = μ₁(X) - μ₀(X) + T(Y-μ₁(X))/e(X) - (1-T)(Y-μ₀(X))/(1-e(X))`

**P1-2: ATT (Average Treatment effect on Treated)**
- Implement `ATT` target
- Formula: `E[Y(1) - Y(0) | T=1]`
- Requires propensity weighting for untreated counterfactual

**P1-3: CATE (Conditional Average Treatment Effect)**
- Implement `CATE` wrapper
- Returns τ(x) = E[Y(1) - Y(0) | X=x] for each observation
- Support aggregation methods: mean, quantiles, subgroup

**P1-4: New Economic Targets**
- `Elasticity`: `H(x, θ, t) = (∂log(μ)/∂log(t)) = t·β/μ` for log-link models
- `WTP`: Willingness to pay from discrete choice
- `ConsumerWelfare`: Aggregate surplus measure

### P2: Advanced (Nice to Have)

**P2-1: IV/2SLS**
- Implement `MomentModel` protocol for general moment conditions
- Implement `IV2SLS` class for linear IV with instruments Z
- GMM objective: `min_θ g(θ)'W g(θ)` where `g(θ) = E[Z(Y - Xθ)]`
- Cross-fit both first stage and structural equation

**P2-2: Implicit Targets**
- Support targets defined implicitly via `G(μ*, θ) = 0`
- Autodiff through implicit function theorem
- Use case: equilibrium prices, fixed points

---

## 4. Technical Specifications

### 4.1 IV/2SLS Specification

**New Protocol: `MomentModel`**

```python
@runtime_checkable
class MomentModel(Protocol):
    """
    Protocol for moment-based estimation (GMM/IV).

    A moment model defines:
    - n_moments: Number of moment conditions
    - theta_dim: Dimension of parameters
    - moments(): Moment function g(Y, T, Z, X; θ)

    For IV/2SLS: moments = Z ⊗ (Y - T'θ)
    """

    n_moments: int
    """Number of moment conditions (dim of Z × dim of residual)."""

    theta_dim: int
    """Dimension of parameter vector θ."""

    def moments(
        self,
        y: Tensor,      # (n,) outcomes
        t: Tensor,      # (n, d_t) endogenous
        z: Tensor,      # (n, d_z) instruments
        x: Tensor,      # (n, d_x) exogenous
        theta: Tensor   # (n, d_theta) parameters
    ) -> Tensor:
        """
        Compute moment conditions g(Y, T, Z, X; θ).

        Returns:
            (n, n_moments) moment values
        """
        ...

    def moment_jacobian(
        self,
        y: Tensor,
        t: Tensor,
        z: Tensor,
        x: Tensor,
        theta: Tensor
    ) -> Optional[Tensor]:
        """
        Jacobian of moments w.r.t. theta (optional, autodiff fallback).

        Returns:
            (n, n_moments, d_theta) or None
        """
        ...
```

**New Class: `IV2SLS`**

```python
@dataclass
class IV2SLS:
    """
    Two-Stage Least Squares with heterogeneous effects.

    Stage 1: T̂ = E[T | Z, X] (fitted values)
    Stage 2: Y = θ(X)'T̂ + ε

    Influence function corrects for:
    1. First-stage estimation error
    2. Neural network regularization
    """

    # Specification
    endog_dim: int = 1  # Dimension of endogenous T
    instrument_dim: int = 1  # Dimension of instruments Z

    # First stage model
    first_stage: Literal["linear", "neural"] = "neural"
    first_stage_hidden_dims: List[int] = field(default_factory=lambda: [64, 32])

    # GMM settings
    weight_matrix: Literal["identity", "efficient"] = "identity"

    def fit(
        self,
        Y: Tensor,      # (n,) outcomes
        T: Tensor,      # (n, d_t) endogenous
        Z: Tensor,      # (n, d_z) instruments
        X: Tensor,      # (n, d_x) covariates
        **kwargs
    ) -> "IVResult":
        """
        Fit IV model with cross-fitting.

        Returns IVResult with theta_hat, se, ci, diagnostics including:
        - first_stage_F: First-stage F-statistic
        - overid_test: Hansen J-test if overidentified
        """
        ...
```

**GMM Objective Details:**

For moments `g_i(θ) = Z_i ⊗ (Y_i - T_i'θ(X_i))`:

1. **Sample moments**: `ĝ(θ) = (1/n) Σᵢ gᵢ(θ)`
2. **Weighting matrix**: `W = I` (identity) or `Ŵ = (ĝĝ')⁻¹` (efficient)
3. **Objective**: `Q(θ) = ĝ(θ)'W ĝ(θ)`
4. **Influence function**: Standard GMM IF with neural network correction

**Cross-Fitting for IV:**

- 3-way split required: first-stage, second-stage, evaluation
- First stage fit on fold A: `π̂_A` such that `T̂ = Z'π̂`
- Second stage fit on fold B using `T̂_B = Z_B'π̂_A`
- Evaluate on fold C

### 4.2 Doubly Robust ATE Specification

**New Protocol: `PropensityEstimator`**

```python
@runtime_checkable
class PropensityEstimator(Protocol):
    """
    Protocol for propensity score estimation.

    Estimates e(x) = P(T=1 | X=x) for binary treatment.
    """

    def fit(self, T: Tensor, X: Tensor) -> Self:
        """
        Fit propensity model.

        Args:
            T: (n,) binary treatment
            X: (n, d_x) covariates

        Returns:
            Self for chaining
        """
        ...

    def predict(self, X: Tensor) -> Tensor:
        """
        Predict propensity scores.

        Args:
            X: (n, d_x) covariates

        Returns:
            (n,) propensity scores e(x)
        """
        ...

    def predict_bounded(
        self,
        X: Tensor,
        clip_min: float = 0.01,
        clip_max: float = 0.99
    ) -> Tensor:
        """
        Predict with probability clipping for stability.

        Args:
            X: (n, d_x) covariates
            clip_min: Lower bound
            clip_max: Upper bound

        Returns:
            (n,) clipped propensity scores
        """
        ...
```

**New Class: `DoublyRobustATE`**

```python
@dataclass
class DoublyRobustATE:
    """
    Doubly Robust Average Treatment Effect estimator.

    Combines IPW and outcome regression:

    ψ_DR(Z) = μ̂₁(X) - μ̂₀(X)
            + T·(Y - μ̂₁(X))/ê(X)
            - (1-T)·(Y - μ̂₀(X))/(1 - ê(X))

    Doubly robust: consistent if EITHER propensity OR outcome model correct.

    References:
    - Kennedy (2023): "Semiparametric doubly robust targeted double machine learning"
    - Chernozhukov et al. (2018): "Double/debiased machine learning"
    """

    # Propensity model
    propensity_estimator: PropensityEstimator = field(
        default_factory=lambda: NeuralPropensity()
    )
    propensity_clip: Tuple[float, float] = (0.01, 0.99)

    # Outcome model
    outcome_model: Literal["linear", "logit", "neural"] = "neural"
    outcome_hidden_dims: List[int] = field(default_factory=lambda: [64, 32])

    # Cross-fitting
    n_folds: int = 5

    def fit(
        self,
        Y: Tensor,
        T: Tensor,  # Binary {0, 1}
        X: Tensor,
        **kwargs
    ) -> "DRATEResult":
        """
        Fit doubly robust ATE.

        Cross-fitting procedure:
        1. For each fold k:
           - Fit propensity ê on other folds
           - Fit μ̂₀, μ̂₁ on other folds
           - Compute ψ_DR on fold k
        2. Average ψ values

        Returns:
            DRATEResult with:
            - ate: Point estimate
            - se: Standard error
            - ci: Confidence interval
            - diagnostics: Propensity overlap, outcome fit quality
        """
        ...
```

**DR Influence Function Assembly:**

```python
def compute_dr_psi(
    Y: Tensor,           # (n,) outcomes
    T: Tensor,           # (n,) binary treatment
    mu_0: Tensor,        # (n,) E[Y|T=0,X]
    mu_1: Tensor,        # (n,) E[Y|T=1,X]
    e_hat: Tensor,       # (n,) propensity scores
    clip: Tuple = (0.01, 0.99)
) -> Tensor:
    """
    Compute doubly robust influence function.

    ψ_DR = (μ̂₁ - μ̂₀) + T(Y - μ̂₁)/ê - (1-T)(Y - μ̂₀)/(1-ê)

    Returns:
        (n,) ψ values
    """
    e_clipped = e_hat.clamp(clip[0], clip[1])

    # Outcome regression component
    or_component = mu_1 - mu_0

    # IPW correction for treated
    ipw_treated = T * (Y - mu_1) / e_clipped

    # IPW correction for control
    ipw_control = (1 - T) * (Y - mu_0) / (1 - e_clipped)

    return or_component + ipw_treated - ipw_control
```

### 4.3 Causal Estimands Specification

**4.3.1 ATE (Average Treatment Effect)**

```python
@dataclass
class ATE:
    """
    Average Treatment Effect: E[Y(1) - Y(0)]

    For binary treatment, this is the population-level causal effect.
    Requires unconfoundedness (no unmeasured confounders).
    """

    estimator: Literal["outcome_regression", "ipw", "doubly_robust"] = "doubly_robust"

    def __call__(
        self,
        Y: np.ndarray,
        T: np.ndarray,  # Binary
        X: np.ndarray,
        **kwargs
    ) -> InferenceResult:
        """Estimate ATE with valid inference."""
        ...
```

**4.3.2 ATT (Average Treatment effect on Treated)**

```python
@dataclass
class ATT:
    """
    Average Treatment Effect on the Treated: E[Y(1) - Y(0) | T=1]

    The effect for those who actually received treatment.
    Often more policy-relevant than ATE when treatment targeting exists.

    Influence function:
    ψ_ATT = T(Y - μ₀(X))/p - (1-T)e(X)(Y - μ₀(X))/((1-e(X))p)

    where p = P(T=1) and e(X) = P(T=1|X).
    """

    def target_h(
        self,
        x: Tensor,
        theta: Tensor,  # (mu_0, mu_1, e)
        t_tilde: Tensor  # Not used, but required by protocol
    ) -> Tensor:
        """H(x, θ) = μ₁(x) - μ₀(x) weighted by treated population."""
        mu_0, mu_1 = theta[0], theta[1]
        return mu_1 - mu_0
```

**4.3.3 CATE (Conditional Average Treatment Effect)**

```python
@dataclass
class CATE:
    """
    Conditional Average Treatment Effect: τ(x) = E[Y(1) - Y(0) | X=x]

    Returns heterogeneous effects for each observation.
    Can be aggregated to get ATE, ATT, or subgroup effects.

    This is a wrapper that:
    1. Uses existing `inference()` infrastructure
    2. Returns τ̂(x) for all x in dataset
    3. Provides uncertainty for individual τ̂(x) values
    """

    base_model: Literal["linear", "logit", "neural"] = "neural"

    def fit(
        self,
        Y: np.ndarray,
        T: np.ndarray,
        X: np.ndarray,
        **kwargs
    ) -> "CATEResult":
        """
        Fit CATE model.

        Returns CATEResult with:
        - tau_hat: (n,) individual treatment effects
        - tau_se: (n,) standard errors for each τ̂(x)
        - predict(X_new): Method to predict τ for new X
        """
        ...

    def aggregate(
        self,
        group_fn: Optional[Callable[[Tensor], Tensor]] = None
    ) -> InferenceResult:
        """
        Aggregate CATE to get ATE or subgroup ATE.

        Args:
            group_fn: Function mapping X to group indicator
                      If None, returns overall ATE

        Returns:
            InferenceResult with aggregated effect and SE
        """
        ...
```

### 4.4 New Targets Specification

**4.4.1 Elasticity**

```python
@dataclass
class Elasticity(BaseTarget):
    """
    Price elasticity of demand.

    For log-link models (Poisson, NegBin, Gamma):
        η = ∂log(Q)/∂log(P) = P·(∂Q/∂P)/Q = β·P/μ(P)

    For logit (probability):
        η = ∂log(P(Y=1))/∂log(T) = (1-p)·β·T

    Requires t_tilde to specify evaluation price point.
    """

    output_dim: int = 1
    model_type: Literal["logit", "poisson", "gamma", "negbin"] = "logit"

    def h(self, x: Tensor, theta: Tensor, t_tilde: Tensor) -> Tensor:
        """
        Compute elasticity at evaluation point.

        Args:
            x: Covariates (d_x,)
            theta: (alpha, beta) parameters
            t_tilde: Price point for evaluation

        Returns:
            Scalar elasticity value
        """
        alpha, beta = theta[0], theta[1]

        if self.model_type == "logit":
            # η = (1-σ(η))·β·t where η = α + β·t
            eta = alpha + beta * t_tilde
            p = torch.sigmoid(eta)
            return (1 - p) * beta * t_tilde

        elif self.model_type in ["poisson", "gamma", "negbin"]:
            # η = β·t (for log-link)
            return beta * t_tilde
```

**4.4.2 Willingness To Pay (WTP)**

```python
@dataclass
class WTP(BaseTarget):
    """
    Willingness To Pay from discrete choice model.

    WTP = -β_attribute / β_price

    Measures how much price can increase while maintaining
    same probability of purchase when attribute improves by 1 unit.

    Requires specifying which theta indices are attribute vs price.
    """

    output_dim: int = 1
    attribute_index: int = 1  # Index of attribute coefficient
    price_index: int = 2      # Index of price coefficient

    def h(self, x: Tensor, theta: Tensor, t_tilde: Tensor) -> Tensor:
        """
        Compute WTP = -θ_attribute / θ_price.

        Delta method SE automatically via autodiff jacobian.
        """
        beta_attr = theta[self.attribute_index]
        beta_price = theta[self.price_index]

        # Avoid division by zero
        if torch.abs(beta_price) < 1e-8:
            return torch.tensor(float('nan'))

        return -beta_attr / beta_price
```

**4.4.3 Consumer Welfare**

```python
@dataclass
class ConsumerWelfare(BaseTarget):
    """
    Expected Consumer Surplus from logit demand.

    For logit: CS = E[max_j V_j] / β_price = log(Σexp(V_j)) / |β_price|

    Simplified for binary choice:
    CS = log(1 + exp(α + β·t)) / |β_price|

    See Small & Rosen (1981) for derivation.
    """

    output_dim: int = 1
    price_coef_index: int = 1  # Index of price coefficient in theta

    def h(self, x: Tensor, theta: Tensor, t_tilde: Tensor) -> Tensor:
        """
        Compute expected consumer surplus.

        For binary logit:
        CS = log(1 + exp(V)) / |β_price|

        where V = α + β·t is the utility of the good.
        """
        alpha = theta[0]
        beta_price = theta[self.price_coef_index]

        # Utility at evaluation point
        V = alpha + beta_price * t_tilde

        # Inclusive value (logsum)
        inclusive_value = torch.log(1 + torch.exp(V))

        # Scale by price coefficient to convert to dollars
        return inclusive_value / torch.abs(beta_price)
```

### 4.5 Architectural Improvements

**4.5.1 Unified API**

```python
# src/deep_inference/__init__.py

def inference(
    Y: np.ndarray,
    T: np.ndarray,
    X: np.ndarray,
    # Model specification (multiple ways to specify)
    model: Optional[str] = None,        # "linear", "logit", "poisson", etc.
    family: Optional[str] = None,       # ALIAS for model (backward compat)
    loss: Optional[Callable] = None,    # Custom loss function
    # Target specification
    target: Optional[str] = None,       # "beta", "ame", "elasticity", etc.
    target_fn: Optional[Callable] = None,
    # Treatment evaluation
    t_tilde: Optional[float] = None,    # Default: target-specific
    # ... rest unchanged
) -> InferenceResult:
    """
    Unified entry point for structural inference.

    Deprecation: `structural_dml()` is deprecated. Use `inference()` instead.
    Migration: `family=` is accepted as alias for `model=`.
    """
    # Handle backward compatibility
    if family is not None:
        if model is not None:
            raise ValueError("Cannot specify both 'model' and 'family'")
        warnings.warn(
            "family= is deprecated, use model= instead",
            DeprecationWarning
        )
        model = family

    # ... implementation
```

**4.5.2 TreatmentEvaluator**

```python
# src/deep_inference/utils/treatment.py

@dataclass
class TreatmentEvaluator:
    """
    Centralized handling of treatment evaluation point t_tilde.

    Responsibilities:
    1. Determine default t_tilde based on target
    2. Validate t_tilde is in reasonable range
    3. Warn if extrapolating beyond data
    """

    T: Tensor  # Observed treatments
    target_type: str  # "beta", "ame", "elasticity", etc.

    def get_default(self) -> Tensor:
        """
        Get target-appropriate default t_tilde.

        Defaults:
        - "beta": mean(T) - evaluating average log-odds
        - "ame": 0.0 - marginal effect at zero
        - "elasticity": median(T) - elasticity at median price
        - "wtp": mean(T) - WTP at average price
        """
        defaults = {
            "beta": self.T.mean(),
            "ame": torch.tensor(0.0),
            "elasticity": self.T.median(),
            "wtp": self.T.mean(),
            "welfare": self.T.mean(),
        }
        return defaults.get(self.target_type, self.T.mean())

    def validate(self, t_tilde: Tensor) -> None:
        """
        Validate t_tilde is in data range.

        Warnings:
        - If t_tilde outside [min(T), max(T)]: extrapolation warning
        - If t_tilde far from any observed T: sparse data warning
        """
        t_min, t_max = self.T.min(), self.T.max()

        if t_tilde < t_min or t_tilde > t_max:
            warnings.warn(
                f"t_tilde={t_tilde:.3f} is outside observed range "
                f"[{t_min:.3f}, {t_max:.3f}]. Extrapolation may be unreliable.",
                UserWarning
            )

        # Check if t_tilde is near any observation
        distances = torch.abs(self.T - t_tilde)
        if distances.min() > 0.1 * (t_max - t_min):
            warnings.warn(
                f"t_tilde={t_tilde:.3f} is far from nearest observation. "
                f"Inference may be imprecise.",
                UserWarning
            )
```

**4.5.3 InferenceDiagnostics**

```python
# src/deep_inference/diagnostics.py

@dataclass
class InferenceDiagnostics:
    """
    Comprehensive diagnostics for inference quality.

    Categories:
    1. Theta recovery: How well does θ̂(x) approximate θ*(x)?
    2. Lambda stability: Is Λ(x) well-conditioned?
    3. SE calibration: Does estimated SE match empirical variance?
    4. Influence function: Is IF correction reasonable?
    """

    # Theta recovery
    theta_corr_alpha: Optional[float] = None  # Corr(α̂, α*) if oracle known
    theta_corr_beta: Optional[float] = None   # Corr(β̂, β*) if oracle known
    theta_rmse: Optional[float] = None

    # Lambda stability
    min_lambda_eigenvalue: float = 0.0
    max_lambda_condition: float = float('inf')
    n_regularized: int = 0
    lambda_method: str = "ridge"

    # SE calibration (only available with multiple runs)
    se_ratio: Optional[float] = None  # Empirical SE / Mean estimated SE

    # Influence function
    correction_ratio: float = 0.0  # |IF correction| / |naive estimate|
    psi_mean: float = 0.0
    psi_std: float = 0.0

    # Cross-fitting
    n_folds: int = 0
    regime: str = "C"  # "A", "B", or "C"

    # Warnings
    warnings: List[str] = field(default_factory=list)

    def check(self) -> bool:
        """
        Run diagnostic checks, populate warnings.

        Returns True if all checks pass.
        """
        self.warnings = []
        passed = True

        # Check Lambda stability
        if self.min_lambda_eigenvalue < 1e-6:
            self.warnings.append(
                f"Very small Lambda eigenvalue ({self.min_lambda_eigenvalue:.2e}). "
                f"Consider increasing ridge regularization."
            )
            passed = False

        if self.max_lambda_condition > 1e6:
            self.warnings.append(
                f"Ill-conditioned Lambda (condition={self.max_lambda_condition:.2e}). "
                f"SE may be unreliable."
            )
            passed = False

        # Check regularization rate
        if self.n_regularized > 0:
            rate = self.n_regularized / self.n_folds if self.n_folds > 0 else 0
            if rate > 0.1:
                self.warnings.append(
                    f"High regularization rate ({rate*100:.1f}%). "
                    f"Consider more data or simpler model."
                )

        # Check IF correction magnitude
        if self.correction_ratio > 5.0:
            self.warnings.append(
                f"Large IF correction ({self.correction_ratio:.1f}x). "
                f"Naive estimate may be severely biased."
            )

        # Check SE ratio if available
        if self.se_ratio is not None:
            if self.se_ratio < 0.5 or self.se_ratio > 2.0:
                self.warnings.append(
                    f"SE ratio out of range ({self.se_ratio:.2f}). "
                    f"Consider more folds (K >= 50)."
                )
                passed = False

        return passed

    def summary(self) -> str:
        """Human-readable diagnostic summary."""
        lines = [
            "=" * 60,
            "INFERENCE DIAGNOSTICS",
            "=" * 60,
            "",
            f"Regime: {self.regime}",
            f"Folds: {self.n_folds}",
            f"Lambda method: {self.lambda_method}",
            "",
            "--- Lambda Stability ---",
            f"  Min eigenvalue: {self.min_lambda_eigenvalue:.6f}",
            f"  Max condition:  {self.max_lambda_condition:.2f}",
            f"  Regularized:    {self.n_regularized}",
            "",
            "--- Influence Function ---",
            f"  Correction ratio: {self.correction_ratio:.4f}",
            f"  ψ mean: {self.psi_mean:.6f}",
            f"  ψ std:  {self.psi_std:.6f}",
        ]

        if self.theta_corr_beta is not None:
            lines.extend([
                "",
                "--- Theta Recovery ---",
                f"  Corr(α̂, α*): {self.theta_corr_alpha:.4f}" if self.theta_corr_alpha else "",
                f"  Corr(β̂, β*): {self.theta_corr_beta:.4f}",
            ])

        if self.se_ratio is not None:
            lines.extend([
                "",
                "--- SE Calibration ---",
                f"  SE ratio: {self.se_ratio:.4f}",
            ])

        if self.warnings:
            lines.extend([
                "",
                "--- Warnings ---",
            ])
            for w in self.warnings:
                lines.append(f"  ⚠️ {w}")
        else:
            lines.extend([
                "",
                "--- Status ---",
                "  ✓ All checks passed",
            ])

        lines.append("=" * 60)
        return "\n".join(lines)
```

**4.5.4 Type-Safe Shapes (jaxtyping)**

```python
# src/deep_inference/types.py

from typing import TypeVar, Generic
from jaxtyping import Float, Int
from torch import Tensor

# Dimension type variables
N = TypeVar('N')  # Number of observations
DX = TypeVar('DX')  # Covariate dimension
DT = TypeVar('DT')  # Treatment dimension
DTheta = TypeVar('DTheta')  # Parameter dimension

# Type aliases for common shapes
Outcomes = Float[Tensor, "n"]
Treatments = Float[Tensor, "n"]
Covariates = Float[Tensor, "n d_x"]
Parameters = Float[Tensor, "n d_theta"]
LambdaMatrices = Float[Tensor, "n d_theta d_theta"]
PsiValues = Float[Tensor, "n"]

# Function signatures with shape checking
def example_typed_function(
    Y: Float[Tensor, "n"],
    T: Float[Tensor, "n"],
    X: Float[Tensor, "n d_x"],
    theta: Float[Tensor, "n d_theta"],
) -> Float[Tensor, "n"]:
    """Example function with jaxtyping annotations."""
    ...

# Runtime shape validation decorator
def validate_shapes(func):
    """
    Decorator that validates tensor shapes at runtime.

    Usage:
        @validate_shapes
        def my_func(Y: Float[Tensor, "n"], X: Float[Tensor, "n d"]):
            ...
    """
    from functools import wraps
    from jaxtyping import jaxtyped
    from beartype import beartype

    return jaxtyped(beartype(func))
```

---

## 5. API Design

### 5.1 Usage Examples

**Example 1: Basic ATE with Doubly Robust Estimation**

```python
from deep_inference import inference
from deep_inference.estimands import DoublyRobustATE

# Binary treatment, continuous outcome
Y = np.random.randn(1000)
T = np.random.binomial(1, 0.5, 1000)
X = np.random.randn(1000, 5)

# Simple API
result = inference(Y, T, X, model="linear", target="ate")
print(f"ATE: {result.mu_hat:.3f} ± {result.se:.3f}")

# Full control API
dr_ate = DoublyRobustATE(
    propensity_clip=(0.05, 0.95),
    outcome_model="neural",
    n_folds=20
)
result = dr_ate.fit(Y, T, X)
print(result.summary())
```

**Example 2: Heterogeneous Treatment Effects (CATE)**

```python
from deep_inference.estimands import CATE

# Fit CATE model
cate = CATE(base_model="neural")
result = cate.fit(Y, T, X, epochs=100)

# Individual effects
print(f"τ(x) range: [{result.tau_hat.min():.3f}, {result.tau_hat.max():.3f}]")

# Aggregate to ATE
ate_result = result.aggregate()
print(f"ATE: {ate_result.mu_hat:.3f}")

# Subgroup effects (e.g., by X[:, 0] > 0)
def high_x0(X):
    return X[:, 0] > 0

subgroup_result = result.aggregate(group_fn=high_x0)
print(f"ATE for X[0] > 0: {subgroup_result.mu_hat:.3f}")
```

**Example 3: IV/2SLS with Endogenous Treatment**

```python
from deep_inference.estimands import IV2SLS

# Endogenous treatment with instrument
Y = outcome  # (n,)
T = price    # (n,) endogenous
Z = cost     # (n,) instrument (supply shifter)
X = controls # (n, d_x)

# Fit IV model
iv = IV2SLS(first_stage="neural")
result = iv.fit(Y, T, Z, X)

print(f"β (IV): {result.mu_hat:.3f} ± {result.se:.3f}")
print(f"First-stage F: {result.diagnostics['first_stage_F']:.1f}")
```

**Example 4: Price Elasticity**

```python
from deep_inference import inference

# Demand model: P(purchase) = σ(α(x) + β(x)·price)
Y = purchased  # Binary
T = price      # Continuous
X = demographics

result = inference(
    Y, T, X,
    model="logit",
    target="elasticity",
    t_tilde=median_price,  # Evaluate at median price
)

print(f"Price elasticity at p={median_price}: {result.mu_hat:.3f}")
# Interpretation: 1% price increase → {result.mu_hat}% change in purchase prob
```

**Example 5: Consumer Welfare**

```python
from deep_inference import inference
from deep_inference.targets import ConsumerWelfare

# Fit demand model
result = inference(
    Y, T, X,
    model="logit",
    target=ConsumerWelfare(price_coef_index=1),
    t_tilde=current_price,
)

print(f"Consumer surplus: ${result.mu_hat:.2f}")

# Counterfactual: welfare change from price increase
result_high = inference(Y, T, X, model="logit",
                        target=ConsumerWelfare(), t_tilde=current_price * 1.1)
welfare_change = result_high.mu_hat - result.mu_hat
print(f"Welfare loss from 10% price increase: ${-welfare_change:.2f}")
```

### 5.2 Protocol Signatures Summary

```python
# Models
class StructuralModel(Protocol):
    theta_dim: int
    hessian_depends_on_theta: bool
    hessian_depends_on_y: bool
    def loss(y, t, theta) -> Tensor: ...
    def score(y, t, theta) -> Optional[Tensor]: ...
    def hessian(y, t, theta) -> Optional[Tensor]: ...

# Targets
class Target(Protocol):
    output_dim: int
    def h(x, theta, t_tilde) -> Tensor: ...
    def jacobian(x, theta, t_tilde) -> Optional[Tensor]: ...

# Lambda
class LambdaStrategy(Protocol):
    requires_theta: bool
    requires_separate_fold: bool
    def fit(X, T, Y, theta_hat, model) -> None: ...
    def predict(X, theta_hat) -> Tensor: ...

# NEW: Moments (for IV)
class MomentModel(Protocol):
    n_moments: int
    theta_dim: int
    def moments(y, t, z, x, theta) -> Tensor: ...
    def moment_jacobian(...) -> Optional[Tensor]: ...

# NEW: Propensity
class PropensityEstimator(Protocol):
    def fit(T, X) -> Self: ...
    def predict(X) -> Tensor: ...
    def predict_bounded(X, clip_min, clip_max) -> Tensor: ...
```

---

## 6. Implementation Roadmap

### Phase 1: Foundation (Weeks 1-2)

**Week 1: API Unification**
- [ ] Add `family=` alias to `inference()` with deprecation warning
- [ ] Create unified `InferenceResult` replacing both `DMLResult` and old `InferenceResult`
- [ ] Add deprecation warning to `structural_dml()`
- [ ] Update all internal calls to use `inference()`
- [ ] Tests: All existing tests pass

**Week 2: Treatment & Diagnostics**
- [ ] Implement `TreatmentEvaluator` class
- [ ] Add t_tilde validation and warnings
- [ ] Implement `InferenceDiagnostics` dataclass
- [ ] Auto-populate diagnostics in `inference()`
- [ ] Add `.diagnostics.summary()` method

**Deliverables:**
- Unified API with backward compatibility
- Centralized t_tilde handling
- Comprehensive diagnostics

### Phase 2: Causal Estimands (Weeks 3-4)

**Week 3: Doubly Robust ATE**
- [ ] Implement `PropensityEstimator` protocol
- [ ] Implement `NeuralPropensity` class
- [ ] Implement `DoublyRobustATE` class
- [ ] Add propensity cross-fitting
- [ ] Implement `compute_dr_psi()`

**Week 4: ATT & CATE**
- [ ] Implement `ATT` target
- [ ] Implement `CATE` wrapper
- [ ] Add `aggregate()` method to CATE
- [ ] Add `predict()` for new observations
- [ ] Tests: Coverage for DR, ATT, CATE

**Deliverables:**
- `DoublyRobustATE`, `ATT`, `CATE` classes
- eval_09_dr_ate.py passing

### Phase 3: New Targets (Week 5)

**Week 5: Economic Targets**
- [ ] Implement `Elasticity` target
- [ ] Implement `WTP` target
- [ ] Implement `ConsumerWelfare` target
- [ ] Add target to string registry ("elasticity", "wtp", "welfare")
- [ ] Tests: Target Jacobians match autodiff

**Deliverables:**
- Three new economic targets
- Updated target registry

### Phase 4: IV/2SLS (Weeks 6-7)

**Week 6: Moment Infrastructure**
- [ ] Implement `MomentModel` protocol
- [ ] Implement `LinearIVMoments` class
- [ ] Implement GMM objective and optimizer
- [ ] Add weight matrix options

**Week 7: IV2SLS Class**
- [ ] Implement `IV2SLS` class
- [ ] Implement 3-way cross-fitting for IV
- [ ] Add first-stage diagnostics (F-stat)
- [ ] Add over-identification test (if applicable)
- [ ] Tests: eval_10_iv.py

**Deliverables:**
- `MomentModel`, `IV2SLS` classes
- eval_10_iv.py passing

### Phase 5: Polish & Documentation (Week 8)

**Week 8: Finalization**
- [ ] Type annotations with jaxtyping
- [ ] Update CLAUDE.md with new features
- [ ] Update tutorials
- [ ] Performance benchmarks
- [ ] Release notes

**Deliverables:**
- v0.3.0 release candidate
- Updated documentation

---

## 7. Testing Strategy

### 7.1 Eval Evolution

**Existing Evals (maintain):**

| Eval | Tests | Changes |
|------|-------|---------|
| eval_01_theta | θ̂ recovery | Add CATE theta recovery |
| eval_02_autodiff | Score/Hessian | Add propensity gradients |
| eval_03_lambda | Λ estimation | No change |
| eval_04_jacobian | Target H_θ | Add Elasticity, WTP, Welfare Jacobians |
| eval_05_psi | ψ assembly | Add DR-ATE ψ |
| eval_06_coverage | MC coverage | Add DR-ATE, ATT coverage |
| eval_07_e2e | End-to-end | Add new estimands |
| eval_08_regularization | Lambda stability | No change |

### 7.2 New Evals

**eval_09_dr_ate.py: Doubly Robust Coverage**

```python
"""
Eval 09: Doubly Robust ATE Coverage

Goal: Validate DR estimator achieves nominal coverage.

DGP:
    X ~ N(0, I_5)
    e(X) = σ(0.5 + 0.3*X[0])  # True propensity
    T ~ Bernoulli(e(X))
    Y(0) = X[0] + X[1] + ε
    Y(1) = Y(0) + τ(X)
    τ(X) = 1 + 0.5*X[0]  # Heterogeneous effect
    Y = T*Y(1) + (1-T)*Y(0)

True ATE = E[τ(X)] = 1.0

Metrics:
    - Coverage in [90%, 99%]
    - SE ratio in [0.7, 1.5]
    - |Bias| < 0.1

Test robustness:
    1. Correct propensity, correct outcome → Valid
    2. Wrong propensity, correct outcome → Valid (DR property)
    3. Correct propensity, wrong outcome → Valid (DR property)
    4. Wrong propensity, wrong outcome → Invalid (expected)
"""

def run_eval_09(M: int = 50, n: int = 2000):
    # Test all four scenarios
    for propensity_correct, outcome_correct in product([True, False], repeat=2):
        results = run_simulations(
            M=M, n=n,
            propensity_model="neural" if propensity_correct else "constant",
            outcome_model="neural" if outcome_correct else "constant"
        )

        metrics = compute_coverage_metrics(results, ate_true=1.0)

        # Should achieve coverage if at least one is correct
        expected_valid = propensity_correct or outcome_correct
        if expected_valid:
            assert 0.90 <= metrics["coverage"] <= 0.99
        else:
            # Both wrong: coverage may fail (this is expected)
            pass
```

**eval_10_iv.py: IV/2SLS Coverage**

```python
"""
Eval 10: IV/2SLS Coverage

Goal: Validate IV estimator with endogenous treatment.

DGP (simultaneous equations):
    X ~ N(0, I_3)              # Exogenous covariates
    Z ~ N(0, 1)                # Instrument (excluded)
    U ~ N(0, 1)                # Unobserved confounder

    T = γ*Z + δ*X[0] + U + ε_T   # First stage (Z→T, U confounds)
    Y = α + β*T + X[0] + U + ε_Y # Structural (β is causal effect)

    Parameters:
        γ = 0.5  # Instrument strength
        β = 1.0  # True causal effect

    OLS is biased (Cov(T, U) ≠ 0)
    IV is consistent (Cov(Z, U) = 0)

Metrics:
    - Coverage in [90%, 99%]
    - First-stage F > 10 (strong instrument)
    - OLS vs IV bias comparison

Tests:
    1. Strong instrument (γ=0.5): Valid
    2. Weak instrument (γ=0.1): Warning but still consistent
    3. Invalid instrument (Z→Y direct): Should fail
"""

def run_eval_10(M: int = 50, n: int = 2000):
    # Strong instrument test
    results = run_iv_simulations(
        M=M, n=n,
        gamma=0.5,  # Strong
        beta_true=1.0
    )

    metrics = compute_coverage_metrics(results, beta_true=1.0)

    assert 0.90 <= metrics["coverage"] <= 0.99
    assert metrics["mean_first_stage_F"] > 10

    # Compare to OLS
    ols_bias = abs(metrics["ols_mean_beta"] - 1.0)
    iv_bias = abs(metrics["iv_mean_beta"] - 1.0)
    assert iv_bias < ols_bias  # IV should be less biased
```

### 7.3 Eval Hook Contract

Every eval MUST implement these hooks:

```python
def run_eval_XX(
    M: int = 50,           # Monte Carlo replications
    n: int = 2000,         # Sample size
    n_folds: int = 20,     # Cross-fitting folds
    epochs: int = 100,     # Training epochs
    **kwargs
) -> Dict:
    """
    Hook 1: Standard signature for run_all.py integration.

    Returns dict with:
        - "metrics": Aggregated statistics
        - "results": List of per-simulation results
        - "passed": bool
    """
    ...

def compute_metrics(results: List, true_value: float) -> Dict:
    """
    Hook 2: Standardized metric computation.

    Must include:
        - coverage: float
        - se_ratio: float
        - bias: float
        - n_valid: int
        - n_failed: int
    """
    ...

VALIDATION_CRITERIA = {
    "Coverage in [90%, 99%]": lambda m: 0.90 <= m["coverage"] <= 0.99,
    "SE Ratio in [0.7, 1.5]": lambda m: 0.7 <= m["se_ratio"] <= 1.5,
    "|Bias| < 0.1": lambda m: abs(m["bias"]) < 0.1,
}
"""
Hook 3: Validation criteria as dict of lambdas.
"""

def print_results(metrics: Dict, criteria: Dict) -> bool:
    """
    Hook 4: Standardized result printing.

    Must print:
    - Simulation summary
    - Point estimation stats
    - SE stats
    - Coverage
    - Pass/fail for each criterion
    """
    ...

if __name__ == "__main__":
    """
    Hook 5: CLI entry point.

    Supports:
        python -m evals.eval_XX
        python -m evals.eval_XX --quick  (M=10, n=500)
    """
    ...
```

---

## 8. Migration Guide

### 8.1 Breaking Changes

**v0.3.0 (this release): No breaking changes**
- `structural_dml()` deprecated with warning, still works
- `family=` accepted as alias for `model=`
- All existing code continues to work

**v0.4.0 (planned): Soft breaking**
- `structural_dml()` removed
- `family=` alias removed
- Users must migrate to `inference(model=...)`

### 8.2 Deprecation Timeline

| Version | `structural_dml()` | `family=` alias | `DMLResult` |
|---------|-------------------|-----------------|-------------|
| v0.2.x (current) | ✓ Works | N/A | ✓ Returned |
| v0.3.0 | ⚠️ Warning | ⚠️ Warning | ⚠️ Warning |
| v0.4.0 | ❌ Removed | ❌ Removed | ❌ Removed |

### 8.3 Migration Examples

**Before (v0.2.x):**
```python
from deep_inference import structural_dml

result = structural_dml(
    Y, T, X,
    family='logit',
    target='ame',
    n_folds=50,
    epochs=100
)

# Access results
print(result.mu_hat)  # DMLResult
```

**After (v0.3.0+):**
```python
from deep_inference import inference

result = inference(
    Y, T, X,
    model='logit',      # Changed: family → model
    target='ame',
    n_folds=50,
    epochs=100
)

# Access results (InferenceResult has same fields)
print(result.mu_hat)

# NEW: Diagnostics
print(result.diagnostics.summary())
```

**Backward compatible (v0.3.0):**
```python
# This still works in v0.3.0 (with deprecation warning)
from deep_inference import inference

result = inference(
    Y, T, X,
    family='logit',  # DeprecationWarning: use model= instead
    target='ame'
)
```

**New features (v0.3.0+):**
```python
from deep_inference import inference
from deep_inference.estimands import DoublyRobustATE, CATE, IV2SLS

# Doubly robust ATE
result = inference(Y, T, X, model="linear", target="ate", estimator="dr")

# Or full control
dr = DoublyRobustATE(propensity_clip=(0.05, 0.95))
result = dr.fit(Y, T, X)

# CATE with aggregation
cate = CATE()
result = cate.fit(Y, T, X)
ate = result.aggregate()

# IV/2SLS
iv = IV2SLS()
result = iv.fit(Y, T, Z, X)
```

---

## Appendix A: LOC Estimates

| Component | New LOC | Refactor LOC | Total |
|-----------|---------|--------------|-------|
| **P0: Foundation** | | | |
| API Unification | 100 | 200 | 300 |
| TreatmentEvaluator | 100 | 50 | 150 |
| InferenceDiagnostics | 300 | 100 | 400 |
| **P1: Causal Estimands** | | | |
| PropensityEstimator | 200 | 0 | 200 |
| DoublyRobustATE | 300 | 0 | 300 |
| ATT | 150 | 0 | 150 |
| CATE | 200 | 0 | 200 |
| **P1: New Targets** | | | |
| Elasticity | 100 | 0 | 100 |
| WTP | 100 | 0 | 100 |
| ConsumerWelfare | 100 | 0 | 100 |
| **P2: IV/2SLS** | | | |
| MomentModel protocol | 150 | 0 | 150 |
| IV2SLS class | 400 | 0 | 400 |
| **Infrastructure** | | | |
| Type annotations | 200 | 300 | 500 |
| eval_09_dr_ate | 200 | 0 | 200 |
| eval_10_iv | 200 | 0 | 200 |
| **Total** | **~2800** | **~650** | **~3450** |

---

## Appendix B: New Package Structure

```
src/deep_inference/
├── __init__.py              # Unified exports
├── core/
│   ├── algorithm.py         # Legacy (deprecated)
│   └── ...
├── models/
│   ├── base.py              # StructuralModel protocol
│   ├── linear.py            # Linear model
│   ├── logit.py             # Logit model
│   ├── structural_net.py    # Neural network θ(x)
│   └── custom.py            # CustomModel
├── targets/
│   ├── base.py              # Target protocol
│   ├── average_parameter.py # E[θ_j]
│   ├── marginal_effect.py   # AME
│   ├── elasticity.py        # NEW: Price elasticity
│   ├── wtp.py               # NEW: Willingness to pay
│   ├── welfare.py           # NEW: Consumer welfare
│   └── custom.py            # CustomTarget
├── lambda_/
│   ├── base.py              # LambdaStrategy protocol
│   ├── compute.py           # Regime A
│   ├── analytic.py          # Regime B
│   ├── estimate.py          # Regime C
│   └── selector.py          # Auto-detection
├── estimands/               # NEW
│   ├── __init__.py
│   ├── base.py              # CausalEstimand protocol
│   ├── propensity.py        # PropensityEstimator
│   ├── doubly_robust.py     # DoublyRobustATE
│   ├── att.py               # ATT
│   ├── cate.py              # CATE
│   ├── moments.py           # MomentModel protocol
│   ├── gmm.py               # GMM objective
│   └── iv.py                # IV2SLS
├── engine/
│   ├── crossfit.py          # Cross-fitting (2-way, 3-way)
│   ├── assembler.py         # Influence function assembly
│   └── variance.py          # SE/CI computation
├── utils/
│   ├── treatment.py         # NEW: TreatmentEvaluator
│   └── linalg.py
├── diagnostics.py           # NEW: InferenceDiagnostics
├── types.py                 # NEW: jaxtyping annotations
└── families/                # Legacy (deprecated)
    └── ...
```

---

## Appendix C: References

1. Farrell, M. H., Liang, T., & Misra, S. (2021). Deep neural networks for estimation and inference. *Econometrica*, 89(1), 181-213.

2. Farrell, M. H., Liang, T., & Misra, S. (2025). Deep learning for structural estimation. *arXiv preprint*.

3. Chernozhukov, V., et al. (2018). Double/debiased machine learning for treatment and structural parameters. *The Econometrics Journal*, 21(1), C1-C68.

4. Kennedy, E. H. (2023). Semiparametric doubly robust targeted double machine learning: A review. *arXiv preprint*.

5. Athey, S., & Imbens, G. (2016). Recursive partitioning for heterogeneous causal effects. *PNAS*, 113(27), 7353-7360.

6. Small, K. A., & Rosen, H. S. (1981). Applied welfare economics with discrete choice models. *Econometrica*, 49(1), 105-130.

---

**Document Status**: Ready for Review
**Next Steps**: Engineering review → Prioritization → Sprint planning
