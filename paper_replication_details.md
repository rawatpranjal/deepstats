# FLM Paper Replication Details

## 1. Paper Coverage Analysis

### Papers Analyzed
- **FLM2021**: "Deep Neural Networks for Estimation and Inference" (Econometrica)
- **FLM2025**: "Deep Learning for Individual Heterogeneity" (arXiv v3, April 2025)

### Coverage Summary: ~67% of Paper Scope

| Category | Implemented | Missing |
|----------|-------------|---------|
| Core Math Objects (ψ, ℓ_θ, ℓ_θθ, H_θ, Λ) | 6/6 | 0 |
| Lambda Regimes (A/B/C) | 3/3 | 0 |
| Cross-fitting (2-way, 3-way) | 2/2 | 0 |
| GLM Families | 12 | 4 |
| Target Functionals | 3 | 5 |
| Causal Estimands | 1 | 3 |
| Applications | 0 | 4 |

---

## 2. Fully Implemented ✅

### Core Algorithm (Theorem 2)
| Component | Paper Reference | Repo Location |
|-----------|-----------------|---------------|
| Influence Function ψ | Eq 3.6: `ψ = H - H_θ·Λ⁻¹·ℓ_θ` | `engine/assembler.py` |
| Score ℓ_θ | Gradient of loss | `autodiff/score.py` |
| Hessian ℓ_θθ | Second derivative | `autodiff/hessian.py` |
| Target Jacobian H_θ | ∂H/∂θ | `autodiff/jacobian.py` |
| Lambda Λ(x) | E[ℓ_θθ\|X=x] | `lambda_/` |

### Three Lambda Regimes
| Regime | When | Method | Repo |
|--------|------|--------|------|
| A | RCT + known F_T | MC integration | `lambda_/compute.py` |
| B | Linear model | Analytic Λ=E[TT'\|X] | `lambda_/analytic.py` |
| C | Observational + nonlinear | Estimate via regression | `lambda_/estimate.py` |

### Cross-Fitting
- **2-way split**: Regimes A, B (θ̂ and Λ̂ on same fold)
- **3-way split**: Regime C (separate folds for θ̂ and Λ̂)
- **K=50 folds**: Paper recommendation, repo default

### GLM Families (12 implemented)
Linear, Logit, Poisson, Gamma, Gaussian, Gumbel, Tobit, NegBin, Weibull, Probit, Beta, ZIP

### Targets
- `AverageParameter`: E[θ_j]
- `AverageMarginalEffect`: E[G'·β] for logit
- `CustomTarget`: User-defined H(x,θ,t̃)

---

## 3. Not Implemented ❌

### HIGH Priority Gaps

#### 3.1 Instrumental Variables (2SLS/IV)
**Paper**: FLM2025 Section C.6
**Why needed**: Endogeneity is ubiquitous in economics
**Complexity**: Requires two-equation system, moment conditions

#### 3.2 Doubly Robust ATE
**Paper**: FLM2021 Eq 4.1-4.3
**Why needed**: Classic DML formulation, IPW + outcome regression
**What's missing**: Propensity score estimation, IPW weighting

#### 3.3 Vector Treatments (d_T > 1)
**Paper**: Both papers support arbitrary d_T
**Why needed**: Demand systems, multiple treatments
**Current**: Repo assumes scalar T

### MEDIUM Priority Gaps

#### 3.4 Berry Logit (Demand Estimation)
**Paper**: FLM2025 Section C.3
**Use case**: Industrial organization, market share models

#### 3.5 Fractional Response
**Paper**: FLM2025 Section C.4
**Use case**: Outcomes bounded in [0,1]

#### 3.6 ATT (Average Treatment on Treated)
**Paper**: FLM2021 Eq 3.8
**Use case**: Policy evaluation when treatment isn't randomized

#### 3.7 Advanced Targets
- Elasticity: `(1-G)·β·t̃`
- Willingness to Pay: `β/α`
- Consumer Welfare: `-log(1+exp(α+β·t))/β`
- Optimal Price: Implicit (fixed-point)
- Expected Profits: Non-closed-form

### LOW Priority Gaps
- Short Stacking (model averaging)
- Higher-Order IF (conjecture only)
- Cobb-Douglas production

---

## 4. Package Design Analysis

### 4.1 Current Architecture

```
src/deep_inference/
├── families/          # GLM loss functions (Protocol: Family)
├── models/            # Structural models (Protocol: StructuralModel)
├── targets/           # Target functionals (Protocol: Target)
├── lambda_/           # Lambda strategies (Protocol: LambdaStrategy)
├── engine/            # Cross-fitting orchestration
├── autodiff/          # Batched derivatives (score, hessian, jacobian)
└── core/              # Legacy API
```

### 4.2 Current Abstractions

**Family Protocol**:
```python
class Family(Protocol):
    def loss(y, t, theta) -> Tensor
    def gradient(y, t, theta) -> Optional[Tensor]  # None = autodiff
    def hessian(y, t, theta) -> Optional[Tensor]   # None = autodiff
    def hessian_depends_on_theta() -> bool         # Regime selector
```

**Target Protocol**:
```python
class Target(Protocol):
    def h(x, theta, t_tilde) -> Tensor
    def jacobian(x, theta, t_tilde) -> Optional[Tensor]  # None = autodiff
```

**LambdaStrategy Protocol**:
```python
class LambdaStrategy(Protocol):
    requires_theta: bool           # Regime A/B vs C
    requires_separate_fold: bool   # 2-way vs 3-way
    def fit(X, hessians, theta) -> Self
    def predict(X) -> Tensor
```

### 4.3 Are We on the Right Track? YES ✅

**Strengths of Current Design**:

1. **Protocol-based extensibility**: New families/targets/strategies plug in cleanly
2. **Autodiff fallback**: Closed-form optional, autodiff always available
3. **Regime detection**: `hessian_depends_on_theta` drives 2-way vs 3-way automatically
4. **Separation of concerns**: Loss, target, Lambda, cross-fitting are independent

**Evidence it works**:
- 12 families added without core changes
- 3 Lambda regimes coexist
- Custom targets work via autodiff
- Validated 96% coverage

### 4.4 What's Missing for Full Paper Coverage

#### A. Moment-Based Inference (for IV/GMM)

Current design assumes single loss function. IV requires **moment conditions**:

```python
# NEEDED: MomentModel protocol
class MomentModel(Protocol):
    def moments(y, t, x, z, theta) -> Tensor  # (n, n_moments)
    def moment_jacobian(y, t, x, z, theta) -> Tensor  # (n, n_moments, d_theta)
```

**Impact**: New protocol, new engine path, but reuses autodiff infrastructure.

#### B. Multi-Equation Systems

Berry Logit and demand systems have **multiple equations**:

```python
# NEEDED: SystemModel protocol
class SystemModel(Protocol):
    n_equations: int
    def residuals(y, t, x, theta) -> Tensor  # (n, n_eq)
```

**Impact**: Extension of StructuralModel, manageable.

#### C. Vector Treatments

Current: `T: (n,)` scalar
Needed: `T: (n, d_t)` matrix

**Impact**:
- Family signatures change: `loss(y, t, theta)` where `t: (d_t,)`
- Hessian becomes `(d_theta, d_theta)` where `d_theta = 1 + d_t`
- Moderate refactor, but protocol stays same

#### D. Propensity Scores (for Doubly Robust)

Current: No propensity estimation
Needed: `π(x) = P(T=1|X=x)` estimator

```python
# NEEDED: PropensityEstimator
class PropensityEstimator(Protocol):
    def fit(T, X) -> Self
    def predict(X) -> Tensor  # (n,) probabilities
```

**Impact**: New component, integrates into cross-fitting for IPW.

#### E. Implicit Targets (Optimal Pricing)

Current: `H(x, θ, t̃)` explicit function
Needed: `H(x, θ)` where target solves `∂π/∂r = 0` implicitly

```python
# NEEDED: ImplicitTarget
class ImplicitTarget(Target):
    def solve(x, theta) -> Tensor  # Find r* via fixed-point
    def h(x, theta, t_tilde=None) -> Tensor  # Returns profit at r*
```

**Impact**: Extension of Target protocol with solver loop.

### 4.5 Recommended Architecture Extensions

```
src/deep_inference/
├── families/           # ✅ Keep as-is
├── models/
│   ├── structural.py   # ✅ Current: loss-based
│   ├── moment.py       # NEW: IV/GMM moment conditions
│   └── system.py       # NEW: Multi-equation systems
├── targets/
│   ├── explicit.py     # ✅ Current: H(x,θ,t̃)
│   └── implicit.py     # NEW: Fixed-point targets
├── lambda_/            # ✅ Keep as-is
├── engine/
│   ├── crossfit.py     # ✅ Keep, extend for DR
│   └── propensity.py   # NEW: IPW estimation
├── autodiff/           # ✅ Keep as-is
└── estimands/          # NEW: ATE, ATT, CATE wrappers
    ├── ate.py
    ├── att.py
    └── doubly_robust.py
```

### 4.6 Complexity Estimates

| Gap | New Code | Refactor | Difficulty |
|-----|----------|----------|------------|
| Vector treatments | ~200 LOC | ~500 LOC | Medium |
| Doubly Robust ATE | ~300 LOC | ~100 LOC | Medium |
| ATT | ~150 LOC | ~50 LOC | Easy |
| IV/2SLS | ~500 LOC | ~200 LOC | Hard |
| Berry Logit | ~200 LOC | 0 | Medium |
| Implicit Targets | ~300 LOC | ~100 LOC | Medium |
| Advanced Targets | ~400 LOC | 0 | Easy |

**Total for full paper coverage**: ~2000 new LOC, ~950 refactor LOC

---

## 5. Summary: Current State vs Full Replication

### What We Have
- **Complete Theorem 2 implementation** (influence functions, all derivatives)
- **All three Lambda regimes** with automatic detection
- **Validated inference** (96% coverage with ridge)
- **12 GLM families** with extensible protocol
- **Flexible targets** via autodiff
- **Production-ready API** (`structural_dml()`, `inference()`)

### What We Don't Have
- **Causal estimands beyond ATE** (ATT, doubly robust, policy eval)
- **Instrumental variables** (moment-based inference)
- **Demand estimation** (Berry logit, systems)
- **Vector treatments** (multiple treatments)
- **Advanced targets** (elasticity, WTP, welfare, implicit)

### Verdict

The current architecture is **fundamentally sound** and follows the paper's structure closely. The gaps are **additive extensions**, not architectural rewrites. The protocol-based design means:

1. New families → implement `Family` protocol
2. New targets → implement `Target` protocol
3. New Lambda methods → implement `LambdaStrategy` protocol
4. New estimands → compose existing components

**Estimated effort for 100% paper coverage**: 4-6 weeks of focused development.

---

## 6. World-Class Architecture Gap Analysis

### 6.1 What Google Engineers Would Build Differently

#### A. **Type-Safe Tensor Shapes via Generics**

**Current state**: Shape assumptions scattered across code, runtime errors
```python
# Current: implicit shapes, no compile-time checks
def loss(y, t, theta):  # What shapes? Who knows!
    return ...
```

**Google approach**: Typed tensor shapes (like JAX's `ShapedArray` or TensorFlow's shape annotations)
```python
# Google-style: explicit shape contracts
@typeguard
def loss(
    y: Float[Tensor, "batch"],           # (n,)
    t: Float[Tensor, "batch treatment"],  # (n, d_t)
    theta: Float[Tensor, "batch params"]  # (n, d_theta)
) -> Float[Tensor, "batch"]:
    ...
```

**Gap**: ~500 LOC to add comprehensive type annotations + runtime shape validation

---

#### B. **Single Unified Computation Graph**

**Current state**: Two parallel implementations
- Legacy: `structural_dml()` → `structural_dml_core()` → custom training loop
- New: `inference()` → `run_crossfit()` → separate training loop

**Google approach**: Single graph-based computation
```python
# Define computation as composable transforms
pipeline = (
    DataLoader(Y, T, X)
    | CrossFitSplitter(K=50)
    | ThetaEstimator(model, hidden_dims=[64,32])
    | LambdaEstimator(regime=auto_detect)
    | InfluenceFunctionAssembler()
    | VarianceEstimator()
)
result = pipeline.run()
```

**Benefits**:
- Automatic parallelization across folds
- Caching of intermediate results
- Clear debugging/profiling points

**Gap**: ~1500 LOC architectural rewrite

---

#### C. **First-Class Vector Treatments**

**Current state**: Hardcoded scalar T assumption throughout
```python
# Broken for d_t > 1
logits = alpha + beta * t  # scalar multiplication
```

**Google approach**: Generic treatment dimension
```python
class Model(Generic[ThetaDim, TreatmentDim]):
    def loss(self, y, t: Tensor[..., TreatmentDim], theta: Tensor[..., ThetaDim]) -> Tensor
```

**Paper requirement**: FLM2025 explicitly supports `d_T > 1` for demand systems, multiple treatments

**Gap**: ~500 LOC refactor across all families, Lambda estimation, cross-fitting

---

#### D. **Declarative Model Specification**

**Current state**: Imperative class definitions with duplicated logic
```python
class LogitFamily:
    def loss(...): return log(1 + exp(eta)) - y * eta
    def gradient(...): return (sigmoid(eta) - y) * T1
    def hessian(...): return sigmoid(eta) * (1 - sigmoid(eta)) * outer(T1, T1)
```

**Google approach**: Declarative specification + code generation
```python
# Just declare the loss - everything else derived
@structural_model
def logit(y, t, alpha, beta):
    eta = alpha + beta @ t
    return -y * eta + log(1 + exp(eta))

# Autodiff generates: gradient, hessian, score
# Static analysis detects: hessian_depends_on_theta=True, theta_dim=1+d_t
```

**Benefits**:
- Single source of truth (DRY)
- Impossible to have gradient/loss mismatch
- Automatic regime detection from loss structure

**Gap**: ~800 LOC for decorator system + static analysis

---

#### E. **Compile-Time Regime Detection**

**Current state**: Runtime detection via boolean flags
```python
if model.hessian_depends_on_theta:
    regime = Regime.C
```

**Google approach**: Type-level regime encoding
```python
class Model(Protocol[R: Regime]):
    ...

class Logit(Model[Regime.C]):  # Regime encoded in type
    ...

def run_crossfit(model: Model[R]) -> Result[R]:
    # Compiler knows regime at compile time
    # Dead code elimination for unused regime paths
```

**Gap**: Advanced Python typing (~300 LOC) or language change

---

#### F. **Probabilistic Programming Integration**

**Current state**: Point estimates only, no uncertainty in intermediate quantities
```python
theta_hat = network(X)  # Single point estimate
```

**Google approach**: Full posterior over θ(x)
```python
# Bayesian neural network or ensemble
theta_posterior = BayesianNet(X).posterior()  # Distribution
lambda_posterior = integrate(hessian, theta_posterior)  # Propagated uncertainty
```

**Paper mention**: FLM2021 Section 6 discusses bootstrap but doesn't implement

**Gap**: Major architectural change (~2000 LOC) + different theoretical framework

---

### 6.2 Missing Paper Components: Detailed Analysis

#### **CRITICAL GAP: Instrumental Variables (IV/2SLS)**

**Paper reference**: FLM2025 Appendix C.6, Berry Logit

**What papers describe**:
```
E[log(s_jm/s_0m) | X, Z] = α_j(x) + β(x)·(p_jm - p_0m)
```
Where Z is instrument for endogenous price P.

**Why missing is serious**:
- Endogeneity is THE core problem in empirical economics
- Without IV, package limited to RCTs and exogenous variation
- Berry logit is workhorse model for demand estimation

**Implementation requirements**:
1. `MomentModel` protocol for moment conditions g(Y, T, Z, θ) = 0
2. GMM objective: min_θ g'Wg where W is weighting matrix
3. Two-step GMM with optimal weighting
4. Influence function for GMM: different formula than MLE

**Complexity**: ~600 LOC, Hard difficulty

---

#### **CRITICAL GAP: Doubly Robust ATE**

**Paper reference**: FLM2021 Eq 4.1-4.3

**The formula** (from paper):
```
ψ_t(z) = 1{T=t}(y - μ_t(x)) / P[T=t|X=x] + μ_t(x)
τ̂ = E_n[ψ̂_1(z) - ψ̂_0(z)]
```

**What's missing**:
1. Propensity score estimation: π(x) = P(T=1|X)
2. Inverse probability weighting (IPW)
3. Doubly robust combination
4. Overlap trimming for propensity extremes

**Why current approach differs**:
- Current: Structural model ℓ(Y, T, θ(X)) where T enters through θ
- Paper: Potential outcomes Y(0), Y(1) with separate regressions μ_0(x), μ_1(x)

**These are fundamentally different frameworks**:
- Structural: T affects Y through known mechanism
- Potential outcomes: T switches between two unknown functions

**Complexity**: ~400 LOC, Medium difficulty (but conceptually different)

---

#### **MODERATE GAP: Implicit Targets (Optimal Pricing)**

**Paper reference**: FLM2025 Section 4.2 (Bertrand application)

**What papers describe**:
```
r*(x) = argmax_r π(r, x) where π involves P[default], P[accept], loan terms
μ* = E[r*(X)] or E[π(r*(X), X)]
```

**Challenge**: r* is not closed-form, requires fixed-point iteration

**Implementation**:
```python
class ImplicitTarget(Target):
    def solve(self, x, theta) -> Tensor:
        """Find r* that maximizes profit"""
        def profit(r):
            return self.profit_fn(r, x, theta)
        return scipy.optimize.minimize_scalar(-profit).x

    def h(self, x, theta, t_tilde=None) -> Tensor:
        r_star = self.solve(x, theta)
        return self.profit_fn(r_star, x, theta)

    def jacobian(self, x, theta, t_tilde=None) -> Tensor:
        # Implicit function theorem: dr*/dθ = -∂²π/∂r∂θ / ∂²π/∂r²
        return autodiff_implicit_jacobian(self.profit_fn, x, theta)
```

**Complexity**: ~300 LOC, Medium difficulty

---

### 6.3 Architectural Debt: Current Codebase Issues

#### **DEBT 1: Redundant Legacy/New API Split**

**Problem**: Two implementations of same algorithm
- `structural_dml()` calls `structural_dml_core()`
- `inference()` calls `run_crossfit()`
- Both do K-fold cross-fitting with theta estimation + Lambda

**Technical debt**: Bug fixes must be applied twice, divergence risk

**Fix**: Unify on `run_crossfit()`, make `structural_dml()` a thin wrapper
```python
def structural_dml(Y, T, X, family='linear', ...):
    model = FAMILY_MAP[family]()
    target = AverageParameter(index=1)  # β
    return inference(Y, T, X, model=model, target=target, ...)
```

**Effort**: ~200 LOC refactor

---

#### **DEBT 2: Family vs Model Duplication**

**Problem**: `families/logit.py::LogitFamily` vs `models/logit.py::Logit`
- Same math, different interfaces
- Family used by legacy API, Model by new API

**Fix**: Single implementation with adapter
```python
class Logit(StructuralModel):
    ...

class LogitFamily:
    """Legacy adapter"""
    def __init__(self):
        self._model = Logit()

    def loss(self, *args):
        return self._model.loss(*args)
```

**Effort**: ~150 LOC

---

#### **DEBT 3: Scattered t_tilde Handling**

**Problem**: Treatment evaluation point handled inconsistently
- Sometimes scalar, sometimes (n,) tensor
- Default calculation in multiple places
- Broadcasting rules unclear

**Fix**: Centralized `TreatmentEvaluator` class
```python
class TreatmentEvaluator:
    def __init__(self, t_tilde: Optional[float] = None, method: str = 'mean'):
        self.t_tilde = t_tilde
        self.method = method

    def evaluate(self, T: Tensor, n: int) -> Tensor:
        if self.t_tilde is not None:
            return torch.full((n,), self.t_tilde)
        elif self.method == 'mean':
            return torch.full((n,), T.mean())
        elif self.method == 'per_observation':
            return T
```

**Effort**: ~100 LOC

---

#### **DEBT 4: No Diagnostic Framework**

**Problem**: Diagnostics scattered, incomplete
- `min_lambda_eigenvalue` in result
- `correction_ratio` computed but not systematically
- No unified health check

**Google approach**: Structured diagnostics
```python
@dataclass
class InferenceDiagnostics:
    # Lambda quality
    lambda_condition_numbers: Tensor  # (n,)
    lambda_min_eigenvalues: Tensor    # (n,)
    lambda_psd_violations: int

    # Influence function quality
    psi_variance: float
    psi_skewness: float
    psi_outlier_count: int

    # Coverage prediction
    estimated_coverage: float  # Based on normality tests

    def summary(self) -> str:
        """Formatted diagnostic report"""

    def health_check(self) -> List[Warning]:
        """Return list of potential issues"""
```

**Effort**: ~300 LOC

---

### 6.4 What Full Replication Requires: Complete Checklist

#### **Tier 1: Core Theory (DONE ✅)**
- [x] Influence function formula ψ = H - H_θ Λ⁻¹ ℓ_θ
- [x] Score ℓ_θ via autodiff
- [x] Hessian ℓ_θθ via autodiff
- [x] Target Jacobian H_θ via autodiff
- [x] Lambda estimation (3 regimes)
- [x] K-fold cross-fitting
- [x] Variance estimation
- [x] Confidence intervals

#### **Tier 2: Model Coverage (MOSTLY DONE)**
- [x] Linear (OLS)
- [x] Logit (binary choice)
- [x] Poisson (counts)
- [x] Gamma (positive continuous)
- [x] Tobit (censored)
- [x] NegBin (overdispersed counts)
- [x] Weibull (survival)
- [x] Gumbel (extreme values)
- [x] Probit (binary, normal link)
- [x] Beta (proportions)
- [x] ZIP (zero-inflated)
- [ ] Fractional response (quasi-likelihood)
- [ ] Berry logit (demand)
- [ ] Multinomial logit
- [ ] Cobb-Douglas

#### **Tier 3: Targets (PARTIAL)**
- [x] Average parameter E[θ_j]
- [x] Average marginal effect E[G'·β]
- [x] Custom explicit targets
- [ ] Elasticity (1-G)·β·t̃
- [ ] Willingness to pay β/α
- [ ] Consumer welfare
- [ ] Implicit targets (optimal pricing)
- [ ] Policy profit comparison

#### **Tier 4: Causal Estimands (MINIMAL)**
- [x] ATE (via structural interpretation)
- [ ] ATT (treatment on treated)
- [ ] Doubly robust ATE
- [ ] CATE (conditional ATE)
- [ ] Policy regret bounds

#### **Tier 5: Extensions (NOT STARTED)**
- [ ] Vector treatments d_T > 1
- [ ] Instrumental variables
- [ ] GMM/moment conditions
- [ ] Multi-equation systems
- [ ] Short stacking (ensemble)
- [ ] Bayesian uncertainty

---

### 6.5 Recommended Roadmap

**Phase 1: Consolidation (1-2 weeks)**
- Unify legacy/new API (~200 LOC)
- Add comprehensive diagnostics (~300 LOC)
- Fix t_tilde handling (~100 LOC)
- Add type annotations to core modules (~500 LOC)

**Phase 2: Vector Treatments (1 week)**
- Generalize T from (n,) to (n, d_t)
- Update all family/model signatures
- Update Lambda estimation for larger matrices
- ~500 LOC refactor

**Phase 3: Missing Targets (1 week)**
- Elasticity, WTP, welfare targets (~200 LOC)
- Implicit target protocol + solver (~300 LOC)

**Phase 4: Causal Estimands (2 weeks)**
- Propensity score estimation (~200 LOC)
- Doubly robust ATE (~300 LOC)
- ATT (~150 LOC)

**Phase 5: Instrumental Variables (2-3 weeks)**
- Moment model protocol (~200 LOC)
- GMM objective + optimal weighting (~300 LOC)
- Berry logit family (~200 LOC)
- IV influence function (~300 LOC)

**Total: 6-9 weeks for complete paper coverage**

---

### 6.6 The Honest Assessment

**What we did well**:
1. Core Theorem 2 implementation is mathematically correct
2. Protocol-based design allows extension without rewrites
3. 96% coverage validation proves inference is valid
4. Three regimes properly distinguished
5. Autodiff fallback means new families "just work"

**What a Google team would do better**:
1. Type-safe tensor shapes from day one
2. Single computation graph, not two APIs
3. Vector treatments as first-class citizens
4. Declarative model specification
5. Comprehensive test coverage (property-based testing)
6. Performance profiling and optimization
7. Documentation with executable examples

**The fundamental question**: Is this package for researchers (flexibility) or practitioners (reliability)?

- **Current**: Researcher-focused. Flexible but requires understanding internals.
- **Google**: Practitioner-focused. Opinionated, batteries-included, hard to misuse.

**Verdict**: The architecture is sound for a research package. The gaps are feature gaps, not design flaws. A Google team would add more guardrails and automation, but wouldn't fundamentally restructure the approach.
