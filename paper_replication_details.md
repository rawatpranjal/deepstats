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
