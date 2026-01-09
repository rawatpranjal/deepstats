# Changelog

## 2026-01-08

### Major Refactor: src2 as Main Package
- **src2 is now the main package** - clean API with direct parameters
- New API: `from src2 import structural_dml; result = structural_dml(Y, T, X, family='linear')`
- Ported all 8 families: linear, logit, poisson, gamma, gumbel, tobit, negbin, weibull
- Archived `src/deep_inference/` to `archive/deep_inference_v1/`
- MC infrastructure (run_mc, metrics, logging, dgp) now in archive for reference
- Updated pyproject.toml, README.md, CLAUDE.md for new structure

### Package Rename (earlier)
- Renamed package from `deepstats` to `deep-inference` for PyPI availability
- Python import: `from src2 import ...` (new), `from deep_inference import ...` (archived)
- PyPI install: `pip install deep-inference`

## 2026-01-07

### Documentation Update
- Added verification page comparing deepstats against original FLM2 repository
- Updated tutorials with final linear validation (M=100, N=20K): Naive 8% → IF 95% coverage

### Comprehensive Validation Study Script
- Added `prototypes/validation_study.py` - main example for ReadTheDocs
- Compares Naive vs IF inference across linear, logit, poisson families
- Config: N=50K, M=50, K=50, epochs=500, [64,64,64,32] architecture
- Tracks: RMSE(μ̂), Corr(α/β), SE ratio, Coverage, training quality
- Outputs: CSV, JSON report, PNG plots (coverage/SE comparison)
- Uses src2's `structural_dml()` API with tqdm progress

### Data-Rich Validation (Part XV)
- **Goal**: Achieve BOTH valid inference AND good parameter recovery
- **Finding**: N=20000 achieves Corr(α)=0.86, Corr(β)=0.58 with Coverage 96%
- **Scaling**: Corr(β) improves from 0.28 (N=2000) → 0.43 (N=5000) → 0.58 (N=20000)
- Added Part XV to VALIDATION_REPORT.md with sample size recommendations
- For rich heterogeneity recovery, use N=20000+

### Poisson Family Implementation & Validation
- **Added PoissonFamily to src2**: Y ~ Poisson(λ), λ = exp(α + βT)
- Model: log-linear rate with heterogeneous treatment effects
- Closed-form gradient and Hessian (weight = λ)
- Requires three-way splitting (Hessian depends on θ)
- **Validation Results (M=30, N=2000, K=50)**:
  - Aggregate Lambda: Coverage **96.7%**, SE ratio 0.88, reg rate 0%
  - MLP Lambda: Coverage 80% (fails - underfits)
- **N=5000 confirms validity**: Coverage **95.0%**, SE ratio **1.17**
- Recommendation: Use `lambda_method='aggregate'` for Poisson

### Additional Validation (Phase 13)
- **Logit continuous T**: SE ratio 1.06 (nearly perfect!), coverage 100%, reg rate 0%
- MLP Lambda still overfits with continuous T (40% reg rate, negative eigenvalues)
- **Naive vs Debiased**: Both achieve valid coverage for well-specified models
- IF correction provides theoretical validity and robustness
- Recommendation: Use `lambda_method='aggregate'` regardless of T type

### SE Ratio Optimization (Phase 12)
- **Root cause of SE ratio 1.66**: Insufficient cross-fitting folds (K=20 vs K=50)
- **With K=50**: SE ratio drops to ~1.12-1.19 for Aggregate Lambda
- **With N=5000, K=50**: SE ratio = **1.04** (nearly perfect!)
- Ridge(α=0.001) gives similar results to Aggregate
- Added `ridge_alpha` parameter to LambdaEstimator for configurable regularization
- Updated VALIDATION_REPORT.md with Phase 12 findings

### Lambda Estimation Investigation (Phase 11)
- **Root cause identified**: MLP Lambda estimator was overfitting, producing singular predictions
- **Key finding**: Conditional expectation Λ(x) = E[ℓ_θθ | X=x] should average over both T=0 and T=1
- **When properly averaged, Λ is FULL RANK** - min eigenvalue = 0.046 (not singular!)
- Added `lambda_method='aggregate'` option for three-way splitting (ensures full-rank Λ for binary T)
- MLP produced negative eigenvalues (-0.062) while Ridge/Aggregate/Propensity produced 0% singular predictions
- Implemented PropensityWeightedLambdaEstimator (theoretically correct for heterogeneous propensity)

### Post-Validation Fixes
- Changed default K from 20 to 50 for stable SE estimation
- Added adaptive eigenvalue monitoring to safe_inverse and batch_inverse
- Added diagnostics: min_lambda_eigenvalue, n_regularized, pct_regularized
- Added warnings for high Lambda regularization rate and high correction variance ratio
- Updated documentation with requirements for valid inference

### Validation Results
- **Comprehensive MC validation completed** - Algorithm validated for well-specified models
- Simple Linear DGP: SE ratio 1.00-1.05, coverage 92-100% across N=500-5000 (M=50 per size)
- Logit DGP: Coverage 95-97% with three-way splitting (M=100)
- K≥50 folds required for stable SE estimation
- Complex DGP fails due to model misspecification (not algorithm bug)
- Created detailed VALIDATION_REPORT.md with all test results

### Features
- Implemented three-way splitting for nonlinear families (logit, poisson, etc.) where Λ depends on θ
- Added nonparametric Λ(x) estimation via LambdaEstimator class (MLP or LightGBM)
- Made treatment centering configurable (default off to match paper formula)
- Updated variance estimation to use within-fold formula from paper
- Added `lambda_depends_on_theta()` and `hessian_at_point()` methods to all families

### Documentation
- Added Sphinx documentation with Read the Docs support
- Created tutorials for core models: linear, logit, poisson, tobit, negbin
- Added comprehensive theory section with FLM framework and Structural DML algorithm
- Split Theory and Algorithm into separate top-level navigation sections
- Updated documentation with comprehensive FLM framework and algorithm content
- Added practical Usage Guide in Getting Started section
- Added Validation section with Monte Carlo simulation study results
- Set up autodoc API reference generation
