# Changelog

## 2026-01-13

### Eval 07: End-to-End User Experience + Round G SE Calibration
- Created `evals/eval_07_e2e.py` demonstrating full analyst workflow
- Scenario: Loan application with heterogeneous rate sensitivity
- **Parts A-F**: Bootstrap Oracle inference, structural_dml() NN inference, comparison tables
- Tests: μ* coverage (Oracle, Bootstrap, NN IF), heterogeneity recovery (β̂(x) correlation), θ(x) bootstrap coverage
- **Results: 7/7 PASS** - All methods achieve valid coverage, Oracle-NN correlation 0.73/0.40
- **Round G (NEW)**: Multi-seed SE calibration (M=100 seeds)
  - Validates SE ratio = mean_SE / empirical_SE ∈ [0.9, 1.1]
  - Validates coverage ∈ [93%, 97%]
  - CLI: `--round-g` (M=100), `--quick-g` (M=20)
  - **Results: PASS** - Coverage 95%, SE Ratio 0.91

### Tutorial: New `inference()` API Section Added
- Added Section 5 to `tutorials/01_showcase.ipynb` demonstrating new API
- Covers: Basic comparison vs `structural_dml()`, AME target, custom target with autodiff, randomization mode (Regime A)
- All 4 test cases pass: beta, AME, custom target, ComputeLambda

### New `inference()` API Validated
- Discovered and validated new modular architecture: `models/`, `targets/`, `lambda_/`, `engine/`
- **Flexible targets**: `CustomTarget(h_fn)` with autodiff Jacobian, `AME` with closed-form
- **Randomization mode**: `ComputeLambda` computes Λ via Monte Carlo (Regime A, 2-way split)
- **Regime detection**: Auto-detects A/B/C and selects appropriate Lambda strategy
- **Validation**: 20/20 coverage (100%) on canonical DGP AME target
- Usage: `inference(Y, T, X, model='logit', target='ame', is_randomized=True, treatment_dist=Normal())`

### Eval 05: Influence Function Assembly (Ruthless Rewrite)
- Complete rewrite with 4 rounds of validation: Mechanical Assembly, Neyman Orthogonality, Variance Formula, Multi-Seed Coverage
- **Round A**: Ruthless tolerances (Corr > 0.999, |Bias| < 0.001, Max|diff| < 0.01) for AME + AverageParameter targets
- **Round B**: Neyman orthogonality - perturb θ, verify bias scales as O(δ²)
- **Round C**: Variance formula sanity checks
- **Round D**: 50-seed coverage test (88-98% threshold)
- **Key fix**: Previous test used different Lambda for oracle vs package, causing spurious 0.1-0.15 diffs. Now uses identical Lambda_inv.
- **Results: 4/4 PASS** - Assembly is now verified to machine precision (max|diff| = 1e-6)
- Coverage at lower bound (88%), SE ratio 0.87 - worth monitoring

### Eval 04: Target Jacobian Expansion (Ruthless Firewall)
- Expanded from narrow AME+Logit test to full **Targets × Families × Edge Cases** matrix
- **Part 1 - Targets (Logit)**: AverageParameter, AME, AveragePrediction (45 tests)
- **Part 2 - Families (AME)**: Linear, Poisson, Probit (25 tests)
- **Part 3 - Edge Cases**: Near-boundary σ(±5), tiny β=0.001, large β=10 (8 tests)
- **Part 4 - Batched vmap**: 14 configs × 100 random θ each
- Added 12 oracle Jacobian functions to `dgp.py` with full derivations
- **Results: 92/92 PASS** - max|err| = 1.78e-15 (machine precision)

### Eval 03: Timing & Enhanced Statistics
- Added execution timing for all parts and methods (Part A: 0.16s, Part B: 0.02s, Part C: 22.01s)
- Added LightGBM method to EstimateLambda (lgbm: Corr=0.977, 3/3 PASS)
- **Part A**: Added eigenvalues, condition number (1.62), determinant
- **Part B**: Added condition number range [1.00, 2.69] across x values
- **Part C Extended Table**: Corr, Frob, Max, P95, MinEig, PSD%, Time per method
- **Additional Stats Table**: Bias, MAE, VarRatio, R² per method
- **Statistics Summary**: Best method (mlp) with full metrics: R²=0.967, VarRatio=1.12
- Per-method timing: aggregate 0.02s, ridge 0.08s, rf 0.29s, lgbm 1.24s, mlp 12.55s

### Eval 03: Ruthless Redesign (3-Regime Testing) → 9/9 PASS
- Complete rewrite to test Lambda across ALL THREE REGIMES with tight tolerances
- **Part A (RCT)**: Gauss-Hermite quadrature oracle, MC convergence rate (√M), Y-independence
- **Part B (Linear)**: Analytical E[TT'|X] oracle, θ-independence, confounded T handling
- **Part C (Observational)**: Tests 5 methods: aggregate, mlp, ridge, rf, lgbm
- **Results: 9/9 PASS** with `method="mlp"` (Corr=0.997, Frob=0.017)
- Method comparison: aggregate 1/3, ridge 2/3, rf 3/3, lgbm 3/3, **mlp 3/3** (best)
- Added RUTHLESS EVALS rule to CLAUDE.md: "Evals are firewalls. They MUST be brutal."

### Eval 01: Multi-Seed Validation + Scale Ratio Diagnostic
- **Multi-seed validation (5 seeds)**: Run 42, 123, 456, 789, 999 by default, report mean ± std
- **Scale ratio diagnostic**: Detects scale identification issues (high corr + high RMSE = scale shift, not bug)
- **Scale-normalized RMSE**: When scale shift detected (ratio std < 0.2), reports RMSE after normalizing
- Pass criteria based on mean metrics across seeds, not single seed
- CLI: `--seeds 42,123,999` and `--single-seed` flags for backwards compatibility
- Addresses binary family (logit, probit) scale non-identifiability
- **Eval 01: 12/12 PASS** (3 seeds, n=2000, epochs=100)

### Eval 02: Autodiff vs Calculus (All Families)
- Enhanced `eval_02_autodiff.py` with 3-part validation
- **Part 1 - Oracle @ Random θ (7 families)**: Linear, Logit, Poisson, NegBin, Gamma, Weibull, Gumbel
  - Closed-form score/Hessian oracles vs torch.func autodiff
  - All errors < 1e-6 (machine precision)
- **Part 2 - Autodiff-Only (5 families)**: Probit, Beta, Gaussian, Tobit, ZIP
  - Verifies gradients and Hessians are finite at random parameter values
- **Part 3 - Fitted θ̂ Validation (7 families)**: Tests at optimum after fitting
  - Generate data from DGP (α*=0.5, β*=0.3), fit via LBFGS
  - Verify Hessian matches oracle at fitted θ̂
  - Verify Hessian is PSD at optimum (all min eigenvalues > 0)
- **Eval 02: 19/19 PASS**

### Eval 01: Auxiliary Parameter Checking
- Updated `compute_recovery_metrics()` to handle any theta_dim dynamically
- For varying params (α, β): check RMSE < 0.3 AND Corr > 0.7
- For constant params (γ, δ): check RMSE < 0.5 only (correlation undefined for flat line)
- Now displays `RMSE(γ)` and `RMSE(δ)` with "(constant, true=X.XX)" annotation
- Critical for Stage 2 inference: wrong σ estimate → wrong Hessian scaling → invalid CIs

### Added 3 New GLM Families: Probit, Beta, ZIP
- **Probit**: Binary classification with normal CDF link Φ(η), theta_dim=2
- **Beta**: Proportions Y∈(0,1) with logit link for mean, fixed precision φ, theta_dim=2
- **ZIP**: Zero-Inflated Poisson mixture model, theta_dim=4 (rate + inflation params)
- All 3 use autodiff for Hessian (closed-form gradients where practical)
- **Eval 01: 12/12 PASS** (9 existing + 3 new families)

### Added GLM Formula Cheat Sheet
- Added `## GLM Family Formulas` table to CLAUDE.md with all 12 families
- Documents Loss, Gradient, Hessian Weight, and θ_dim for each family
- Cross-verified all formulas against standard statistical references

### Fixed NegBin, Gaussian, and Weibull Families
- **NegBin**: Changed from Poisson-like loss to true Negative Binomial NLL using `lgamma` terms
- **Gaussian**: Now estimates sigma via MLE (theta_dim=3 with gamma=log(sigma)), distinct from Linear
- **Weibull DGP bug fix**: `_generate_weibull` was missing `size=` parameter, generating 1 sample broadcast to all n
- Both NegBin/Gaussian pass autodiff verification (gradient/Hessian error < 1e-6)
- Gaussian gamma recovery: RMSE=0.013, σ̂=0.989 vs σ*=1.0
- **Eval 01: 9/9 PASS** (Weibull was 8/9 before fix)

### Multi-Family Parameter Recovery (Eval 01)
- Extended `eval_01_theta.py` to test all 9 families: Linear, Gaussian, Logit, Poisson, NegBin, Gamma, Weibull, Gumbel, Tobit
- Added family-specific DGPs with appropriate coefficient ranges
- All 9/9 families pass parameter recovery with relaxed thresholds (RMSE < 0.3, Corr > 0.7)
- Verified Tobit implementation against Tobias/Purdue lecture notes

### Eval 01 Investigation
- Created `evals/eval.md` documenting learnings from each eval
- **Finding 1**: Default `patience=10` too aggressive, stops training at epoch ~15
- **Finding 2**: Flat loss surface - α and β compensate (Δloss=0.0003 for α+0.1,β-0.1)
- **Finding 3**: n=20k helps but RMSE(β)<0.1 unrealistic for this DGP
- Recommendation: Increase patience to 50+, relax RMSE(β) threshold to 0.2

### Enriched Evaluation Framework
- Added automatic report generation: JSON, Markdown, TXT summary in `evals/reports/`
- Increased default MC simulations from M=20 to M=100 for better coverage estimates
- Added `--M` and `--output-dir` CLI flags to `run_all.py`
- Created `evals/common/` with cross-fitting isolation and numerical stability tests
- Report links printed at end of evaluation run

### Comprehensive 3-Regime Evaluation Framework
- Reorganized `evals/` to support all 3 Lambda regimes with isolated ground-truth tests
- **Regime A (RCT Logit)**: ComputeLambda (Monte Carlo), 4 evals for randomized experiments
- **Regime B (Linear)**: AnalyticLambda (E[TT'|X]), 5 evals including Robinson 1988 closed-form ψ
- **Regime C (Obs Logit)**: EstimateLambda (3-way split), 6 evals for confounded treatment
- Created `evals/dgps/` with 3 DGP definitions and oracle formulas
- Added `python -m evals.run_all --regime [a/b/c]` for regime-specific testing

## 2026-01-12

### Poisson E2E Validation
- Added `tutorials/04_poisson_e2e_test.ipynb` - validates Poisson family across 4 DGPs
- Results: NN Naive 25% coverage, NN IF 100% coverage, SE Ratio 5.1x-6.9x (mean 6.1x)
- Confirms `lambda_method='aggregate'` requirement for Poisson

### Release 0.1.1
- Bumped version to 0.1.1 and published to PyPI
- Simplified logit tutorial: Oracle MC (M=500) + single NN run with IF-based SE
- Added NN Naive vs IF comparison showing SE ratio ~7.5x (naive severely underestimates uncertainty)

### E2E Logit Validation
- Rewrote `tutorials/02_logit_oracle.ipynb` with comprehensive E2E validation
  - Three scenarios: simple (1D linear), complex (5D nonlinear), high-dim (20D, 2 signal)
  - Target μ* = 0.5 (not 0)
  - M=100, N=1000 Monte Carlo validation
- **CRITICAL FINDING**: Logit requires `lambda_method='aggregate'` for stability
  - Default `lambda_method='mlp'` produces negative Hessian eigenvalues → unstable estimates
  - Updated docs/tutorials/logit.md with warning and all code examples
- Added links to README.md (GitHub, PyPI, ReadTheDocs, Papers)

### Release Cleanup
- Fixed version mismatch: docs/conf.py now matches pyproject.toml (0.1.0)
- Fixed pyproject.toml testpaths: `["tests"]` → `["src/deep_inference/tests"]`
- Removed unused `Tuple` import from `src/deep_inference/core/autodiff.py`
- Removed `src2` reference from MANIFEST.in

### Complete Tutorial Coverage (All 8 Families)
- Added `docs/tutorials/gamma.md` - Gamma model for positive right-skewed data
- Added `docs/tutorials/gumbel.md` - Gumbel model for extreme values
- Added `docs/tutorials/weibull.md` - Weibull model for survival analysis
- Updated `docs/tutorials/index.md` with all 8 families

### Archive Cleanup
- Archived broken prototype scripts (basic_usage.py, run_python.py, validation_suite.py) to `archive/prototypes_broken/`
- Archived logs/ directory to `archive/logs_historical/`
- Archived unused MNIST data to `archive/data_mnist/`
- Archived FLM comparison data to `archive/prototypes_flm_comparison_data/`
- Added `logs/` to .gitignore

### Tutorial: Logit Oracle Comparison
- Added `tutorials/02_logit_oracle.ipynb` - validates structural_dml against logistic regression oracle
- DGP: P(Y=1) = sigmoid(α(X) + β(X)·T) with heterogeneous parameters
- Compares: naive vs delta-corrected oracle, neural network with influence functions
- Monte Carlo validation: M=100, N=1000

### Documentation Alignment
- Updated all 21 docs files from `deepstats` to `deep_inference` API
- All code examples now use `from deep_inference import structural_dml`
- Updated validation docs with Python-based MC examples (archived command-line tools)

## 2026-01-09

### Cleanup
- Removed VALIDATION_REPORT.md (content preserved in logs/)

### Tutorial: Linear Oracle Comparison
- Added `tutorials/01_linear_oracle.ipynb` - validates structural_dml against OLS oracle
- DGP: Y = α(X) + β(X)·T + ε with heteroskedastic errors, linear α(X) and β(X)
- Compares: (a) parameter recovery, (b) training diagnostics, (c) bias/variance, (d) coverage/SE calibration
- Demonstrates that naive OLS under-covers (~89%) while delta-corrected OLS and NN achieve valid coverage (~95%)
- Added math verification: delta method derivation, numerical check, independence assumption test

## 2026-01-08

### Proper Package Structure
- Renamed `src2/` to `src/deep_inference/` for standard Python package layout
- **New import**: `from deep_inference import structural_dml`
- PyPI install: `pip install deep-inference`

### Major Refactor: Clean API
- Ported all 8 families: linear, logit, poisson, gamma, gumbel, tobit, negbin, weibull
- Archived old `src/deep_inference/` to `archive/deep_inference_v1/`
- MC infrastructure (run_mc, metrics, logging, dgp) now in archive for reference

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
