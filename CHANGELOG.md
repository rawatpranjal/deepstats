# Changelog

## 2026-01-07

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
- Set up autodoc API reference generation
