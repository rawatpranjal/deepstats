# Changelog

## 2026-01-07

- Implemented three-way splitting for nonlinear families (logit, poisson, etc.) where Λ depends on θ
- Added nonparametric Λ(x) estimation via LambdaEstimator class (MLP or LightGBM)
- Made treatment centering configurable (default off to match paper formula)
- Updated variance estimation to use within-fold formula from paper
- Added `lambda_depends_on_theta()` and `hessian_at_point()` methods to all families
- Added Sphinx documentation with Read the Docs support
- Created tutorials for core models: linear, logit, poisson, tobit, negbin
- Added comprehensive theory section with FLM framework and Structural DML algorithm
- Split Theory and Algorithm into separate top-level navigation sections
- Updated documentation with comprehensive FLM framework and algorithm content
- Added practical Usage Guide in Getting Started section
- Set up autodoc API reference generation
