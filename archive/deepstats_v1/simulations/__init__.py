"""Simulation studies for HTE estimator evaluation.

This module provides tools for Monte Carlo simulation studies to evaluate
heterogeneous treatment effect estimators, including:
- Overfitting/underfitting diagnosis
- ATE bias and coverage analysis
- ITE quantile accuracy
- Train/validation loss curve analysis

Examples
--------
>>> from deepstats.simulations import SimulationStudy, make_overfit_scenario
>>> from deepstats import DeepHTE
>>>
>>> # Define estimator factory
>>> def make_estimator():
...     return DeepHTE(
...         formula="Y ~ a(X1 + X2 + X3) + b(X1 + X2 + X3) * T",
...         backbone="mlp",
...         epochs=100,
...     )
>>>
>>> # Run simulation study
>>> study = SimulationStudy(
...     dgp=make_overfit_scenario,
...     estimator_factory=make_estimator,
...     n_simulations=50,
... )
>>> results = study.run()
>>> print(results.summary())
"""

from .metrics import (
    compute_ate_metrics,
    compute_ite_metrics,
    compute_quantile_metrics,
    compute_simulation_metrics,
)

from .diagnostics import (
    diagnose_fitting,
    FittingDiagnosis,
    analyze_loss_curves,
)

from .study import (
    SimulationResult,
    SimulationSummary,
    SimulationStudy,
)

from .dgp import (
    make_overfit_scenario,
    make_underfit_scenario,
    make_balanced_scenario,
    make_high_noise_scenario,
    make_sparse_scenario,
)

from .placebo import (
    make_placebo_scenario,
    make_near_zero_scenario,
    run_placebo_test,
    run_placebo_monte_carlo,
    PlaceboTestResult,
)

from .tough_dgp import (
    make_tough_highdim_scenario,
    make_deep_interaction_scenario,
    make_threshold_scenario,
    make_multifreq_scenario,
    make_sparse_nonlinear_scenario,
    make_mixed_tough_scenario,
)

from .dgp_poisson import (
    PoissonSimulationData,
    make_poisson_dgp_lowdim,
    make_poisson_dgp_highdim,
    make_poisson_dgp_nonlinear,
    make_poisson_dgp_high_heterogeneity,
)

from .poisson_study import (
    PoissonSimulationStudy,
    PoissonSimulationResult,
    PoissonSimulationSummary,
)

__all__ = [
    # Metrics
    "compute_ate_metrics",
    "compute_ite_metrics",
    "compute_quantile_metrics",
    "compute_simulation_metrics",
    # Diagnostics
    "diagnose_fitting",
    "FittingDiagnosis",
    "analyze_loss_curves",
    # Study
    "SimulationResult",
    "SimulationSummary",
    "SimulationStudy",
    # DGPs
    "make_overfit_scenario",
    "make_underfit_scenario",
    "make_balanced_scenario",
    "make_high_noise_scenario",
    "make_sparse_scenario",
    # Placebo tests
    "make_placebo_scenario",
    "make_near_zero_scenario",
    "run_placebo_test",
    "run_placebo_monte_carlo",
    "PlaceboTestResult",
    # Tough DGPs
    "make_tough_highdim_scenario",
    "make_deep_interaction_scenario",
    "make_threshold_scenario",
    "make_multifreq_scenario",
    "make_sparse_nonlinear_scenario",
    "make_mixed_tough_scenario",
    # Poisson simulations
    "PoissonSimulationData",
    "PoissonSimulationStudy",
    "PoissonSimulationResult",
    "PoissonSimulationSummary",
    "make_poisson_dgp_lowdim",
    "make_poisson_dgp_highdim",
    "make_poisson_dgp_nonlinear",
    "make_poisson_dgp_high_heterogeneity",
]
