# Metrics Module

Monte Carlo metrics computation and reporting.

## Main Functions

### compute_metrics

```{eval-rst}
.. autofunction:: deepstats.compute_metrics
```

### print_table

```{eval-rst}
.. autofunction:: deepstats.print_table
```

## Usage Example

```python
from deepstats import compute_metrics, print_table
import pandas as pd

# After running Monte Carlo simulations
# results_df has columns: sim_id, model, method, mu_hat, se, mu_true, covered

metrics = compute_metrics(results_df)
print_table(metrics)
```

## Computed Metrics

| Metric | Formula | Target |
|--------|---------|--------|
| `bias` | $E[\hat\mu] - \mu^*$ | 0 |
| `variance` | $\text{Var}(\hat\mu)$ | - |
| `rmse` | $\sqrt{\text{Bias}^2 + \text{Var}}$ | Small |
| `empirical_se` | $\sqrt{\text{Var}}$ | - |
| `se_mean` | $E[\hat{SE}]$ | - |
| `se_ratio` | $\hat{SE} / SE_{emp}$ | 1.0 |
| `ci_width` | $2 \times 1.96 \times \hat{SE}$ | - |
| `coverage` | $P(\mu^* \in CI)$ | 95% |

## Output Format

### Console Output

```
==================================================================
MONTE CARLO RESULTS
==================================================================
Model    Method      mu*    Bias    Var    RMSE  SE(emp) SE(est) Ratio Coverage
------------------------------------------------------------------
linear   naive     0.001  -0.005  0.015  0.123   0.124   0.023  0.19    30.0%
linear   influence 0.001  -0.007  0.008  0.091   0.091   0.067  0.73    80.0%
...
==================================================================
```

### CSV Output

The `run_mc.py` script outputs two CSV files:

1. **Raw results** (`mc_results.csv`):
   - One row per simulation
   - Columns: sim_id, model, method, mu_hat, se, mu_true, bias, covered

2. **Aggregated metrics** (`mc_results.metrics.csv`):
   - One row per model/method combination
   - All computed metrics

## Validation Scorecard

The metrics module also produces a validation scorecard:

```
VALIDATION SCORECARD
====================
Coverage:    PASS (95.0% in [93%, 97%])
SE Ratio:    PASS (1.02 in [0.9, 1.2])
RMSE:        PASS (below threshold)
Hessian:     PASS (min eigenvalue > 1e-4)

Overall:     PASS
```

## Diagnostic Plots

```python
import matplotlib.pyplot as plt
from deepstats.metrics import plot_coverage, plot_se_ratio

# Coverage comparison
plot_coverage(metrics)
plt.savefig("coverage.png")

# SE ratio by model
plot_se_ratio(metrics)
plt.savefig("se_ratio.png")
```
