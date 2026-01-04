# deepstats

Influence function validation for neural network inference.
Implements Farrell, Liang, Misra (2021, 2025) approach.

## What This Is

Neural nets output structural parameters directly (NOT DML).
Influence functions correct for regularization bias.

Target: μ* = E[β(X)] with valid 95% confidence intervals.

## Package Structure

```
src/deepstats/
├── dgp.py        # 8 DGPs (linear, gamma, poisson, logit, etc.)
├── families.py   # 8 families with loss/residual/weight
├── models.py     # StructuralNet, NuisanceNet
├── inference.py  # naive, influence, bootstrap methods
├── metrics.py    # MC metrics computation
└── run_mc.py     # Monte Carlo entry point
```

## Quick Start

```python
from src.deepstats import get_dgp, get_family, influence

# Generate data
dgp = get_dgp("linear")
data = dgp.generate(1000)

# Run influence function inference
family = get_family("linear")
# mu_hat, se = influence(data.X, data.T, data.Y, family, config)
```

## Monte Carlo Validation

```bash
# Quick test
python src/deepstats/run_mc.py --M 10 --N 500 --epochs 50 --models linear

# Full simulation
python src/deepstats/run_mc.py --M 50 --N 1000 --epochs 100 --models linear gamma poisson logit
```

## Key Files

- `src/deepstats/CLAUDE.md` - Detailed simulation study spec
- `references/` - Academic papers
- `paper/` - Our paper (LaTeX)
- `prototypes/` - Experiments and validation scripts
- `archive/deepstats_v1/` - Old implementation

## References

- Farrell, Liang, Misra (2021): Econometrica
- Farrell, Liang, Misra (2025): Extended theory
