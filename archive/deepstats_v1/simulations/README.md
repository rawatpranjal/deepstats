# DeepHTE Simulation Studies

This directory contains the organized simulation framework for evaluating DeepHTE against comparison methods.

## Quick Start

```bash
# Run all simulations (20 reps each modality)
python simulations/run_all.py --all

# Run only tabular simulations
python simulations/run_all.py --tabular

# Run specific DGPs
python simulations/run_all.py --tabular --dgp mixed sparse_nonlinear

# List all available DGPs
python simulations/run_all.py --list-dgps
```

## Directory Structure

```
simulations/
├── config/           # YAML configuration files
│   ├── base.yaml     # Default parameters
│   ├── tabular.yaml  # Tabular-specific settings
│   ├── image.yaml    # Image simulation settings
│   ├── text.yaml     # Text simulation settings
│   └── graph.yaml    # Graph simulation settings
├── dgp/              # Data Generating Processes
│   ├── tabular.py    # Tabular DGP wrappers
│   ├── image.py      # Image DGP wrappers
│   ├── text.py       # Text DGP wrappers
│   └── graph.py      # Graph DGP wrappers
├── runners/          # Simulation runners
│   ├── base_runner.py     # Abstract base class
│   ├── tabular_runner.py  # Tabular simulations
│   ├── image_runner.py    # Image simulations
│   ├── text_runner.py     # Text simulations
│   └── graph_runner.py    # Graph simulations
├── analysis/         # Analysis and visualization
│   ├── ate_analysis.py      # ATE distribution analysis
│   ├── coverage_analysis.py # Coverage computation
│   ├── quantile_analysis.py # Quantile accuracy
│   └── figures.py           # Figure generation
├── results/          # Output directory (gitignored)
└── run_all.py        # Master runner script
```

## Methods Compared

- **DeepHTE**: Deep learning for heterogeneous treatment effects
- **CausalForest**: EconML CausalForestDML
- **LinearDML**: EconML LinearDML (linear CATE assumption)
- **QuantileForest**: Quantile forest for quantile comparison

## Available DGPs

### Tabular
- `mixed`: Mixed linear/nonlinear pattern
- `sparse_nonlinear`: Sparse nonlinear effects
- `overfit`: Small n, high p (overfitting test)
- `underfit`: Large n, complex function
- `high_dimensional`: 100+ covariates

### Image
- `brightness`: Effect varies with image brightness
- `texture`: Effect varies with edge density
- `color`: Effect varies with color distribution
- `complex`: Combination of features

### Text
- `length`: Effect varies with sequence length
- `frequency`: Effect varies with word frequency
- `pattern`: Effect varies with token patterns

### Graph
- `density`: Effect varies with graph density
- `size`: Effect varies with graph size
- `centrality`: Effect varies with centrality

## Metrics

### ATE Estimation
- Bias
- RMSE
- Coverage (95% CI)
- SE calibration ratio

### ITE Estimation
- RMSE
- Correlation with true effects
- Rank correlation

### Quantiles
- Bias per quantile
- RMSE per quantile

## Reproducibility

All simulations are fully reproducible via the `--seed` flag:

```bash
python simulations/run_all.py --all --seed 42 --n_reps 100
```

Results are saved with full configuration metadata in JSON format.

## Output Files

Each simulation produces:
- `{modality}/{dgp}.csv`: Results in tabular format
- `{modality}/{dgp}.pkl`: Full results object (pickle)
- `{modality}/{dgp}.json`: Configuration metadata
