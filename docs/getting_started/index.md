# Getting Started

This section will help you get up and running with `deep-inference`.

```{toctree}
:maxdepth: 2
:caption: Getting Started

installation
quickstart
usage
```

## Overview

`deep-inference` provides valid statistical inference for neural network estimators. The core workflow is:

1. **Generate or load data** with covariates $X$, treatment $T$, and outcome $Y$
2. **Select a statistical family** that matches your outcome distribution
3. **Run inference** using the influence function method
4. **Interpret results** with valid confidence intervals

## Why Influence Functions?

Neural networks are powerful function approximators, but naive inference (just averaging predictions) severely underestimates uncertainty. The influence function approach:

- Corrects for regularization bias
- Provides Neyman-orthogonal scores
- Yields valid confidence intervals with proper coverage

## Supported Models

| Family | Use Case | Example |
|--------|----------|---------|
| Linear | Continuous outcomes | Wages, test scores |
| Logit | Binary outcomes | Purchase decisions |
| Poisson | Count data | Patent counts |
| Tobit | Censored data | Labor supply |
| NegBin | Overdispersed counts | Doctor visits |
