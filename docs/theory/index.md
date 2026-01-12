# Theory

Mathematical foundations of the Farrell-Liang-Misra framework for deep learning with valid inference.

```{toctree}
:maxdepth: 2

influence_functions
```

## Overview

This section explains the theoretical foundations of `deep-inference`, specifically the enriched structural model approach from Farrell, Liang, and Misra.

## Key References

- Farrell, Liang, Misra (2021): "Deep Neural Networks for Estimation and Inference" *Econometrica*
- Farrell, Liang, Misra (2025): "Deep Learning for Individual Heterogeneity" *Working Paper*

## The Core Insight

**Machine learning and economic structure are complements, not substitutes.**

- **ML alone** fits data well but extrapolates nonsensically and can't answer causal questions
- **Structure alone** provides interpretability but misses heterogeneity
- **Combined**: ML learns heterogeneity patterns $\theta(X)$ while structure ensures valid economics
