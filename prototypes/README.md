# Sanjog Misra Paper Validation

This folder contains a prototype simulation to validate the results from "Deep Learning for Individual Heterogeneity" (Farrell, Liang, Misra).

## Files

*   `validate_misra.py`: Main simulation script.

## Description

The script compares three inference methods for estimating the average partial effect (or average treatment effect) in Linear and Logit models with heterogeneity:

1.  **Naive**: Train a Deep Neural Network to predict the outcome (or parameters), then average the estimated parameter $\hat{\theta}(X)$. Standard errors are computed naively (ignoring estimation uncertainty).
2.  **Influence Function**: Use the influence function correction from Farrell-Liang-Misra.
    *   For Linear: Influence function for ATE.
    *   For Logit: The specific influence function derived in the paper, involving the Jacobian of the structural model and a correction term.
3.  **Bootstrap**: Bootstrap the Naive estimator to get standard errors.

## Running

```bash
python3 validate_misra.py
```

## Expected Results

*   **Linear**:
    *   Naive: Good point estimates (low bias) but poor coverage (SEs too small).
    *   Influence: Good point estimates and Good coverage.
    *   Bootstrap: Better SEs than Naive, but if bias exists, coverage might still be off.
*   **Logit**:
    *   Naive: Biased estimates (regularization bias of DNN). Poor coverage.
    *   Influence: Bias correction should improve the estimate and provide valid coverage (if the propensity/treatment model is well-specified and the denominator is stable).
    *   Bootstrap: Captures variance but not bias. Coverage remains poor.
