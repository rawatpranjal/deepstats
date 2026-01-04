# deepstats

**Influence Function Validation for Neural Network Inference**

Implements the Farrell, Liang, Misra (2021, 2025) approach for valid statistical inference with neural network estimators.

## Key Insight

Neural networks introduce regularization bias that causes standard errors to be underestimated. This leads to confidence intervals that are too narrow and poor coverage (30-50% instead of 95%).

The **influence function correction** fixes this by accounting for the bias in the neural network estimator.

## Installation

```bash
pip install -e .
```

## Quick Example

```python
from src.deepstats import get_dgp, get_family, verify_ground_truth

# Verify ground truth for all models
verify_ground_truth()

# Generate data
dgp = get_dgp("linear")
data = dgp.generate(1000)
print(f"True μ* = {data.mu_true:.4f}")
```

## Monte Carlo Validation

The package validates the influence function approach across 8 model families:

| Family | Link | Expected Coverage |
|--------|------|------------------|
| linear | identity | ~95% |
| gamma | log | ~90% |
| poisson | log | ~98% |
| logit | logit | ~98% |
| tobit | identity | ~85% |
| negbin | log | ~92% |
| weibull | log | ~90% |
| gumbel | identity | ~85% |

Run simulations:
```bash
python src/deepstats/run_mc.py --M 50 --N 1000 --models linear poisson logit
```

## Repository Structure

```
src/deepstats/     # Main package
references/        # Academic papers
paper/             # Our paper (LaTeX)
prototypes/        # Experiments
archive/           # Old implementation
```

## References

- Farrell, Liang, Misra (2021): "Deep Neural Networks for Estimation and Inference" *Econometrica*
- Farrell, Liang, Misra (2025): Extended inference theory

## License

MIT
