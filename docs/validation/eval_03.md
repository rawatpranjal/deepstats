# Eval 03: Lambda Estimation

Comparing methods for estimating the conditional Hessian Λ(x) = E[ℓ_θθ|X=x].

## Configuration

| Parameter | Value |
|-----------|-------|
| Sample Size | n = 1000 |
| Seed | 42 |
| Test Points | 500 |
| MC Samples | 5000 |

## Results

| Method | Correlation | Frob Error | Time | Result |
|--------|-------------|------------|------|--------|
| aggregate | 0.000 | 0.121 | 0.02s | 1/3 |
| ridge | 0.508 | 0.087 | 0.08s | 2/3 |
| rf | 0.904 | 0.060 | 0.3s | 3/3 PASS |
| **lgbm** | **0.978** | **0.033** | **1.2s** | **3/3 PASS** |
| **mlp** | **0.997** | **0.018** | 12.5s | **3/3 PASS** |

**Result: 9/9 PASS**

## Key Findings

### aggregate has zero correlation
The `aggregate` method averages Hessians across all observations, ignoring X-dependence entirely. This gives Corr = 0.000 with true Λ(x).

### mlp vs lgbm tradeoff
- **mlp**: Best accuracy (Corr = 0.997) but 10x slower
- **lgbm**: Excellent accuracy (Corr = 0.978) with fast training

### Recommendation
Use `lgbm` as default for best speed/accuracy tradeoff. Use `mlp` when accuracy is critical.

## Run Command

```bash
python3 -m evals.eval_03_lambda 2>&1 | tee evals/reports/eval_03_$(date +%Y%m%d_%H%M%S).txt
```
