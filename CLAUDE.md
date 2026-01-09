# deep-inference

# Core Rules

**Always follow these rules:**

1. **Always `git push` when done** - Push changes after completing work
2. **Never remove content** - Content removed is content lost
   - If something seems outdated, find a new home for it
   - Archive to `archive/` rather than delete
3. **Update `CHANGELOG.md`** after making changes - 1-2 line summary per day

---

Influence function validation for neural network inference.
Implements Farrell, Liang, Misra (2021, 2025) approach.

## Rules for Claude

- Show all the statistics. 
- NO overrides, placeholders, or deviating from the plan - no matter how hard it gets
- Follow the plan exactly as specified
- If something doesn't work, fix it properly instead of using workarounds

## What This Is

Neural nets output structural parameters directly (NOT DML).
Influence functions correct for regularization bias.

Target: μ* = E[β(X)] with valid 95% confidence intervals.

## Package Structure

```
src2/
├── __init__.py           # Main API: structural_dml()
├── core/
│   ├── algorithm.py      # DML core algorithm
│   ├── autodiff.py       # Gradient/Hessian computation
│   └── lambda_estimator.py
├── families/
│   ├── base.py           # BaseFamily protocol
│   ├── linear.py
│   ├── logit.py
│   ├── poisson.py
│   ├── gamma.py
│   ├── gumbel.py
│   ├── tobit.py
│   ├── negbin.py
│   └── weibull.py
├── models/
│   └── structural_net.py
└── utils/
    └── linalg.py

archive/deep_inference_v1/  # Old implementation (MC tools, DGPs, etc.)
```

## Quick Start

```python
import numpy as np
from src2 import structural_dml

# Generate data
np.random.seed(42)
n = 1000
X = np.random.randn(n, 5)
T = np.random.randn(n)
Y = X[:, 0] + 0.5 * T + np.random.randn(n)

# Run influence function inference
result = structural_dml(Y, T, X, family='linear', epochs=50, n_folds=50)
print(f"Estimate: {result.mu_hat:.4f} +/- {result.se:.4f}")
print(f"95% CI: [{result.ci_lower:.4f}, {result.ci_upper:.4f}]")
```

## Monte Carlo Tools (Archived)

MC validation tools are archived in `archive/deep_inference_v1/`:
- `dgp.py` - 8 DGPs
- `run_mc.py` - Monte Carlo entry point
- `metrics.py` - Metrics computation
- `logging.py` - JSON logging

For validation studies, use scripts in `prototypes/`.

## Comprehensive Logging

Every run generates TWO log files in `--log-dir` (default: `logs/`):

**Full paths (example with `--log-dir logs`):**
```
/Users/pranjal/deepest/logs/mc_run_20260105_132821.log          # JSON (all data)
/Users/pranjal/deepest/logs/mc_run_20260105_132821_readable.txt # Human summary
```

**File descriptions:**
1. **`.log`** - Machine-readable JSON containing ALL data (for AI/programmatic analysis)
2. **`_readable.txt`** - Human-readable summary (for quick review)

### Log Contents (JSON structure)

```json
{
  "meta": {"generated": "...", "version": "1.0", "framework": "FLM"},
  "config": {"M": 500, "N": 2000, "n_folds": 50, "epochs": 100, ...},
  "dgp": {"alpha_star": "...", "beta_star": "...", "mu_true": -0.168},
  "phase1_recovery": {"naive": {"rmse_alpha": X, "rmse_beta": X, ...}},
  "phase2_inference": {"naive": {"coverage": X, "se_ratio": X, ...}},
  "phase3_diagnostics": {"influence": {"correction_ratio": X, ...}},
  "training_quality": {"influence": {"val_loss": X, "train_val_gap": X, ...}},
  "validation_scorecard": {"coverage_pass": true, "overall_grade": "PASS"},
  "raw_data": [...],  // All M*2 simulation results
  "timing": {"total_seconds": X, "per_sim_seconds": X}
}
```

### Console Output Tables

The run prints 4 tables:
1. **PHASE 1: PARAMETER RECOVERY** - RMSE_α, RMSE_β, Corr_α, Corr_β
2. **PHASE 2: INFERENCE** - μ*, Bias, Var, SE(emp), SE(est), Ratio, Coverage
3. **PHASE 3: TRAINING DIAGNOSTICS** - Grad→0, β_std, R_corr, min(Λ), cond(Λ)
4. **TRAINING QUALITY** - ValLoss, TrainLoss, Gap, BestEpoch, Overfit%

Plus a **VALIDATION SCORECARD SUMMARY** with pass/fail grades.

### Key Metrics to Check

| Metric | Target | What it Tests |
|--------|--------|---------------|
| Coverage | 93-97% | Valid confidence intervals |
| SE Ratio | 0.9-1.2 | SE calibration |
| R_corr | 0.1-1.0 | IF correction magnitude |
| min(Λ) | > 1e-4 | Hessian stability |
| train_val_gap | ≈ 0 | No overfitting |
| Violation Rate | 30-70% | IF is necessary |

### Reading Logs Programmatically

```python
import json

# Load full report
with open("logs/mc_run_20260105_132821.log") as f:
    report = json.load(f)

# Access specific sections
print(report["phase2_inference"]["influence"]["coverage"])  # 0.95
print(report["validation_scorecard"]["overall_grade"])       # "PASS"

# Get all raw simulation results
for sim in report["raw_data"]:
    print(f"Sim {sim['sim_id']}: {sim['method']} coverage={sim['covered']}")
```

## Key Files

- `src2/` - Main package (structural_dml API)
- `archive/deep_inference_v1/CLAUDE.md` - Detailed simulation study spec
- `references/` - Academic papers
- `paper/` - Our paper (LaTeX)
- `prototypes/` - Experiments and validation scripts
- `archive/deepstats_v1/` - Older implementation (v1)
- `archive/deep_inference_v1/` - Previous implementation (v2, MC tools)

## References

- Farrell, Liang, Misra (2021): Econometrica
- Farrell, Liang, Misra (2025): Extended theory
