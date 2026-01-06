# deepstats

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
src/deepstats/
├── dgp.py        # 8 DGPs (linear, gamma, poisson, logit, etc.)
├── families.py   # 8 families with loss/residual/weight
├── models.py     # StructuralNet, NuisanceNet, TrainingHistory
├── inference.py  # naive, influence, bootstrap methods
├── metrics.py    # MC metrics computation + console output
├── logging.py    # Comprehensive JSON logging for AI analysis
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
python -m deepstats.run_mc --M 10 --N 500 --epochs 50 --models linear

# Full simulation
python -m deepstats.run_mc --M 500 --N 2000 --epochs 100 --n-folds 50 \
  --models linear --methods naive influence --n-jobs -1
```

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

- `src/deepstats/CLAUDE.md` - Detailed simulation study spec
- `references/` - Academic papers
- `paper/` - Our paper (LaTeX)
- `prototypes/` - Experiments and validation scripts
- `archive/deepstats_v1/` - Old implementation

## References

- Farrell, Liang, Misra (2021): Econometrica
- Farrell, Liang, Misra (2025): Extended theory
