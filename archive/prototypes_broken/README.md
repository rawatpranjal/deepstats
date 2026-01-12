# Archived Prototype Scripts

These scripts use the **v1 `deepstats` API** which has been replaced by `deep_inference`.

## Files

- `basic_usage.py` - Used `import deepstats as ds` with old API
- `run_python.py` - Used `from src.deepstats import` which no longer exists

## Current API

```python
# Old (broken)
from deepstats import get_dgp, get_family, influence

# New (working)
from deep_inference import structural_dml

result = structural_dml(Y, T, X, family='linear')
```

## Date Archived
2026-01-12
