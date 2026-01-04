"""Formula parsing for enriched structural models.

This module provides R-style formula parsing for specifying
heterogeneous treatment effect models.

Formula Syntax
--------------
The formula syntax supports specifying parameter functions:

    Y ~ a(X1 + X2) + b(X1 + X2) * T

Components:
- Y: Outcome variable (LHS)
- a(...): Baseline function covariates
- b(...) * T: Treatment effect function with treatment variable

Examples
--------
>>> from deepstats.formula import FormulaParser
>>> parser = FormulaParser()
>>> result = parser.parse("Y ~ a(X1 + X2) + b(X1 + X2) * T", data)
>>> print(result.outcome)
'Y'
>>> print(result.treatment)
'T'
>>> print(result.a_covariates)
['X1', 'X2']
"""

from .parser import FormulaParser, ParsedFormula

__all__ = ["FormulaParser", "ParsedFormula"]
