"""Formula parser for enriched structural models.

This module provides parsing of R-style formulas for heterogeneous
treatment effect models following Farrell, Liang, Misra (2021, 2023).

Formula Syntax
--------------
    Y ~ a(X1 + X2 + X3) + b(X1 + X2) * T

Where:
- Y is the outcome variable
- a(...) specifies covariates for the baseline function
- b(...) * T specifies covariates for the treatment effect function
- T is the treatment variable

The formula is parsed into:
- outcome: name of Y variable
- a_covariates: list of covariate names for a(X)
- b_covariates: list of covariate names for b(X)
- treatment: name of treatment variable T
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .._typing import Float64Array


@dataclass
class ParsedFormula:
    """Result of parsing a formula.

    Attributes
    ----------
    outcome : str
        Name of the outcome variable.
    treatment : str
        Name of the treatment variable.
    a_covariates : list[str]
        Names of covariates for baseline function a(X).
    b_covariates : list[str]
        Names of covariates for treatment effect function b(X).
    formula : str
        Original formula string.
    """

    outcome: str
    treatment: str
    a_covariates: list[str]
    b_covariates: list[str]
    formula: str

    # Extracted data (filled after extracting from DataFrame)
    y: Float64Array | None = field(default=None, repr=False)
    t: Float64Array | None = field(default=None, repr=False)
    X_a: Float64Array | None = field(default=None, repr=False)
    X_b: Float64Array | None = field(default=None, repr=False)
    n_obs: int = 0

    def extract_data(self, data: pd.DataFrame) -> "ParsedFormula":
        """Extract data arrays from DataFrame.

        Parameters
        ----------
        data : pd.DataFrame
            DataFrame containing all variables.

        Returns
        -------
        ParsedFormula
            Self with data arrays filled in.

        Raises
        ------
        KeyError
            If a variable is not found in the DataFrame.
        """
        # Validate all variables exist
        all_vars = [self.outcome, self.treatment] + self.a_covariates + self.b_covariates
        missing = [v for v in all_vars if v not in data.columns]
        if missing:
            raise KeyError(f"Variables not found in data: {missing}")

        # Extract arrays
        self.y = data[self.outcome].values.astype(np.float64)
        self.t = data[self.treatment].values.astype(np.float64)
        self.X_a = data[self.a_covariates].values.astype(np.float64)
        self.X_b = data[self.b_covariates].values.astype(np.float64)
        self.n_obs = len(self.y)

        return self

    @property
    def n_a_covariates(self) -> int:
        """Number of covariates for a(X)."""
        return len(self.a_covariates)

    @property
    def n_b_covariates(self) -> int:
        """Number of covariates for b(X)."""
        return len(self.b_covariates)

    def __str__(self) -> str:
        return (
            f"ParsedFormula(\n"
            f"  outcome={self.outcome},\n"
            f"  treatment={self.treatment},\n"
            f"  a_covariates={self.a_covariates},\n"
            f"  b_covariates={self.b_covariates}\n"
            f")"
        )


class FormulaParser:
    """Parser for enriched structural model formulas.

    Parses formulas of the form:
        Y ~ a(X1 + X2) + b(X1 + X2) * T

    The parser extracts:
    - Outcome variable (Y)
    - Treatment variable (T)
    - Covariates for baseline function a(X)
    - Covariates for treatment effect function b(X)

    Examples
    --------
    >>> parser = FormulaParser()
    >>> result = parser.parse("Y ~ a(X1 + X2) + b(X1 + X2) * T")
    >>> print(result.outcome)
    'Y'
    >>> print(result.a_covariates)
    ['X1', 'X2']
    """

    # Regex patterns for parsing
    FORMULA_PATTERN = re.compile(
        r"^\s*(\w+)\s*~\s*a\((.*?)\)\s*\+\s*b\((.*?)\)\s*\*\s*(\w+)\s*$"
    )
    COVARIATE_PATTERN = re.compile(r"\w+")

    def parse(
        self,
        formula: str,
        data: pd.DataFrame | None = None,
    ) -> ParsedFormula:
        """Parse a formula string.

        Parameters
        ----------
        formula : str
            Formula string of the form "Y ~ a(X1 + X2) + b(X1 + X2) * T".
        data : pd.DataFrame, optional
            If provided, extract data arrays from this DataFrame.

        Returns
        -------
        ParsedFormula
            Parsed formula with extracted components.

        Raises
        ------
        ValueError
            If the formula doesn't match the expected pattern.

        Examples
        --------
        >>> parser = FormulaParser()
        >>> result = parser.parse("wage ~ a(edu + exp) + b(edu + exp) * training")
        >>> print(result.outcome)
        'wage'
        >>> print(result.treatment)
        'training'
        """
        # Clean up the formula
        formula = formula.strip()

        # Try to match the pattern
        match = self.FORMULA_PATTERN.match(formula)

        if not match:
            raise ValueError(
                f"Invalid formula: '{formula}'\n"
                f"Expected format: Y ~ a(X1 + X2 + ...) + b(X1 + X2 + ...) * T\n"
                f"Example: wage ~ a(edu + exp) + b(edu + exp) * training"
            )

        outcome = match.group(1)
        a_covariates_str = match.group(2)
        b_covariates_str = match.group(3)
        treatment = match.group(4)

        # Extract covariate names
        a_covariates = self._parse_covariates(a_covariates_str)
        b_covariates = self._parse_covariates(b_covariates_str)

        result = ParsedFormula(
            outcome=outcome,
            treatment=treatment,
            a_covariates=a_covariates,
            b_covariates=b_covariates,
            formula=formula,
        )

        # Extract data if provided
        if data is not None:
            result.extract_data(data)

        return result

    def _parse_covariates(self, covariates_str: str) -> list[str]:
        """Parse covariate string into list of variable names.

        Parameters
        ----------
        covariates_str : str
            String like "X1 + X2 + X3".

        Returns
        -------
        list[str]
            List of variable names.
        """
        # Find all word tokens (variable names)
        covariates = self.COVARIATE_PATTERN.findall(covariates_str)
        return covariates

    @staticmethod
    def validate_formula(formula: str) -> bool:
        """Check if a formula is valid.

        Parameters
        ----------
        formula : str
            Formula string to validate.

        Returns
        -------
        bool
            True if the formula is valid.
        """
        parser = FormulaParser()
        try:
            parser.parse(formula)
            return True
        except ValueError:
            return False


def parse_formula(
    formula: str,
    data: pd.DataFrame | None = None,
) -> ParsedFormula:
    """Convenience function to parse a formula.

    Parameters
    ----------
    formula : str
        Formula string.
    data : pd.DataFrame, optional
        If provided, extract data arrays.

    Returns
    -------
    ParsedFormula
        Parsed formula.

    Examples
    --------
    >>> result = parse_formula("Y ~ a(X1 + X2) + b(X1 + X2) * T")
    >>> print(result.outcome)
    'Y'
    """
    return FormulaParser().parse(formula, data)
