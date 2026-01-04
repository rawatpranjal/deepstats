"""Text DGP wrappers for simulation studies.

Wraps the text HTE generator from deepstats.datasets.multimodal.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal

import numpy as np
import pandas as pd

from deepstats.datasets.multimodal import make_text_hte, TextHTEData


@dataclass
class TextDGP:
    """Wrapper for text DGP functions.

    Attributes
    ----------
    name : str
        Name of the DGP scenario.
    heterogeneity : str
        Type of heterogeneity (length, frequency, pattern).
    vocab_size : int
        Vocabulary size.
    seq_length : int
        Sequence length.
    description : str
        Description of the scenario.
    """

    name: str
    heterogeneity: Literal["length", "frequency", "pattern"]
    vocab_size: int = 1000
    seq_length: int = 50
    description: str = ""

    def generate(
        self, seed: int, n: int = 2000, noise_scale: float = 1.0
    ) -> dict:
        """Generate text HTE data.

        Parameters
        ----------
        seed : int
            Random seed.
        n : int, default=2000
            Number of observations.
        noise_scale : float, default=1.0
            Noise scale.

        Returns
        -------
        dict
            Dictionary with text data and ground truth.
        """
        data = make_text_hte(
            n=n,
            vocab_size=self.vocab_size,
            seq_len=self.seq_length,
            heterogeneity=self.heterogeneity,
            noise_scale=noise_scale,
            seed=seed,
        )

        return {
            "tokens": data.tokens,
            "treatment": data.treatment,
            "outcome": data.outcome,
            "true_ate": data.true_ate,
            "true_ite": data.true_ite,
            "text_features": data.text_features,
            "vocab_size": data.vocab_size,
        }


# Registry of text DGPs
TEXT_DGPS: dict[str, TextDGP] = {
    "length": TextDGP(
        name="length",
        heterogeneity="length",
        description="Treatment effect varies with sequence length",
    ),
    "frequency": TextDGP(
        name="frequency",
        heterogeneity="frequency",
        description="Treatment effect varies with word frequency",
    ),
    "pattern": TextDGP(
        name="pattern",
        heterogeneity="pattern",
        description="Treatment effect varies with token patterns",
    ),
}


def get_text_dgp(name: str) -> TextDGP:
    """Get a text DGP by name."""
    if name not in TEXT_DGPS:
        raise ValueError(
            f"Unknown text DGP: {name}. Available: {list(TEXT_DGPS.keys())}"
        )
    return TEXT_DGPS[name]


def list_text_dgps() -> list[str]:
    """List available text DGP names."""
    return list(TEXT_DGPS.keys())
