"""Multimodal (Image+Text) DGP wrappers for simulation studies.

Wraps the image+text HTE generator from deepstats.datasets.multimodal.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal

import numpy as np
import pandas as pd

from deepstats.datasets.multimodal import make_image_text_hte, ImageTextHTEData


@dataclass
class MultimodalDGP:
    """Wrapper for multimodal (image+text) DGP functions.

    Attributes
    ----------
    name : str
        Name of the DGP scenario.
    heterogeneity : str
        Type of heterogeneity (image_dominant, text_dominant, interaction, complex).
    image_size : int
        Size of generated images.
    channels : int
        Number of image channels.
    seq_len : int
        Text sequence length.
    vocab_size : int
        Vocabulary size for text.
    description : str
        Description of the scenario.
    """

    name: str
    heterogeneity: Literal["image_dominant", "text_dominant", "interaction", "complex"]
    image_size: int = 32
    channels: int = 3
    seq_len: int = 50
    vocab_size: int = 1000
    description: str = ""

    def generate(
        self, seed: int, n: int = 2000, noise_scale: float = 1.0
    ) -> dict:
        """Generate multimodal HTE data.

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
            Dictionary with multimodal data and ground truth.
        """
        data = make_image_text_hte(
            n=n,
            image_size=self.image_size,
            channels=self.channels,
            seq_len=self.seq_len,
            vocab_size=self.vocab_size,
            heterogeneity=self.heterogeneity,
            noise_scale=noise_scale,
            seed=seed,
        )

        return {
            "images": data.images,
            "tokens": data.tokens,
            "treatment": data.treatment,
            "outcome": data.outcome,
            "true_ate": data.true_ate,
            "true_ite": data.true_ite,
            "image_features": data.image_features,
            "text_features": data.text_features,
            "vocab_size": data.vocab_size,
        }


# Registry of multimodal DGPs
MULTIMODAL_DGPS: dict[str, MultimodalDGP] = {
    "image_dominant": MultimodalDGP(
        name="image_dominant",
        heterogeneity="image_dominant",
        description="Treatment effect dominated by image features",
    ),
    "text_dominant": MultimodalDGP(
        name="text_dominant",
        heterogeneity="text_dominant",
        description="Treatment effect dominated by text features",
    ),
    "interaction": MultimodalDGP(
        name="interaction",
        heterogeneity="interaction",
        description="Treatment effect from image-text interaction",
    ),
    "complex": MultimodalDGP(
        name="complex",
        heterogeneity="complex",
        description="Complex combination of multimodal features",
    ),
}


def get_multimodal_dgp(name: str) -> MultimodalDGP:
    """Get a multimodal DGP by name."""
    if name not in MULTIMODAL_DGPS:
        raise ValueError(
            f"Unknown multimodal DGP: {name}. Available: {list(MULTIMODAL_DGPS.keys())}"
        )
    return MULTIMODAL_DGPS[name]


def list_multimodal_dgps() -> list[str]:
    """List available multimodal DGP names."""
    return list(MULTIMODAL_DGPS.keys())
