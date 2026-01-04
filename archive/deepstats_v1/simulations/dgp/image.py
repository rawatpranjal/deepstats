"""Image DGP wrappers for simulation studies.

Wraps the image HTE generator from deepstats.datasets.multimodal.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal

import numpy as np
import pandas as pd

from deepstats.datasets.multimodal import make_image_hte, ImageHTEData


@dataclass
class ImageDGP:
    """Wrapper for image DGP functions.

    Attributes
    ----------
    name : str
        Name of the DGP scenario.
    heterogeneity : str
        Type of heterogeneity (brightness, texture, color, complex).
    image_size : int
        Size of generated images.
    description : str
        Description of the scenario.
    """

    name: str
    heterogeneity: Literal["brightness", "texture", "color", "complex"]
    image_size: int = 32
    channels: int = 3
    description: str = ""

    def generate(
        self, seed: int, n: int = 2000, noise_scale: float = 1.0
    ) -> dict:
        """Generate image HTE data.

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
            Dictionary with image data and ground truth.
        """
        data = make_image_hte(
            n=n,
            image_size=self.image_size,
            channels=self.channels,
            heterogeneity=self.heterogeneity,
            noise_scale=noise_scale,
            seed=seed,
        )

        return {
            "images": data.images,
            "treatment": data.treatment,
            "outcome": data.outcome,
            "true_ate": data.true_ate,
            "true_ite": data.true_ite,
            "image_features": data.image_features,
        }


# Registry of image DGPs
IMAGE_DGPS: dict[str, ImageDGP] = {
    "brightness": ImageDGP(
        name="brightness",
        heterogeneity="brightness",
        description="Treatment effect varies with image brightness",
    ),
    "texture": ImageDGP(
        name="texture",
        heterogeneity="texture",
        description="Treatment effect varies with edge density",
    ),
    "color": ImageDGP(
        name="color",
        heterogeneity="color",
        description="Treatment effect varies with color distribution",
    ),
    "complex": ImageDGP(
        name="complex",
        heterogeneity="complex",
        description="Complex combination of image features",
    ),
}


def get_image_dgp(name: str) -> ImageDGP:
    """Get an image DGP by name."""
    if name not in IMAGE_DGPS:
        raise ValueError(
            f"Unknown image DGP: {name}. Available: {list(IMAGE_DGPS.keys())}"
        )
    return IMAGE_DGPS[name]


def list_image_dgps() -> list[str]:
    """List available image DGP names."""
    return list(IMAGE_DGPS.keys())
