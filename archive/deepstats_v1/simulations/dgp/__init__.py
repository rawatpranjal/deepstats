"""Data Generating Processes for simulation studies.

This module provides wrappers around the various DGP implementations
in deepstats to provide a unified interface for simulation runners.
"""

from .tabular import TabularDGP, get_tabular_dgp
from .image import ImageDGP, get_image_dgp
from .text import TextDGP, get_text_dgp
from .graph import GraphDGP, get_graph_dgp
from .timeseries import TimeSeriesDGP, get_timeseries_dgp
from .multimodal import MultimodalDGP, get_multimodal_dgp

__all__ = [
    "TabularDGP",
    "ImageDGP",
    "TextDGP",
    "GraphDGP",
    "TimeSeriesDGP",
    "MultimodalDGP",
    "get_tabular_dgp",
    "get_image_dgp",
    "get_text_dgp",
    "get_graph_dgp",
    "get_timeseries_dgp",
    "get_multimodal_dgp",
]
