"""Dataset generators for testing and examples.

This module provides data generating processes (DGPs) for:
- High-dimensional linear regression
- Poisson count models
- Binary classification
- Non-linear models (to show neural network advantage)
- A/B test data with heterogeneous treatment effects
- Multimodal HTE data (images, text, graphs)

All generators return structured data with true parameters for validation.
"""

from .generators import (
    make_linear_highdim,
    make_poisson_highdim,
    make_binary_highdim,
    make_nonlinear_highdim,
)

from .ab_test import (
    ABTestData,
    make_ab_test,
    make_ab_test_binary,
    make_ab_test_highdim,
)

from .multimodal import (
    ImageHTEData,
    TextHTEData,
    GraphHTEData,
    make_image_hte,
    make_text_hte,
    make_graph_hte,
    multimodal_to_tensors,
)

from .benchmarks import (
    BenchmarkData,
    load_ihdp,
    load_jobs,
    load_twins,
    load_oj,
    load_acic,
    list_benchmarks,
    load_benchmark,
)

from .download import clear_cache

__all__ = [
    # Standard generators
    "make_linear_highdim",
    "make_poisson_highdim",
    "make_binary_highdim",
    "make_nonlinear_highdim",
    # A/B test generators
    "ABTestData",
    "make_ab_test",
    "make_ab_test_binary",
    "make_ab_test_highdim",
    # Multimodal generators
    "ImageHTEData",
    "TextHTEData",
    "GraphHTEData",
    "make_image_hte",
    "make_text_hte",
    "make_graph_hte",
    "multimodal_to_tensors",
    # Benchmark datasets
    "BenchmarkData",
    "load_ihdp",
    "load_jobs",
    "load_twins",
    "load_oj",
    "load_acic",
    "list_benchmarks",
    "load_benchmark",
    "clear_cache",
]
