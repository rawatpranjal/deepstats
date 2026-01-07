"""deepstats: Influence function validation for neural network inference.

Implements the Farrell, Liang, Misra (2021, 2025) approach for valid inference
with neural network estimators. NOT DML - neural nets output structural
parameters directly and influence functions correct for regularization bias.

Usage:
    from deepstats import get_dgp, get_family, naive, influence, bootstrap

    # Generate data
    dgp = get_dgp("linear")
    data = dgp.generate(1000)

    # Get family for loss/residual/weights
    family = get_family("linear")

    # Run inference
    mu_hat, se = influence(data.X, data.T, data.Y, family, config)
"""

from .dgp import get_dgp, DGPS, verify_ground_truth
from .families import get_family, FAMILIES
from .models import StructuralNet, NuisanceNet, train_structural, train_nuisance
from .inference import naive, influence, bootstrap, METHODS
from .metrics import compute_metrics, print_table
from .real_data import load_dataset, DATASETS, RealDataResult
from .classical import classical_logit, classical_poisson, classical_ols, ClassicalResult

__all__ = [
    # DGP
    "get_dgp", "DGPS", "verify_ground_truth",
    # Families
    "get_family", "FAMILIES",
    # Models
    "StructuralNet", "NuisanceNet", "train_structural", "train_nuisance",
    # Inference
    "naive", "influence", "bootstrap", "METHODS",
    # Metrics
    "compute_metrics", "print_table",
    # Real data
    "load_dataset", "DATASETS", "RealDataResult",
    # Classical estimators
    "classical_logit", "classical_poisson", "classical_ols", "ClassicalResult",
]

__version__ = "2.0.0"
