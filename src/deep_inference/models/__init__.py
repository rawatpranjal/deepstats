"""Neural network models and structural model protocols."""

from .structural_net import StructuralNet, train_structural_net, TrainingHistory
from .base import (
    StructuralModel,
    BaseModel,
    CustomModel,
    ModelMetadata,
    model_from_loss,
)
from .linear import LinearModel, Linear
from .logit import LogitModel, Logit

__all__ = [
    # Neural network
    "StructuralNet",
    "train_structural_net",
    "TrainingHistory",
    # Model protocol
    "StructuralModel",
    "BaseModel",
    "CustomModel",
    "ModelMetadata",
    "model_from_loss",
    # Built-in models
    "LinearModel",
    "Linear",
    "LogitModel",
    "Logit",
]
