"""Neural networks and training for structural estimation.

StructuralNet: X → [α(X), β(X)]
NuisanceNet: X → (E[T|X], Var(T|X))
"""

from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# =============================================================================
# Structural Network
# =============================================================================

class StructuralNet(nn.Module):
    """X → [α(X), β(X)] for the structural model Y = α(X) + β(X)*T + noise."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [64, 32]

        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim

        self.backbone = nn.Sequential(*layers)
        self.param_layer = nn.Linear(prev_dim, 2)
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns (batch, 2) where [:, 0]=α, [:, 1]=β."""
        h = self.backbone(x)
        return self.param_layer(h)


# =============================================================================
# Nuisance Network
# =============================================================================

class NuisanceNet(nn.Module):
    """X → (E[T|X], Var(T|X)) for conditional treatment distribution."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = None,
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [32, 16]

        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
            ])
            prev_dim = hidden_dim

        self.backbone = nn.Sequential(*layers)
        self.out = nn.Linear(prev_dim, 2)
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (mean, variance) each of shape (batch,)."""
        h = self.backbone(x)
        out = self.out(h)
        mean = out[:, 0]
        var = torch.nn.functional.softplus(out[:, 1]) + 1e-6
        return mean, var

    def predict_mean(self, x: torch.Tensor) -> torch.Tensor:
        mean, _ = self.forward(x)
        return mean


# =============================================================================
# Training Functions
# =============================================================================

def train_structural(
    model: StructuralNet,
    family,
    X: np.ndarray,
    T: np.ndarray,
    Y: np.ndarray,
    epochs: int = 100,
    lr: float = 0.01,
    batch_size: int = 64,
    weight_decay: float = 1e-4,
    verbose: bool = False,
) -> List[float]:
    """Train structural model with family-specific loss."""
    device = torch.device("cpu")
    model.to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    dataset = TensorDataset(
        torch.FloatTensor(X),
        torch.FloatTensor(T),
        torch.FloatTensor(Y),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    loss_history = []
    for epoch in range(epochs):
        epoch_loss = 0.0
        for bx, bt, by in loader:
            bx, bt, by = bx.to(device), bt.to(device), by.to(device)

            optimizer.zero_grad()
            theta = model(bx)
            loss = family.loss(by, bt, theta).mean()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * len(bx)

        epoch_loss /= len(X)
        loss_history.append(epoch_loss)

        if verbose and (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}")

    return loss_history


def train_nuisance(
    model: NuisanceNet,
    X: np.ndarray,
    T: np.ndarray,
    epochs: int = 50,
    lr: float = 0.01,
    batch_size: int = 64,
    weight_decay: float = 1e-4,
    verbose: bool = False,
) -> List[float]:
    """Train nuisance model for E[T|X], Var(T|X) using Gaussian NLL."""
    device = torch.device("cpu")
    model.to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    dataset = TensorDataset(torch.FloatTensor(X), torch.FloatTensor(T))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    loss_history = []
    for epoch in range(epochs):
        epoch_loss = 0.0
        for bx, bt in loader:
            bx, bt = bx.to(device), bt.to(device)

            optimizer.zero_grad()
            mean, var = model(bx)
            loss = 0.5 * (torch.log(var) + (bt - mean)**2 / var).mean()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * len(bx)

        epoch_loss /= len(X)
        loss_history.append(epoch_loss)

        if verbose and (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}")

    return loss_history


def clone_structural(model: StructuralNet) -> StructuralNet:
    """Create fresh copy with same architecture."""
    hidden_dims = []
    for layer in model.backbone:
        if isinstance(layer, nn.Linear):
            hidden_dims.append(layer.out_features)
    input_dim = model.backbone[0].in_features
    dropout = 0.1
    for layer in model.backbone:
        if isinstance(layer, nn.Dropout):
            dropout = layer.p
            break
    return StructuralNet(input_dim, hidden_dims, dropout)


def clone_nuisance(model: NuisanceNet) -> NuisanceNet:
    """Create fresh copy with same architecture."""
    hidden_dims = []
    for layer in model.backbone:
        if isinstance(layer, nn.Linear):
            hidden_dims.append(layer.out_features)
    input_dim = model.backbone[0].in_features
    return NuisanceNet(input_dim, hidden_dims)
