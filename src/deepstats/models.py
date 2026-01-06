"""Neural networks and training for structural estimation.

StructuralNet: X → [α(X), β(X)]
NuisanceNet: X → (E[T|X], Var(T|X))
"""

import copy
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# =============================================================================
# Training History
# =============================================================================

@dataclass
class TrainingHistory:
    """Training history with overfitting/underfitting detection and diagnostics.

    Phase 3 Diagnostics (FLM Validation Scorecard):
    - grad_norm: Convergence check (should → 0)
    - beta_std: Heterogeneity curve (should stabilize, not explode)
    - beta_mean: Sanity check (should be stable)
    """
    train_loss: List[float]
    val_loss: List[float]
    best_epoch: int
    status: str  # "ok", "overfit", "underfit"
    # Phase 3: Training diagnostics
    grad_norm: List[float] = None  # ||∇_θ ℓ||₂ per epoch
    beta_std: List[float] = None   # std(β̂(X)) per epoch
    beta_mean: List[float] = None  # mean(β̂(X)) per epoch

    def __post_init__(self):
        # Initialize empty lists if None
        if self.grad_norm is None:
            self.grad_norm = []
        if self.beta_std is None:
            self.beta_std = []
        if self.beta_mean is None:
            self.beta_mean = []

    def save(self, path: str):
        """Save history to JSON."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "TrainingHistory":
        """Load history from JSON."""
        with open(path) as f:
            data = json.load(f)
        return cls(**data)


# =============================================================================
# Structural Network
# =============================================================================

class StructuralNet(nn.Module):
    """X → θ(X) for structural models. Default outputs [α(X), β(X)]."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = None,
        dropout: float = 0.1,
        n_params: int = 2,
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [64, 32]

        self.n_params = n_params

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
        self.param_layer = nn.Linear(prev_dim, n_params)
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns (batch, n_params) structural parameters."""
        h = self.backbone(x)
        return self.param_layer(h)


# =============================================================================
# Structural Network with Separate Backbones
# =============================================================================

class StructuralNetSeparate(nn.Module):
    """Separate backbones for α(X) and β(X).

    Unlike StructuralNet which shares a backbone, this uses independent
    networks for each parameter. This may help when α and β have different
    complexity or depend on different features.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = None,
        dropout: float = 0.1,
        n_params: int = 2,
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [64, 32]

        self.n_params = n_params

        # Independent backbone for α
        self.backbone_alpha = self._make_backbone(input_dim, hidden_dims, dropout)
        self.head_alpha = nn.Linear(hidden_dims[-1], 1)

        # Independent backbone for β
        self.backbone_beta = self._make_backbone(input_dim, hidden_dims, dropout)
        self.head_beta = nn.Linear(hidden_dims[-1], 1)

        self._init_weights()

    def _make_backbone(self, input_dim: int, hidden_dims: List[int], dropout: float) -> nn.Sequential:
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim
        return nn.Sequential(*layers)

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns (batch, 2) with [α(X), β(X)]."""
        alpha = self.head_alpha(self.backbone_alpha(x))
        beta = self.head_beta(self.backbone_beta(x))
        return torch.cat([alpha, beta], dim=1)


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
    early_stopping: bool = True,
    patience: int = 10,
    val_split: float = 0.1,
) -> TrainingHistory:
    """Train structural model with early stopping and overfitting detection.

    Args:
        early_stopping: Enable early stopping with model rollback
        patience: Epochs to wait for improvement before stopping
        val_split: Fraction of data for validation (0 to disable)

    Returns:
        TrainingHistory with train/val losses, best epoch, and status
    """
    device = torch.device("cpu")
    model.to(device)

    # Train/val split
    n = len(X)
    if val_split > 0:
        n_val = int(n * val_split)
        idx = np.random.permutation(n)
        val_idx, train_idx = idx[:n_val], idx[n_val:]
        X_tr, T_tr, Y_tr = X[train_idx], T[train_idx], Y[train_idx]
        X_val, T_val, Y_val = X[val_idx], T[val_idx], Y[val_idx]
    else:
        X_tr, T_tr, Y_tr = X, T, Y
        X_val, T_val, Y_val = None, None, None

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    train_dataset = TensorDataset(
        torch.FloatTensor(X_tr),
        torch.FloatTensor(T_tr),
        torch.FloatTensor(Y_tr),
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    train_losses = []
    val_losses = []
    grad_norms = []      # Phase 3: Convergence check
    beta_stds = []       # Phase 3: Heterogeneity curve
    beta_means = []      # Phase 3: Sanity check
    best_val_loss = float('inf')
    best_epoch = 0
    best_state = None
    epochs_no_improve = 0

    for epoch in range(epochs):
        # Training
        model.train()
        epoch_train_loss = 0.0
        epoch_grad_norms = []  # Track gradient norms per batch
        for bx, bt, by in train_loader:
            bx, bt, by = bx.to(device), bt.to(device), by.to(device)

            optimizer.zero_grad()
            theta = model(bx)
            loss = family.loss(by, bt, theta).mean()
            loss.backward()

            # Phase 3: Track gradient norm (before optimizer step)
            batch_grad_norm = sum(
                p.grad.norm().item() ** 2
                for p in model.parameters()
                if p.grad is not None
            ) ** 0.5
            epoch_grad_norms.append(batch_grad_norm)

            optimizer.step()

            epoch_train_loss += loss.item() * len(bx)

        epoch_train_loss /= len(X_tr)
        train_losses.append(epoch_train_loss)
        grad_norms.append(np.mean(epoch_grad_norms))  # Average gradient norm for epoch

        # Validation
        if X_val is not None:
            model.eval()
            with torch.no_grad():
                X_val_t = torch.FloatTensor(X_val).to(device)
                T_val_t = torch.FloatTensor(T_val).to(device)
                Y_val_t = torch.FloatTensor(Y_val).to(device)
                theta_val = model(X_val_t)
                val_loss = family.loss(Y_val_t, T_val_t, theta_val).mean().item()

                # Phase 3: Track heterogeneity (β_std) and sanity (β_mean)
                beta_val = theta_val[:, 1]  # β(X) is second parameter
                beta_stds.append(beta_val.std().item())
                beta_means.append(beta_val.mean().item())
            val_losses.append(val_loss)

            # Early stopping check
            if early_stopping:
                if val_loss < best_val_loss - 0.001:
                    best_val_loss = val_loss
                    best_epoch = epoch
                    best_state = copy.deepcopy(model.state_dict())
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= patience:
                        if verbose:
                            print(f"Early stopping at epoch {epoch + 1}")
                        break
        else:
            val_losses.append(epoch_train_loss)
            # Phase 3: Track on training data if no val split
            model.eval()
            with torch.no_grad():
                X_tr_t = torch.FloatTensor(X_tr).to(device)
                theta_tr = model(X_tr_t)
                beta_tr = theta_tr[:, 1]
                beta_stds.append(beta_tr.std().item())
                beta_means.append(beta_tr.mean().item())

        if verbose and (epoch + 1) % 20 == 0:
            val_str = f", Val: {val_losses[-1]:.4f}" if X_val is not None else ""
            print(f"Epoch {epoch + 1}/{epochs}, Train: {epoch_train_loss:.4f}{val_str}")

    # Rollback to best model
    if early_stopping and best_state is not None:
        model.load_state_dict(best_state)

    # Detect overfitting/underfitting
    status = _detect_fit_status(train_losses, val_losses)

    return TrainingHistory(
        train_loss=train_losses,
        val_loss=val_losses,
        best_epoch=best_epoch,
        status=status,
        grad_norm=grad_norms,
        beta_std=beta_stds,
        beta_mean=beta_means,
    )


def _detect_fit_status(train_losses: List[float], val_losses: List[float]) -> str:
    """Detect if model is overfitting, underfitting, or ok."""
    if len(train_losses) < 5:
        return "ok"

    final_train = np.mean(train_losses[-5:])
    final_val = np.mean(val_losses[-5:])
    initial_train = np.mean(train_losses[:5])

    # Overfitting: val loss much higher than train loss
    if final_val > final_train * 1.5 and final_val > initial_train:
        return "overfit"

    # Underfitting: train loss barely decreased
    if final_train > initial_train * 0.9:
        return "underfit"

    return "ok"


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
    n_params = getattr(model, 'n_params', 2)
    return StructuralNet(input_dim, hidden_dims, dropout, n_params)


def clone_nuisance(model: NuisanceNet) -> NuisanceNet:
    """Create fresh copy with same architecture."""
    hidden_dims = []
    for layer in model.backbone:
        if isinstance(layer, nn.Linear):
            hidden_dims.append(layer.out_features)
    input_dim = model.backbone[0].in_features
    return NuisanceNet(input_dim, hidden_dims)
