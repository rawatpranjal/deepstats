"""Structural neural network for parameter estimation."""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Callable, List, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np


@dataclass
class TrainingHistory:
    """Training history for monitoring convergence."""

    train_losses: List[float] = field(default_factory=list)
    val_losses: List[float] = field(default_factory=list)
    grad_norms: List[float] = field(default_factory=list)
    best_epoch: int = 0
    best_val_loss: float = float('inf')

    def add(self, train_loss: float, val_loss: float, grad_norm: float = 0.0):
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.grad_norms.append(grad_norm)

        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_epoch = len(self.val_losses) - 1


class StructuralNet(nn.Module):
    """
    Neural network mapping X -> theta(X).

    Architecture:
        Input(d_x) -> Hidden(h1) -> ReLU -> Dropout -> ... -> Output(theta_dim)

    The network learns the structural parameters theta(x) = (alpha(x), beta(x), ...)
    as functions of covariates x.
    """

    def __init__(
        self,
        input_dim: int,
        theta_dim: int,
        hidden_dims: List[int] = [64, 32],
        dropout: float = 0.1,
    ):
        """
        Initialize structural network.

        Args:
            input_dim: Dimension of input covariates X
            theta_dim: Dimension of output parameters theta
            hidden_dims: List of hidden layer sizes
            dropout: Dropout probability
        """
        super().__init__()

        self.input_dim = input_dim
        self.theta_dim = theta_dim
        self.hidden_dims = hidden_dims

        # Build network
        layers = []
        prev_dim = input_dim

        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h

        layers.append(nn.Linear(prev_dim, theta_dim))

        self.network = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Xavier uniform."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass.

        Args:
            x: (n, input_dim) input covariates

        Returns:
            (n, theta_dim) predicted parameters
        """
        return self.network(x)


def train_structural_net(
    model: StructuralNet,
    X: Tensor,
    T: Tensor,
    Y: Tensor,
    loss_fn: Callable[[Tensor, Tensor, Tensor], Tensor],
    epochs: int = 100,
    lr: float = 0.01,
    val_frac: float = 0.1,
    patience: int = 10,
    batch_size: Optional[int] = None,
    verbose: bool = False,
) -> TrainingHistory:
    """
    Train structural network with early stopping.

    Args:
        model: StructuralNet to train
        X: (n, d_x) covariates
        T: (n,) treatments
        Y: (n,) outcomes
        loss_fn: Structural loss function (y, t, theta) -> (n,) losses
        epochs: Maximum training epochs
        lr: Learning rate
        val_frac: Fraction for validation
        patience: Early stopping patience
        batch_size: Mini-batch size (None for full batch)
        verbose: Print progress

    Returns:
        TrainingHistory with training metrics
    """
    n = len(Y)
    history = TrainingHistory()

    # Train/val split
    n_val = max(1, int(n * val_frac))
    perm = torch.randperm(n)
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]

    X_train, T_train, Y_train = X[train_idx], T[train_idx], Y[train_idx]
    X_val, T_val, Y_val = X[val_idx], T[val_idx], Y[val_idx]

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # For early stopping
    best_val_loss = float('inf')
    best_state = None
    epochs_without_improvement = 0

    # Batch size
    if batch_size is None or batch_size >= len(X_train):
        batch_size = len(X_train)

    for epoch in range(epochs):
        model.train()

        # Shuffle training data
        perm_train = torch.randperm(len(X_train))
        total_train_loss = 0.0
        n_batches = 0

        # Mini-batch training
        for i in range(0, len(X_train), batch_size):
            batch_idx = perm_train[i:i+batch_size]
            X_batch = X_train[batch_idx]
            T_batch = T_train[batch_idx]
            Y_batch = Y_train[batch_idx]

            optimizer.zero_grad()

            theta_batch = model(X_batch)
            losses = loss_fn(Y_batch, T_batch, theta_batch)
            loss = losses.mean()

            loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)

            optimizer.step()

            total_train_loss += loss.item() * len(batch_idx)
            n_batches += 1

        train_loss = total_train_loss / len(X_train)

        # Validation
        model.eval()
        with torch.no_grad():
            theta_val = model(X_val)
            val_losses = loss_fn(Y_val, T_val, theta_val)
            val_loss = val_losses.mean().item()

        # Compute gradient norm (for diagnostics)
        grad_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                grad_norm += p.grad.norm().item() ** 2
        grad_norm = np.sqrt(grad_norm)

        history.add(train_loss, val_loss, grad_norm)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if verbose and (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}: train={train_loss:.4f}, val={val_loss:.4f}")

        if epochs_without_improvement >= patience:
            if verbose:
                print(f"Early stopping at epoch {epoch+1}")
            break

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)

    return history


def clone_model(model: StructuralNet) -> StructuralNet:
    """Create a copy of a structural network."""
    new_model = StructuralNet(
        input_dim=model.input_dim,
        theta_dim=model.theta_dim,
        hidden_dims=model.hidden_dims,
    )
    new_model.load_state_dict(model.state_dict())
    return new_model
