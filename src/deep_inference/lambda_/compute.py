"""
Regime A: Compute Lambda via Monte Carlo integration.

For randomized experiments where:
- Treatment distribution F_T is known
- Hessian doesn't depend on Y

Λ(x) = ∫ ℓ_θθ(t, θ̂(x)) dF_T(t)
     ≈ (1/M) Σ_m ℓ_θθ(t_m, θ̂(x))

This is COMPUTED, not estimated. No additional neural network needed.
"""

from typing import Optional, TYPE_CHECKING
import torch
from torch import Tensor

from .base import BaseLambdaStrategy

if TYPE_CHECKING:
    from deep_inference.models import StructuralModel


class TreatmentDistribution:
    """Base class for treatment distributions."""

    def sample(self, shape: tuple) -> Tensor:
        """Sample from the distribution."""
        raise NotImplementedError()


class Normal(TreatmentDistribution):
    """Normal distribution for continuous treatment."""

    def __init__(self, mean: float = 0.0, std: float = 1.0):
        self.mean = mean
        self.std = std

    def sample(self, shape: tuple) -> Tensor:
        return torch.randn(shape) * self.std + self.mean


class Bernoulli(TreatmentDistribution):
    """Bernoulli distribution for binary treatment."""

    def __init__(self, p: float = 0.5):
        self.p = p

    def sample(self, shape: tuple) -> Tensor:
        return (torch.rand(shape) < self.p).float()


class Uniform(TreatmentDistribution):
    """Uniform distribution."""

    def __init__(self, low: float = 0.0, high: float = 1.0):
        self.low = low
        self.high = high

    def sample(self, shape: tuple) -> Tensor:
        return torch.rand(shape) * (self.high - self.low) + self.low


class ComputeLambda(BaseLambdaStrategy):
    """
    Regime A: Compute Lambda via Monte Carlo.

    For randomized experiments where Hessian doesn't depend on Y.

    Λ(x) = ∫ ℓ_θθ(t, θ̂(x)) dF_T(t)
    """

    requires_theta = True
    requires_separate_fold = False  # 2-way split sufficient

    def __init__(
        self,
        treatment_dist: Optional[TreatmentDistribution] = None,
        model: Optional["StructuralModel"] = None,
        n_mc_samples: int = 1000,
        seed: Optional[int] = None,
    ):
        """
        Initialize ComputeLambda strategy.

        Args:
            treatment_dist: Distribution of treatment (F_T)
            model: The structural model (for computing Hessians)
            n_mc_samples: Number of Monte Carlo samples
            seed: Random seed for reproducibility
        """
        self.treatment_dist = treatment_dist
        self.model = model
        self.n_mc = n_mc_samples
        self.seed = seed
        self._t_samples = None  # Cached samples

    def fit(
        self,
        X: Tensor,
        T: Tensor,
        Y: Tensor,
        theta_hat: Optional[Tensor],
        model: "StructuralModel",
    ) -> None:
        """
        Fit is a no-op for ComputeLambda (we compute on demand).

        We just store the model and sample t values.
        """
        self.model = model

        # Pre-sample treatment values for consistency
        if self.seed is not None:
            torch.manual_seed(self.seed)

        if self.treatment_dist is not None:
            self._t_samples = self.treatment_dist.sample((self.n_mc,))
        else:
            # Fall back to empirical distribution (resample from T)
            n = T.shape[0]
            indices = torch.randint(0, n, (self.n_mc,))
            self._t_samples = T[indices]

    def predict(self, X: Tensor, theta_hat: Optional[Tensor] = None) -> Tensor:
        """
        Compute Λ(x) via Monte Carlo for each observation.

        Λ(x_i) = (1/M) Σ_m ℓ_θθ(t_m, θ̂(x_i))

        Args:
            X: (n, d_x) covariates (not directly used)
            theta_hat: (n, d_theta) estimated parameters

        Returns:
            (n, d_theta, d_theta) Lambda matrices
        """
        if theta_hat is None:
            raise ValueError("ComputeLambda requires theta_hat")

        if self.model is None:
            raise ValueError("Model not set. Call fit() first.")

        n = theta_hat.shape[0]
        d_theta = theta_hat.shape[1]
        device = theta_hat.device
        dtype = theta_hat.dtype

        Lambda = torch.zeros(n, d_theta, d_theta, dtype=dtype, device=device)

        # Get t samples
        if self._t_samples is None:
            raise ValueError("No treatment samples. Call fit() first.")

        t_samples = self._t_samples.to(device)

        # For each observation, compute Monte Carlo average
        for i in range(n):
            theta_i = theta_hat[i]

            for m in range(self.n_mc):
                t_m = t_samples[m]

                # Compute Hessian at (t_m, theta_i)
                # Note: Y is not used (hessian_depends_on_y = False)
                y_dummy = torch.tensor(0.0, dtype=dtype, device=device)

                hess_m = self.model.hessian(y_dummy, t_m, theta_i)

                if hess_m is not None:
                    Lambda[i] += hess_m
                else:
                    # Fall back to autodiff
                    from deep_inference.autodiff.hessian import compute_hessian_loop

                    hess_m = compute_hessian_loop(
                        self.model.loss,
                        y_dummy.unsqueeze(0),
                        t_m.unsqueeze(0),
                        theta_i.unsqueeze(0),
                    )[0]
                    Lambda[i] += hess_m

        Lambda /= self.n_mc

        return Lambda

    def predict_single(self, theta: Tensor) -> Tensor:
        """
        Compute Λ for a single theta value.

        Args:
            theta: (d_theta,) parameter vector

        Returns:
            (d_theta, d_theta) Lambda matrix
        """
        return self.predict(
            X=None, theta_hat=theta.unsqueeze(0)
        )[0]
