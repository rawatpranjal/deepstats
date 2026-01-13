"""Pre-built families with closed-form optimizations."""

from .base import BaseFamily, Family
from .linear import LinearFamily
from .logit import LogitFamily
from .poisson import PoissonFamily
from .gamma import GammaFamily
from .gaussian import GaussianFamily
from .gumbel import GumbelFamily
from .tobit import TobitFamily
from .negbin import NegBinFamily
from .weibull import WeibullFamily

FAMILY_REGISTRY = {
    "linear": LinearFamily,
    "logit": LogitFamily,
    "poisson": PoissonFamily,
    "gamma": GammaFamily,
    "gaussian": GaussianFamily,
    "gumbel": GumbelFamily,
    "tobit": TobitFamily,
    "negbin": NegBinFamily,
    "weibull": WeibullFamily,
}


def get_family(name: str, **kwargs) -> BaseFamily:
    """
    Get a family by name.

    Args:
        name: Family name. Available:
              - 'linear': Y = alpha + beta*T + eps (squared error)
              - 'logit': P(Y=1) = sigmoid(alpha + beta*T)
              - 'poisson': Y ~ Poisson(exp(alpha + beta*T))
              - 'gamma': Y ~ Gamma(shape, exp(alpha + beta*T))
              - 'gaussian': Y ~ N(alpha + beta*T, sigma) (Gaussian NLL)
              - 'gumbel': Y ~ Gumbel(alpha + beta*T, scale)
              - 'tobit': Y = max(0, alpha + beta*T + sigma*eps)
              - 'negbin': Y ~ NegBin(exp(alpha + beta*T), r)
              - 'weibull': Y ~ Weibull(shape, exp(alpha + beta*T))
        **kwargs: Additional arguments passed to family constructor.
                  Examples:
                  - target='ame' for logit (average marginal effect)
                  - target='observed' for tobit (effect on observed Y)
                  - shape=2.0 for gamma/weibull
                  - scale=1.0 for gumbel
                  - overdispersion=0.5 for negbin

    Returns:
        Instantiated family object
    """
    if name not in FAMILY_REGISTRY:
        raise ValueError(f"Unknown family: {name}. Available: {list(FAMILY_REGISTRY.keys())}")
    return FAMILY_REGISTRY[name](**kwargs)


__all__ = [
    "BaseFamily",
    "Family",
    "LinearFamily",
    "LogitFamily",
    "PoissonFamily",
    "GammaFamily",
    "GaussianFamily",
    "GumbelFamily",
    "TobitFamily",
    "NegBinFamily",
    "WeibullFamily",
    "get_family",
    "FAMILY_REGISTRY",
]
