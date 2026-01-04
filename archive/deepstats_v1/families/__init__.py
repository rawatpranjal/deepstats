"""Distribution families for GLM estimation.

This module provides exponential family distributions for use with
DeepGLM, DeepHTE, and other estimators.

Available Families
------------------
Normal : Gaussian distribution (continuous outcomes)
Poisson : Poisson distribution (count data)
Bernoulli : Bernoulli distribution (binary outcomes)
Gamma : Gamma distribution (positive continuous, e.g., costs, durations)
Exponential : Exponential distribution (positive continuous, memoryless)

Examples
--------
>>> from deepstats.families import Normal, Poisson, Bernoulli, Gamma, Exponential
>>> import torch
>>>
>>> # Normal family with identity link
>>> normal = Normal()
>>> mu = torch.tensor([1.0, 2.0, 3.0])
>>> ll = normal.log_likelihood(y, mu, dispersion=1.0)
>>>
>>> # Poisson family with log link
>>> poisson = Poisson()
>>> eta = torch.tensor([0.0, 1.0, 2.0])
>>> mu = poisson.inverse_link(eta)  # exp(eta)
>>>
>>> # Bernoulli family with logit link
>>> bernoulli = Bernoulli()
>>> p = bernoulli.inverse_link(eta)  # sigmoid(eta)
>>>
>>> # Gamma family for positive continuous data
>>> gamma = Gamma()
>>> mu = gamma.inverse_link(eta)  # exp(eta) with log link
>>>
>>> # Exponential family (Gamma with shape=1)
>>> exp = Exponential()
>>> mu = exp.inverse_link(eta)  # exp(eta) with log link
"""

from .base import ExponentialFamily
from .bernoulli import Bernoulli
from .exponential import Exponential
from .gamma import Gamma
from .normal import Normal
from .poisson import Poisson

__all__ = [
    "ExponentialFamily",
    "Normal",
    "Poisson",
    "Bernoulli",
    "Gamma",
    "Exponential",
]
