"""Pre-built families with closed-form optimizations."""

from .base import BaseFamily, Family
from .linear import LinearFamily
from .logit import LogitFamily

FAMILY_REGISTRY = {
    "linear": LinearFamily,
    "logit": LogitFamily,
}


def get_family(name: str) -> BaseFamily:
    """Get a family by name."""
    if name not in FAMILY_REGISTRY:
        raise ValueError(f"Unknown family: {name}. Available: {list(FAMILY_REGISTRY.keys())}")
    return FAMILY_REGISTRY[name]()


__all__ = [
    "BaseFamily",
    "Family",
    "LinearFamily",
    "LogitFamily",
    "get_family",
    "FAMILY_REGISTRY",
]
