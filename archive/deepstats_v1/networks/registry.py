"""Architecture registry for neural networks.

This module provides a registry pattern for network architectures,
enabling flexible architecture selection via string identifiers.

Examples
--------
>>> from deepstats.networks import ArchitectureRegistry
>>>
>>> # List available architectures
>>> print(ArchitectureRegistry.available())
['mlp', 'transformer', 'lstm']
>>>
>>> # Create a network
>>> backbone = ArchitectureRegistry.create("mlp", input_dim=10, hidden_dims=[64, 32])
"""

from __future__ import annotations

from typing import Any, Callable, Type

import torch.nn as nn

from .base import BackboneNetwork, NetworkArchitecture


# Type alias for factory functions
ArchitectureFactory = Callable[..., NetworkArchitecture]


class ArchitectureRegistry:
    """Registry for neural network architectures.

    This class provides a decorator-based registration system for
    network architectures, enabling runtime selection of architectures
    by string name.

    The registry pattern allows:
    - Easy addition of new architectures without modifying existing code
    - Runtime discovery of available architectures
    - Consistent interface for creating networks

    Examples
    --------
    >>> @ArchitectureRegistry.register("custom_net")
    ... def create_custom(input_dim: int, **kwargs) -> NetworkArchitecture:
    ...     return CustomNetwork(input_dim, **kwargs)
    >>>
    >>> net = ArchitectureRegistry.create("custom_net", input_dim=10)
    """

    _registry: dict[str, ArchitectureFactory] = {}
    _configs: dict[str, dict[str, Any]] = {}

    @classmethod
    def register(
        cls,
        name: str,
        default_config: dict[str, Any] | None = None,
    ) -> Callable[[ArchitectureFactory], ArchitectureFactory]:
        """Decorator to register an architecture factory.

        Parameters
        ----------
        name : str
            Name to register the architecture under.
        default_config : dict, optional
            Default configuration for this architecture.

        Returns
        -------
        Callable
            Decorator function.

        Examples
        --------
        >>> @ArchitectureRegistry.register("my_arch")
        ... def create_my_arch(input_dim: int, **kwargs):
        ...     return MyArchitecture(input_dim, **kwargs)
        """
        def decorator(factory_fn: ArchitectureFactory) -> ArchitectureFactory:
            cls._registry[name.lower()] = factory_fn
            if default_config is not None:
                cls._configs[name.lower()] = default_config
            return factory_fn
        return decorator

    @classmethod
    def register_class(
        cls,
        name: str,
        arch_class: Type[NetworkArchitecture],
        default_config: dict[str, Any] | None = None,
    ) -> None:
        """Register a network class directly.

        Parameters
        ----------
        name : str
            Name to register the architecture under.
        arch_class : Type[NetworkArchitecture]
            Network class to register.
        default_config : dict, optional
            Default configuration for this architecture.
        """
        def factory(input_dim: int, **kwargs) -> NetworkArchitecture:
            return arch_class(input_dim=input_dim, **kwargs)

        cls._registry[name.lower()] = factory
        if default_config is not None:
            cls._configs[name.lower()] = default_config

    @classmethod
    def create(
        cls,
        architecture: str,
        input_dim: int,
        **kwargs,
    ) -> NetworkArchitecture:
        """Create a network from the registry.

        Parameters
        ----------
        architecture : str
            Name of the architecture to create.
        input_dim : int
            Number of input features.
        **kwargs
            Additional arguments passed to the factory.

        Returns
        -------
        NetworkArchitecture
            Constructed network.

        Raises
        ------
        ValueError
            If the architecture is not registered.

        Examples
        --------
        >>> backbone = ArchitectureRegistry.create("mlp", input_dim=10)
        >>> print(type(backbone).__name__)
        'MLPBackbone'
        """
        name = architecture.lower()
        if name not in cls._registry:
            available = list(cls._registry.keys())
            raise ValueError(
                f"Unknown architecture: '{architecture}'. "
                f"Available: {available}"
            )

        # Merge default config with provided kwargs
        config = cls._configs.get(name, {}).copy()
        config.update(kwargs)

        return cls._registry[name](input_dim=input_dim, **config)

    @classmethod
    def available(cls) -> list[str]:
        """List available architecture names.

        Returns
        -------
        list[str]
            Sorted list of registered architecture names.
        """
        return sorted(cls._registry.keys())

    @classmethod
    def get_default_config(cls, name: str) -> dict[str, Any]:
        """Get default configuration for an architecture.

        Parameters
        ----------
        name : str
            Architecture name.

        Returns
        -------
        dict
            Default configuration.
        """
        return cls._configs.get(name.lower(), {}).copy()

    @classmethod
    def is_registered(cls, name: str) -> bool:
        """Check if an architecture is registered.

        Parameters
        ----------
        name : str
            Architecture name.

        Returns
        -------
        bool
            True if registered.
        """
        return name.lower() in cls._registry


def create_backbone(
    architecture: str,
    input_dim: int,
    **kwargs,
) -> BackboneNetwork:
    """Convenience function to create a backbone network.

    Parameters
    ----------
    architecture : str
        Architecture name (e.g., "mlp", "transformer", "lstm").
    input_dim : int
        Number of input features.
    **kwargs
        Architecture-specific configuration.

    Returns
    -------
    BackboneNetwork
        Constructed backbone network.

    Examples
    --------
    >>> backbone = create_backbone("mlp", input_dim=10, hidden_dims=[64, 32])
    >>> x = torch.randn(100, 10)
    >>> h = backbone(x)  # Shape: (100, 32)
    """
    return ArchitectureRegistry.create(architecture, input_dim=input_dim, **kwargs)
