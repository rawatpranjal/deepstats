"""
Basic usage examples for deepstats package.

This script demonstrates how to use deepstats for various statistical
models where the conditional mean is modeled by a neural network.
"""

import numpy as np
import pandas as pd

import deepstats as ds


def example_normal():
    """Example: Normal regression Y ~ N(g(X), sigma^2)"""
    print("=" * 60)
    print("Example 1: Normal Regression")
    print("=" * 60)

    # Generate synthetic data
    np.random.seed(42)
    n = 1000

    df = pd.DataFrame({
        "education": np.random.randn(n),
        "experience": np.random.randn(n),
        "ability": np.random.randn(n),
    })
    # True model: Y = 2 + 0.5*education - 0.3*experience + 0.8*ability + noise
    df["wage"] = (
        2
        + 0.5 * df["education"]
        - 0.3 * df["experience"]
        + 0.8 * df["ability"]
        + 0.5 * np.random.randn(n)
    )

    # Fit model using formula
    result = ds.deep_fit(
        "wage ~ education + experience + ability",
        data=df,
        family=ds.Normal(),
        epochs=100,
        verbose=0,
        seed=42,
    )

    print(result.summary())
    print("\nAverage Marginal Effects:")
    effects = result.compute_marginal_effects()
    for name, effect in effects.items():
        print(f"  {name}: {effect:.4f}")
    print("\nTrue coefficients: education=0.5, experience=-0.3, ability=0.8")


def example_poisson():
    """Example: Poisson regression Y ~ Poisson(exp(g(X)))"""
    print("\n" + "=" * 60)
    print("Example 2: Poisson Regression (Count Data)")
    print("=" * 60)

    np.random.seed(42)
    n = 1000

    df = pd.DataFrame({
        "advertising": np.random.randn(n),
        "price": np.random.randn(n),
    })
    # True model: count = Poisson(exp(1 + 0.3*advertising - 0.2*price))
    rate = np.exp(1 + 0.3 * df["advertising"] - 0.2 * df["price"])
    df["sales_count"] = np.random.poisson(rate)

    result = ds.deep_fit(
        "sales_count ~ advertising + price",
        data=df,
        family=ds.Poisson(),
        epochs=100,
        verbose=0,
        seed=42,
    )

    print(result.summary())
    print("\nAverage Marginal Effects (on log-scale):")
    effects = result.compute_marginal_effects()
    for name, effect in effects.items():
        print(f"  {name}: {effect:.4f}")


def example_binomial():
    """Example: Binomial regression Y ~ Bernoulli(sigmoid(g(X)))"""
    print("\n" + "=" * 60)
    print("Example 3: Binary Classification")
    print("=" * 60)

    np.random.seed(42)
    n = 1000

    df = pd.DataFrame({
        "income": np.random.randn(n),
        "credit_score": np.random.randn(n),
    })
    # True model: P(default) = sigmoid(0.5 - 0.8*income - 0.5*credit_score)
    prob = 1 / (1 + np.exp(-(0.5 - 0.8 * df["income"] - 0.5 * df["credit_score"])))
    df["default"] = np.random.binomial(1, prob)

    result = ds.deep_fit(
        "default ~ income + credit_score",
        data=df,
        family=ds.Binomial(),
        epochs=100,
        verbose=0,
        seed=42,
    )

    print(result.summary())

    # Accuracy
    preds = result.predict()
    accuracy = np.mean((preds > 0.5) == df["default"].values)
    print(f"\nClassification Accuracy: {accuracy:.4f}")


def example_tobit():
    """Example: Tobit regression for censored data"""
    print("\n" + "=" * 60)
    print("Example 4: Tobit Model (Censored Data)")
    print("=" * 60)

    np.random.seed(42)
    n = 1000

    df = pd.DataFrame({
        "X1": np.random.randn(n),
        "X2": np.random.randn(n),
    })
    # True latent model: Y* = 1 + 0.5*X1 - 0.3*X2 + noise
    # Observed: Y = max(0, Y*)
    latent = 1 + 0.5 * df["X1"] - 0.3 * df["X2"] + 0.5 * np.random.randn(n)
    df["Y"] = np.maximum(0, latent)

    censored_pct = np.mean(df["Y"] == 0) * 100
    print(f"Percentage censored at 0: {censored_pct:.1f}%")

    result = ds.deep_fit(
        "Y ~ X1 + X2",
        data=df,
        family=ds.Tobit(lower=0),
        epochs=100,
        verbose=0,
        seed=42,
    )

    print(result.summary())


def example_custom_network():
    """Example: Using a custom network architecture"""
    print("\n" + "=" * 60)
    print("Example 5: Custom Network Architecture")
    print("=" * 60)

    np.random.seed(42)
    n = 1000
    X = np.random.randn(n, 5)
    y = np.sin(X[:, 0]) + np.cos(X[:, 1]) + 0.5 * X[:, 2] ** 2 + 0.3 * np.random.randn(n)

    # Custom MLP with different architecture
    custom_net = ds.MLP(
        input_dim=5,
        hidden_dims=[128, 64, 32],  # Deeper network
        output_dim=1,
        activation="relu",
        dropout=0.1,
        batch_norm=True,
    )

    result = ds.deep_fit(
        X=X,
        y=y,
        family=ds.Normal(),
        network=custom_net,
        epochs=200,
        lr=1e-3,
        verbose=0,
        seed=42,
    )

    print(result.summary())
    print(f"\nCustom network: {custom_net}")


if __name__ == "__main__":
    example_normal()
    example_poisson()
    example_binomial()
    example_tobit()
    example_custom_network()
    print("\n" + "=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)
