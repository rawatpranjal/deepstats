"""
Babysit DeepHTE training - watch what's happening.
"""
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, "src")

import deepstats as ds

# Generate simple data
print("Generating data...")
np.random.seed(42)
n, p = 2000, 10
X = np.random.randn(n, p)
T = np.random.binomial(1, 0.5, n)

# True effects - simple quadratic
b_true = 2.0 + 0.5 * X[:, 0] + 0.3 * X[:, 1]**2
a_true = 0.5 * X[:, 0] + 0.3 * X[:, 1]
Y = a_true + b_true * T + np.random.randn(n) * 0.5

data = pd.DataFrame(X, columns=[f"X{i+1}" for i in range(p)])
data["T"] = T
data["Y"] = Y

print(f"True ATE: {b_true.mean():.3f}")
print(f"True ITE std: {b_true.std():.3f}")

# Formula
covs = " + ".join([f"X{i+1}" for i in range(p)])
formula = f"Y ~ a({covs}) + b({covs}) * T"

# Train with verbose
print("\n" + "="*60)
print("Training DeepHTE (deeper net, longer training)")
print("="*60)

model = ds.DeepHTE(
    formula=formula,
    epochs=500,
    hidden_dims=[256, 128, 64],  # Deeper
    lr=0.01,                      # Default (not 0.001)
    dropout=0.1,
    weight_decay=1e-4,
    verbose=2,                    # Show progress
    random_state=42,
)

result = model.fit(data)

# Evaluate
pred_ite = result.ite
rmse = np.sqrt(np.mean((pred_ite - b_true)**2))
corr = np.corrcoef(pred_ite, b_true)[0, 1]

print("\n" + "="*60)
print("RESULTS")
print("="*60)
print(f"ATE estimate: {result.ate:.3f} (true: {b_true.mean():.3f})")
print(f"ITE RMSE: {rmse:.4f}")
print(f"ITE Corr: {corr:.4f}")
print(f"ITE range: [{pred_ite.min():.2f}, {pred_ite.max():.2f}]")
print(f"True range: [{b_true.min():.2f}, {b_true.max():.2f}]")
