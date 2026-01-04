# DeepHTE Tuning Results

## Summary

DeepHTE is designed to excel on **high-dimensional non-tabular data** (images, text, graphs) where tree-based methods like CausalForest cannot naturally operate.

On tabular data, CausalForest has structural advantages for capturing interactions and thresholds. This is expected behavior.

## Tabular Tuning (Balanced DGP)

**Best config found:**
```python
epochs=1000
hidden_dims=[256, 128, 64]
lr=0.01
dropout=0.1
```

**Results:**

| Method | ITE RMSE | ITE Corr |
|--------|----------|----------|
| CausalForest | 0.2939 | 0.9413 |
| DeepHTE (tuned) | 0.4016 | 0.8453 |
| LinearDML | 0.4406 | 0.7658 |

**Key findings:**
- DeepHTE beats LinearDML with proper tuning
- CausalForest wins on tabular (expected - trees handle X² interactions naturally)
- Bigger networks don't help ([512,256,128] was worse)
- No dropout leads to overfitting (2000 epochs → RMSE 0.51)

## Default Simulation Config (Conservative)

The simulations use conservative settings for reproducibility:
```python
epochs=200
hidden_dims=[64, 32]
lr=0.001
```

For production use, consider the tuned config above.

## Where DeepHTE Should Win

DeepHTE's advantage is on high-dimensional data:

1. **Images** - CNN backbone processes raw pixels
2. **Text** - Transformer/LSTM processes token sequences
3. **Graphs** - GNN processes node/edge features

On these modalities, CausalForest must use extracted features, losing information.

## ✅ RAW DATA BACKBONE SUPPORT (IMPLEMENTED)

DeepHTE now supports raw data processing with specialized backbones:

```python
# Image data with CNN
model = DeepHTE(backbone="cnn", epochs=500)
result = model.fit_raw(X=images, y=outcome, t=treatment)

# Graph data with GNN
model = DeepHTE(backbone="gnn", epochs=500)
result = model.fit_raw(X=node_features, y=outcome, t=treatment, edge_index=edges)

# Text data with Transformer
model = DeepHTE(backbone="text", epochs=500)
result = model.fit_raw(X=tokens, y=outcome, t=treatment, vocab_size=1000)
```

The simulation runners have been updated to use raw data:
- `image_runner.py`: Uses CNN backbone on raw images
- `graph_runner.py`: Uses GNN backbone on raw graph data
- `text_runner.py`: Uses Text backbone on raw token sequences

This gives DeepHTE a proper advantage on high-dimensional data since:
- CNN learns directly from pixels (CF/LinearDML use lossy extracted features)
- GNN learns graph structure (CF/LinearDML use flat graph statistics)
- Text backbone processes sequences (CF/LinearDML use bag-of-words features)

## Recommended Configs by Modality

| Modality | Backbone | Epochs | Hidden Dims | LR |
|----------|----------|--------|-------------|-----|
| Tabular | MLP | 1000 | [256, 128, 64] | 0.01 |
| Image | CNN | 500 | - | 0.001 |
| Text | Transformer | 500 | d_model=64 | 0.001 |
| Graph | GNN | 500 | - | 0.001 |
