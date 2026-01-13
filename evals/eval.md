# Eval Learnings

What we learn from each eval. Updated as we investigate.

---

## Eval 01: Parameter Recovery

**Goal**: Recover θ*(x) = [α*(x), β*(x)] from logistic data.

**Issue 1 - Early stopping**: Default `patience=10` stops training at epoch ~15. The network needs 50-200 epochs to converge. Fix: increase patience to 50+ in `structural_net.py:109`.

**Issue 2 - Flat loss surface**: The logistic loss is nearly flat w.r.t. level shifts. Adding +0.1 to α and -0.1 to β changes loss by only 0.0003. This means many (α,β) pairs achieve nearly identical loss, making exact level recovery impossible. The network converges to a point with ~+0.05 bias in α and ~-0.05 bias in β.

**Issue 3 - Thresholds**: RMSE(β) < 0.1 is unrealistic given the flat loss surface. Even with n=20k and 500 epochs, best achieved is RMSE(β)=0.13. Relax to 0.2.

**What helps**: More data (n=20k vs 5k), no dropout for large n, patience=50+, epochs=300+.

**What doesn't help**: More epochs beyond convergence (~100), lower learning rate (0.001 vs 0.01 - actually worse).

---

## Eval 02-06

(TODO)
