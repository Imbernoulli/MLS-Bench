# Gaussian Processes: SVGP Learning Rate

## Objective
Choose the Adam learning rate for a fixed stochastic variational GP training
budget. Minimize test NLL across all evaluation datasets.

## Interface
Edit only the returned JSON literal in
`gpytorch-gp/solution/svgp_lr.py`:

```python
def surface_config():
    return {"learning_rate": 0.01}
```

The value must be finite and in `(0, 1]`; booleans are rejected. The verifier
parses a bounded literal AST and never executes agent-authored Python. Extra
keys, imports, executable statements, and malformed values fail verification
instead of selecting another learning rate.

## Evaluation
The fixed SVGP uses 256 k-means inducing points, a Cholesky variational
distribution, ARD-RBF covariance, Gaussian likelihood, batch size 1,024, and
exactly 20 epochs. Every configured evaluation uses a complete, checksum-bound
OpenML regression dataset with fixed train/test ownership and must produce finite
test NLL and RMSE on the original target scale.
