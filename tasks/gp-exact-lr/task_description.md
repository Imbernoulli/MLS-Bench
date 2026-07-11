# Gaussian Processes: ExactGP Learning Rate

## Objective
Choose the Adam learning rate for a fixed-budget ExactGP fit. Minimize test NLL
across all evaluation datasets without changing the model or iteration budget.

## Interface
Edit only the returned JSON literal in
`gpytorch-gp/solution/exact_lr.py`:

```python
def surface_config():
    return {"learning_rate": 0.01}
```

The value must be a finite number in `(0, 1]`; booleans are rejected. The
verifier parses a bounded literal AST and never executes agent-authored Python.
Extra keys, imports, executable statements, and malformed values fail
verification instead of selecting another learning rate.

## Evaluation
The fixed ExactGP uses a scaled ARD Matern-5/2 covariance, constant mean,
Gaussian likelihood, and exactly 200 optimizer iterations. Every configured
evaluation uses a complete, checksum-bound OpenML regression dataset with fixed
train/test ownership and must produce finite test NLL and RMSE on the original
target scale.
