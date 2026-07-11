# Gaussian Processes: Kernel Smoothness

## Objective
Choose the stationary covariance family while ARD and all other ExactGP choices
remain fixed. Minimize test NLL across all evaluation datasets.

## Interface
Edit only the returned JSON literal in
`gpytorch-gp/solution/kernel_smoothness.py`:

```python
def surface_config():
    return {"kernel": "rbf"}
```

`kernel` must be exactly `rbf`, `matern12`, or `matern52`. A trusted
builder constructs the ARD covariance. The verifier parses a bounded literal AST
and never executes agent-authored Python. Extra keys, imports, executable
statements, and unsupported values fail verification.

## Evaluation
A fixed ExactGP trains for exactly 200 marginal-likelihood iterations in every
configured evaluation. Each uses a complete, checksum-bound OpenML regression
dataset with fixed train/test ownership and must produce finite test NLL and RMSE
on the original target scale.
