# Gaussian Processes: Mean Function

## Objective
Choose the mean module of a fixed ExactGP. Minimize test NLL across all
evaluation datasets.

## Interface
Edit only the returned JSON literal in
`gpytorch-gp/solution/mean_function.py`:

```python
def surface_config():
    return {"mean": "zero"}
```

`mean` must be exactly `zero`, `constant`, or `linear`. A trusted
builder constructs the GPyTorch mean. The verifier parses a bounded literal AST
and never executes agent-authored Python. Extra keys, imports, executable
statements, and unsupported values fail verification.

## Evaluation
The covariance is a fixed scaled ARD Matern-5/2 kernel and the Gaussian
likelihood is fixed. Exactly 200 marginal-likelihood iterations run in every
configured evaluation. Each uses a complete, checksum-bound OpenML regression
dataset with fixed train/test ownership and must produce finite test NLL and RMSE
on the original target scale.
