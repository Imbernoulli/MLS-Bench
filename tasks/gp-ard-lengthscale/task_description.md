# Gaussian Processes: Shared vs Per-Dimension Lengthscales

## Objective
Choose whether the fixed RBF covariance uses one shared lengthscale or one
lengthscale per input dimension. Minimize test negative log-likelihood (NLL)
across all evaluation datasets.

## Interface
Edit only the returned JSON literal in
`gpytorch-gp/solution/ard_lengthscale.py`:

```python
def surface_config():
    return {"ard": False}
```

`ard` must be a Python boolean. The verifier parses this function as a bounded
AST and never executes agent-authored Python. Imports, additional statements,
non-literal expressions, extra keys, and malformed values fail verification.

## Evaluation
A trusted builder constructs the covariance and a fixed ExactGP. Every configured
evaluation uses a complete, checksum-bound OpenML regression dataset with fixed
train/test ownership and exactly 200 marginal-likelihood iterations. Inputs and
targets are standardized from training statistics; test NLL and RMSE are reported
on the original target scale. Every configured evaluation must complete.
