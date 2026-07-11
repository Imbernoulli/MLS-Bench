# Gaussian Processes: Sparse Inducing Points

## Objective
Choose the inducing-point method and count for a fixed stochastic variational
GP. Minimize test NLL across all evaluation datasets.

## Interface
Edit only the returned JSON literal in
`gpytorch-gp/solution/inducing.py`:

```python
def surface_config():
    return {"method": "random", "count": 16}
```

`method` must be `random` or `kmeans`; `count` must be a Python integer
in `[1, 2048]`. The trusted verifier selects locations from standardized
training inputs. It parses a bounded literal AST and never executes
agent-authored Python. Extra keys, imports, executable statements, and malformed
values fail verification.

## Evaluation
The fixed SVGP uses an ARD-RBF covariance, constant mean, Cholesky variational
distribution, Gaussian likelihood, batch size 1,024, and exactly 20 epochs.
Every configured evaluation uses a complete, checksum-bound OpenML regression
dataset with fixed train/test ownership and must produce finite test NLL and RMSE
on the original target scale.
