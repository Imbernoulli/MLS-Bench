# Gaussian Processes: Covariance and Mean Design

## Objective
Choose a covariance family and mean function for a fixed ExactGP. Minimize test
NLL across all evaluation datasets.

## Interface
Edit only the returned JSON literal in
`gpytorch-gp/solution/kernel_design.py`. Supported plans are:

```python
def surface_config():
    return {"kernel": "rbf", "ard": False, "mean": "constant"}
```

For `kernel="matern"`, also provide `nu` as `0.5`, `1.5`, or `2.5`.
For `kernel="spectral_mixture"`, provide `num_mixtures` in `[1, 8]` and set
`ard=True`; multidimensional spectral mixtures use one set of mixture scales and
means per input dimension. `ard` must otherwise be boolean and `mean` must be
`zero`, `constant`, or `linear`. A trusted builder constructs all GPyTorch
modules and performs data-dependent initialization. The verifier parses a
bounded literal AST and never executes agent-authored Python. Extra keys,
executable statements, imports, and malformed plans fail verification.

## Evaluation
The likelihood, optimizer, split, and 200-iteration budget are fixed. Every
configured evaluation uses a complete, checksum-bound OpenML regression dataset
with fixed train/test ownership. Exact solves use at most 10,000 CG iterations,
training/evaluation tolerances of 0.01/0.001, and fixed preconditioner ranks of
100 for Concrete and Elevators and 500 for Kin8nm. Every setting must produce
finite test NLL and RMSE on the original target scale. Missing evaluations,
incomplete training, or invalid metrics score zero.
