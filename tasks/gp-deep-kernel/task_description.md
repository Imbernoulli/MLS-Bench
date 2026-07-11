# Gaussian Processes: Deep-Kernel Feature Extractor

## Objective
Choose whether a fixed ExactGP uses raw inputs or a fixed learned MLP feature
extractor. Minimize test NLL across all evaluation datasets.

## Interface
Edit only the returned JSON literal in
`gpytorch-gp/solution/deep_kernel.py`. An identity plan is:

```python
def surface_config():
    return {"extractor": "identity"}
```

The alternative plan is:

```python
def surface_config():
    return {"extractor": "mlp"}
```

The trusted builder fixes the learned extractor to `d -> 64 -> 32 -> 4` with
ReLU activations. This isolates the presence of feature learning; the separate
deep-kernel-width task studies bottleneck capacity. The verifier parses a
bounded literal AST and never executes agent-authored Python. Extra axes,
imports, executable statements, and malformed values fail verification.

## Evaluation
The extractor feeds a fixed scaled ARD-RBF ExactGP head and is trained jointly
for exactly 200 marginal-likelihood iterations. Every configured evaluation uses
a complete, checksum-bound OpenML regression dataset with fixed train/test
ownership. Test NLL and RMSE are measured on the original target scale, and every
configured evaluation must complete.
