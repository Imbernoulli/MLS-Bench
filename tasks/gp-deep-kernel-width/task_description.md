# Gaussian Processes: Deep-Kernel Bottleneck Width

## Objective
Choose the bottleneck width of a fixed two-hidden-layer feature MLP for a
deep-kernel ExactGP. Minimize test NLL across all evaluation datasets.

## Interface
Edit only the returned JSON literal in
`gpytorch-gp/solution/deep_kernel_width.py`:

```python
def surface_config():
    return {"width": 1}
```

`width` must be a Python integer in `[1, 32]`. A trusted builder constructs
the fixed `d -> 64 -> 32 -> width` ReLU MLP. The verifier parses a bounded
literal AST and never executes agent-authored Python. Extra keys, imports,
executable statements, booleans, and out-of-range values fail verification.

## Evaluation
The MLP and fixed scaled ARD-RBF ExactGP head train jointly for exactly 200
marginal-likelihood iterations. Every configured evaluation uses a complete,
checksum-bound OpenML regression dataset with fixed train/test ownership and must
emit finite test NLL and RMSE on the original target scale.
