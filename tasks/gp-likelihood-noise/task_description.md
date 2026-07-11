# Gaussian Processes: Observation Noise

## Objective
Choose whether a fixed ExactGP learns homoscedastic observation noise or uses a
specified fixed value. Minimize test NLL across all evaluation datasets.

## Interface
Edit only the returned JSON literal in
`gpytorch-gp/solution/likelihood_noise.py`:

```python
def surface_config():
    return {"mode": "learned"}
```

A fixed-noise plan is `{"mode": "fixed", "noise": 0.0001}`, where `noise`
must be finite and in `[1e-6, 1]`. A trusted builder constructs the Gaussian
likelihood. The verifier parses a bounded literal AST and never executes
agent-authored Python. Extra keys, imports, executable statements, and malformed
values fail verification.

## Evaluation
The covariance is a fixed scaled ARD Matern-5/2 kernel and the mean is constant.
Exactly 200 marginal-likelihood iterations run in every configured evaluation.
Each uses a complete, checksum-bound OpenML regression dataset with fixed
train/test ownership and must produce finite test NLL and RMSE on the original
target scale.
