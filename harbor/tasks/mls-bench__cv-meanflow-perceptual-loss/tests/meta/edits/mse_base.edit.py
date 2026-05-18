"""Floor baseline: pure MSE on velocity (no inverse-loss adaptive weighting).

The previous mse-only baseline divided by per-sample MSE
(`weight = 1.0 / (loss_mse.detach() + 1e-3)`) which is a reverse-focal
formulation that amplifies easy samples and causes divergence at
step 35-40k. This baseline removes that pathology and uses a clean
mean MSE — establishes the true MSE-only floor.
"""

_FILE = "alphaflow-main/custom_train_perceptual.py"

_MSE_BASE = '''\
            # Pure MSE on mean velocity prediction.
            # No inverse-loss reweighting (which would amplify easy samples
            # and destabilise training around step 35k).
            loss_mse_unscaled = ((pred_mean_vel - mean_vel_target) ** 2).flatten(1).mean(1)
            loss = loss_mse_unscaled.mean()
'''

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 384,
        "end_line": 401,
        "content": _MSE_BASE,
    },
]
