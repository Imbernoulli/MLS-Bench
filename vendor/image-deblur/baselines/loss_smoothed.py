"""WEAK loss baseline: optimise toward an OVER-SMOOTHED (low-pass) target.
The network is trained to match a Gaussian-blurred version of the sharp GT, so it
deliberately blurs away the high-frequency detail it should restore -- the classic
L2-conditional-mean over-smoothing failure. Lower deblur PSNR vs the sharp GT."""
def get_loss_config():
    return {"kind": "charbonnier", "edge_weight": 0.1, "target_smooth": 1.2}
