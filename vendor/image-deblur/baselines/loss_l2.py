"""WEAK loss-kind baseline: plain L2 (MSE), no edge term -> over-smoothed, lower deblur PSNR."""
def get_loss_config():
    return {"kind": "l2", "edge_weight": 0.0, "target_smooth": 0.0}
