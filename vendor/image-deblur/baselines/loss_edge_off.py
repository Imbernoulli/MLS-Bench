"""WEAK edge baseline: NO edge/gradient loss term -> edges under-restored, lower PSNR."""
def get_loss_config():
    return {"kind": "charbonnier", "edge_weight": 0.0, "target_smooth": 0.0}
