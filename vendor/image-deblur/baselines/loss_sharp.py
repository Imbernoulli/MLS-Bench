"""GOOD loss baseline: optimise toward the TRUE SHARP target (no over-smoothing).
Charbonnier + edge term on the sharp GT -> the network restores high-frequency detail
and lands at high deblur PSNR. (Charbonnier: Lai et al. LapSRN; used by MPRNet.)"""
def get_loss_config():
    return {"kind": "charbonnier", "edge_weight": 0.1, "target_smooth": 0.0}
