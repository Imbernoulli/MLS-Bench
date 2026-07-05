"""STRONG loss-kind baseline: Charbonnier + edge term -> sharp restorations, higher PSNR.
Charbonnier: Lai et al. (LapSRN); used by MPRNet (Zamir et al., CVPR 2021)."""
def get_loss_config():
    return {"kind": "charbonnier", "edge_weight": 0.1, "target_smooth": 0.0}
