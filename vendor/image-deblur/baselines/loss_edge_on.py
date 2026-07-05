"""STRONG edge baseline: Charbonnier + STRONG edge/gradient loss -> sharper edges, higher PSNR.
Edge/gradient loss: LapSRN (Lai et al. CVPR 2017); MPRNet (Zamir et al. CVPR 2021)."""
def get_loss_config():
    return {"kind": "charbonnier", "edge_weight": 0.5, "target_smooth": 0.0}
