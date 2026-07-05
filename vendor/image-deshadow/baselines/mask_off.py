"""WEAK mask baseline: blind U-Net that sees only the 3-ch shadowed RGB (DeshadowNet)."""


def get_mask_config():
    return {"use_mask": False}
