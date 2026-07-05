"""MID network baseline: a BLIND U-Net that sees only the 3-channel shadowed RGB and must
both LOCATE and correct the shadow from colour alone (DeshadowNet-style multi-context net
WITHOUT the mask prior). Beats the do-nothing floor but, not knowing exactly where/how-much,
leaks into the lit region and mis-corrects the soft penumbra -> lower shadow-region PSNR than
the mask-guided net."""
def get_network_config():
    return {"arch": "unet_nomask"}
