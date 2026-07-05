"""WEAK network baseline (= the default): copy the shadowed input straight through, NO
removal. The do-nothing floor -- it scores exactly the shadowed-input shadow-region PSNR;
every real deshadower must beat it."""
def get_network_config():
    return {"arch": "copy"}
