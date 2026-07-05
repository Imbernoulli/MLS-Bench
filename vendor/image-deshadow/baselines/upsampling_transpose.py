"""WEAK upsampling baseline: transpose-convolution (deconv) upsampling, checkerboard-prone."""


def get_upsampling_config():
    return {"up": "transpose"}
