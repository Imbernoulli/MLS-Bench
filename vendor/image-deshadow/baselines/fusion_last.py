"""WEAK fusion baseline: use only the last decoder block's features."""


def get_fusion_config():
    return {"fusion": False}
