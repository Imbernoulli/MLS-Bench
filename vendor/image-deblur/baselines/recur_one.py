"""WEAK recurrence baseline: 1 full-res pass -> under-deblurs heavy blur, lower PSNR."""
def get_recurrence_config():
    return {"n_recurrence": 1}
