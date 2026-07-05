"""STRONG recurrence baseline: 4 shared-weight refinement passes (SRN recurrence).
Scale-recurrent refinement: SRN-DeblurNet (Tao et al. CVPR 2018)."""
def get_recurrence_config():
    return {"n_recurrence": 4}
