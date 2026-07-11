"""A DEGENERATE: use A_log directly as A (no -exp). A can go positive -> unstable."""
def compute_A(A_log):
    return A_log.float()
