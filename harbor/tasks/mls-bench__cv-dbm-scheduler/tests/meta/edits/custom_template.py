_TEMPLATE = """\
def get_sigmas_uniform(n, t_min, t_max, device="cpu"):
    \"\"\"
    Requirements:
    1. Length: Must return a 1D PyTorch tensor of exactly length `n + 1`.
    2. Monotonic: The sequence must strictly decrease from `t_max` to `t_min`.
    3. Terminal Value: The final element (index `n`) must exactly equal `t_min`.
    4. Device: Move the tensor to the requested `device`.
    \"\"\"
    # For this task, n will typically be 5 (NFE=5).
    # Implement your novel schedule formulation here...
    raise NotImplementedError("Custom scheduler not implemented yet.")
"""
