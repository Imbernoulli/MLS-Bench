_TEMPLATE = """\
@torch.no_grad()
def sample_dbim(
    denoiser,
    diffusion,
    x,
    ts,
    eta=1.0,
    mask=None,
    seed=None,
    **kwargs,
):
    # =================================================================================
    # 🚨 CRITICAL CONSTRAINTS - DO NOT IGNORE! 🚨
    # 1. Function Signature: You must NOT modify the function name, arguments, or return structure.
    # 2. NFE Match (FATAL I/O ERROR): The framework uses the final returned `nfe` to locate
    #    generated files (e.g., expecting `samples_..._nfe5.npz`). You MUST return
    #    `nfe = len(ts) - 1` regardless of the internal call count.
    # =================================================================================
    
    # TODO: Implement your novel sampling kernel here.
    # Ensure the return structure is: return x, path, nfe, pred_x0, ts, first_noise
    
    raise NotImplementedError("Custom sampler not implemented yet.")
"""
