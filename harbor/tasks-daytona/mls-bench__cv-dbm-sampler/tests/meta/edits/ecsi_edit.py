_FILE = "dbim-codebase/ddbm/karras_diffusion.py"
_ECSI_ALGORITHM = """\
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
    \"\"\"
    ECSI (Endpoint-Conditioned Stochastic Interpolants) sampler.
    Paper: Zhang et al. arXiv:2410.21553
    ('Exploring the Design Space of Diffusion Bridge Models').
    Code: https://github.com/szhan311/ECSI  (sibm/sampling.py: sample_stoch).

    Task-local ECSI-inspired sampler settings:
      * pred_mode = \"vp\"  (already the dbim-codebase e2h default)
      * sigma_min is set below from the local e2h sweep
      * churn_step_ratio = 0.3
      * rho = 0.6
      * NFE = steps (5 for e2h)

    Convention mapping ECSI(alpha,beta,gamma) -> dbim-codebase(b_t,a_t,c_t):
    dbim's x_t = a_t*x_T + b_t*x_0 + c_t*noise, so ECSI's alpha (x_0 coef)
    = dbim's b_t, beta = a_t, gamma = c_t. Derivatives are computed
    analytically from the VP schedule (dbim-codebase exposes f_fn = -(ln alpha)'
    and g2_fn = (rho^2 + 1)' / (rho^2 + 1), which give us alpha'(t) and rho'(t)
    exactly — finite differences would lose ~1e-4 accuracy near the t_max
    boundary where c(t) ~ O(1e-2) itself).
    \"\"\"
    churn = 0.3
    rho_k = 0.6
    sigma_min_ecsi = 0.15   # task-local e2h sweep value
    sigma_max_offset = 5e-4 # paired task-local sweep value
    t_max = diffusion.t_max
    ns = diffusion.noise_schedule
    alpha_T = float(ns.alpha_T)
    rho_T = float(ns.rho_T)
    rho_T2 = rho_T * rho_T

    # --- Karras rho=0.6 schedule (ECSI's native setup for e2h) ------------
    n = len(ts)
    t_lo = sigma_min_ecsi
    t_hi = t_max - sigma_max_offset
    min_inv_rho = t_lo ** (1.0 / rho_k)
    max_inv_rho = t_hi ** (1.0 / rho_k)
    ramp = torch.linspace(0.0, 1.0, n, device=x.device, dtype=torch.float64)
    ts_k = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho_k
    ts = torch.cat([ts_k, torch.tensor([float(diffusion.t_min)], device=x.device, dtype=ts_k.dtype)])

    x_T = x
    path = [x.detach().cpu()]
    pred_x0 = []
    ones = x.new_ones([x.shape[0]])
    indices = range(len(ts) - 1)
    indices = tqdm(indices, disable=(dist.get_rank() != 0))

    generator = BatchedSeedGenerator(seed)
    first_noise = generator.randn_like(x)  # return-contract compatibility

    def _abc_and_deriv(t_scalar):
        \"\"\"Analytical (a, b, c) and their t-derivatives at scalar t for VP.

        VP formulas (see dbim-codebase/ddbm/karras_diffusion.py VPNoiseSchedule):
            alpha(t)   = exp(-0.5 β_min t - 0.25 β_d t^2)
            alpha'(t)  = alpha(t) * f_fn(t)               f_fn = -0.5*(β_min+β_d*t)
            rho(t)     = sqrt(exp(β_min t + 0.5 β_d t^2) - 1)
            rho'(t)    = 0.5 * (rho^2 + 1) * g2_fn(t) / rho     g2_fn = β_min + β_d*t
            a(t) = α_bar * ρ^2 / ρ_T^2, α_bar = α/α_T
            b(t) = α * ρ_bar^2 / ρ_T^2, ρ_bar^2 = ρ_T^2 - ρ^2
            c(t) = α * ρ_bar * ρ / ρ_T
        \"\"\"
        t_clamped = t_scalar.clamp(min=1e-6, max=t_max - 1e-6)
        t = t_clamped * ones
        alpha, alpha_bar, rho, rho_bar = ns.get_alpha_rho(t)
        alpha = append_dims(alpha, x.ndim)
        alpha_bar = append_dims(alpha_bar, x.ndim)
        rho = append_dims(rho, x.ndim)
        rho_bar = append_dims(rho_bar, x.ndim)

        f_t, g2_t = ns.get_f_g2(t)
        f_t = append_dims(f_t, x.ndim)
        g2_t = append_dims(g2_t, x.ndim)

        alpha_d = alpha * f_t
        rho_d = 0.5 * (rho**2 + 1.0) * g2_t / rho

        rho_sq = rho * rho
        rho_bar_sq = rho_bar * rho_bar
        a = alpha_bar * rho_sq / rho_T2
        b = alpha * rho_bar_sq / rho_T2
        c = alpha * rho_bar * rho / rho_T

        alpha_bar_d = alpha_d / alpha_T
        rho_bar_sq_d = -2.0 * rho * rho_d
        rho_bar_d = -rho * rho_d / rho_bar

        a_d = (alpha_bar_d * rho_sq + alpha_bar * 2.0 * rho * rho_d) / rho_T2
        b_d = (alpha_d * rho_bar_sq + alpha * rho_bar_sq_d) / rho_T2
        c_d = (alpha_d * rho_bar * rho + alpha * rho_bar_d * rho + alpha * rho_bar * rho_d) / rho_T

        return (a, b, c), (a_d, b_d, c_d)

    nfe = 0
    n_steps = len(ts) - 1
    for step_idx, i in enumerate(indices):
        s = ts[i]
        t_next = ts[i + 1]

        x0_hat = denoiser(x, s * ones)
        if mask is not None:
            x0_hat = x0_hat * mask + x_T * (1 - mask)

        (a_s, b_s, c_s), (a_d, b_d, c_d) = _abc_and_deriv(s)

        if step_idx >= n_steps - 2:
            # Last 2 iterations: DBIM deterministic transition.
            a_t, b_t, c_t = [append_dims(v, x.ndim) for v in ns.get_abc(t_next * ones)]
            x = b_t * x0_hat + a_t * x_T + (c_t / c_s) * (x - b_s * x0_hat - a_s * x_T)
        else:
            # Euler-SDE step (ECSI).
            eps = churn * (c_s * c_d - (b_d / b_s) * c_s**2)
            eps = eps.clamp(min=0)

            z_hat = (x - b_s * x0_hat - a_s * x_T) / c_s
            drift = b_d * x0_hat + a_d * x_T + (c_d + eps / c_s) * z_hat
            diff_coef = (2.0 * eps).sqrt()

            dt = t_next - s
            step_noise = generator.randn_like(x)
            x = x + drift * dt + diff_coef * step_noise * dt.abs().sqrt()

        if mask is not None:
            # Preserve the known (mask=0) region at x_T after every update.
            # For inpainting, unmasked border pixels must stay put; without
            # this the SDE noise accumulates on the known region and FID
            # explodes in the local inpainting harness.
            x = x * mask + x_T * (1 - mask)

        path.append(x.detach().cpu())
        pred_x0.append(x0_hat.detach().cpu())
        nfe += 1

    return x, path, nfe, pred_x0, ts, first_noise
"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 448,
        "end_line": 470,
        "content": _ECSI_ALGORITHM,
    },
]
