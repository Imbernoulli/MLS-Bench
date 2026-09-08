_FILE = "dbim-codebase/ddbm/karras_diffusion.py"
_DBIM_ALGORITHM = """\
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
    x_T = x
    path = []
    pred_x0 = []

    ones = x.new_ones([x.shape[0]])
    indices = range(len(ts) - 1)
    indices = tqdm(indices, disable=(dist.get_rank() != 0))

    nfe = 0
    x0_hat = denoiser(x, diffusion.t_max * ones)
    generator = BatchedSeedGenerator(seed)
    noise = generator.randn_like(x0_hat)
    first_noise = noise
    if mask is not None:
        x0_hat = x0_hat * mask + x_T * (1 - mask)
    x = diffusion.bridge_sample(x0_hat, x_T, ts[0] * ones, noise)
    path.append(x.detach().cpu())
    pred_x0.append(x0_hat.detach().cpu())
    nfe += 1

    for _, i in enumerate(indices):
        s = ts[i]
        t = ts[i + 1]

        x0_hat = denoiser(x, s * ones)
        if mask is not None:
            x0_hat = x0_hat * mask + x_T * (1 - mask)

        a_s, b_s, c_s = [append_dims(item, x0_hat.ndim) for item in diffusion.noise_schedule.get_abc(s * ones)]
        a_t, b_t, c_t = [append_dims(item, x0_hat.ndim) for item in diffusion.noise_schedule.get_abc(t * ones)]

        _, _, rho_s, _ = [append_dims(item, x0_hat.ndim) for item in diffusion.noise_schedule.get_alpha_rho(s * ones)]
        alpha_t, _, rho_t, _ = [
            append_dims(item, x0_hat.ndim) for item in diffusion.noise_schedule.get_alpha_rho(t * ones)
        ]

        omega_st = eta * (alpha_t * rho_t) * (1 - rho_t**2 / rho_s**2).sqrt()
        tmp_var = (c_t**2 - omega_st**2).sqrt() / c_s
        coeff_xs = tmp_var
        coeff_x0_hat = b_t - tmp_var * b_s
        coeff_xT = a_t - tmp_var * a_s

        noise = generator.randn_like(x0_hat)

        x = coeff_x0_hat * x0_hat + coeff_xT * x_T + coeff_xs * x + (1 if i != len(ts) - 2 else 0) * omega_st * noise

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
        "content": _DBIM_ALGORITHM,
    },
]
