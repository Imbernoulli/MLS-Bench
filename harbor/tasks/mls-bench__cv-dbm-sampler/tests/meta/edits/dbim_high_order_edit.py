_FILE = "dbim-codebase/ddbm/karras_diffusion.py"

_DBIM_ALGORITHM = """\
@torch.no_grad()
def sample_dbim(
    denoiser,
    diffusion,
    x,
    ts,
    mask=None,
    order=2,
    lower_order_final=True,
    seed=None,
    **kwargs,
):
    if order not in [2, 3]:
        order=2
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

    u = diffusion.t_max
    if u == 1.0:
        u -= 5e-5
    u = [u for _ in range(order - 1)]
    xu_hat = [x0_hat.detach().clone() for _ in range(order - 1)]

    for _, i in enumerate(indices):
        s = ts[i]
        t = ts[i + 1]

        # First Order Update, t < s
        if (lower_order_final and i + 1 == len(ts) - 1) or (i == 0):
            if dist.get_rank() == 0:
                print("Step order 1")
            a_s, b_s, c_s = [append_dims(item, x0_hat.ndim) for item in diffusion.noise_schedule.get_abc(s * ones)]
            a_t, b_t, c_t = [append_dims(item, x0_hat.ndim) for item in diffusion.noise_schedule.get_abc(t * ones)]

            tmp_var = c_t / c_s
            coeff_xs = tmp_var
            coeff_x0_hat = b_t - tmp_var * b_s
            coeff_xT = a_t - tmp_var * a_s

            x0_hat = denoiser(x, s * ones)
            if mask is not None:
                x0_hat = x0_hat * mask + x_T * (1 - mask)
            nfe += 1
            x_old = x
            x = coeff_xs * x_old + coeff_x0_hat * x0_hat + coeff_xT * x_T

        # Second Order Update, t < s < u
        elif order == 2 or i == 1:
            if dist.get_rank() == 0:
                print("Step order 2")
            a_u, b_u, c_u = [append_dims(item, x0_hat.ndim) for item in diffusion.noise_schedule.get_abc(u[-1] * ones)]
            a_s, b_s, c_s = [append_dims(item, x0_hat.ndim) for item in diffusion.noise_schedule.get_abc(s * ones)]
            a_t, b_t, c_t = [append_dims(item, x0_hat.ndim) for item in diffusion.noise_schedule.get_abc(t * ones)]
            lambda_u, lambda_s, lambda_t = (
                torch.log(b_u / c_u),
                torch.log(b_s / c_s),
                torch.log(b_t / c_t),
            )

            x0_hat = denoiser(x, s * ones)
            if mask is not None:
                x0_hat = x0_hat * mask + x_T * (1 - mask)
            nfe += 1
            h = lambda_t - lambda_s
            h2 = lambda_s - lambda_u
            integral = torch.exp(lambda_t) * (
                (1 - torch.exp(-h)) * x0_hat + (torch.exp(-h) + h - 1) * (x0_hat - xu_hat[-1]) / h2
            )
            x_old = x
            x = x_old * (c_t / c_s) + x_T * (a_t - a_s * (c_t / c_s)) + c_t * integral

        elif order == 3:
            if dist.get_rank() == 0:
                print("Step order 3")
            a_u1, b_u1, c_u1 = [
                append_dims(item, x0_hat.ndim) for item in diffusion.noise_schedule.get_abc(u[-1] * ones)
            ]
            a_u2, b_u2, c_u2 = [
                append_dims(item, x0_hat.ndim) for item in diffusion.noise_schedule.get_abc(u[-2] * ones)
            ]
            a_s, b_s, c_s = [append_dims(item, x0_hat.ndim) for item in diffusion.noise_schedule.get_abc(s * ones)]
            a_t, b_t, c_t = [append_dims(item, x0_hat.ndim) for item in diffusion.noise_schedule.get_abc(t * ones)]
            lambda_u2, lambda_u1, lambda_s, lambda_t = (
                torch.log(b_u2 / c_u2),
                torch.log(b_u1 / c_u1),
                torch.log(b_s / c_s),
                torch.log(b_t / c_t),
            )
            x0_hat = denoiser(x, s * ones)
            if mask is not None:
                x0_hat = x0_hat * mask + x_T * (1 - mask)
            nfe += 1

            h = lambda_t - lambda_s
            h1 = lambda_s - lambda_u1
            h2 = lambda_u1 - lambda_u2
            dx0_hat = ((x0_hat - xu_hat[-1]) * (2 * h1 + h2) / h1 - (xu_hat[-1] - xu_hat[-2]) * h1 / h2) / (h1 + h2)
            d2x0_hat = 2 * ((x0_hat - xu_hat[-1]) / h1 - (xu_hat[-1] - xu_hat[-2]) / h2) / (h1 + h2)
            integral = torch.exp(lambda_t) * (
                (1 - torch.exp(-h)) * x0_hat
                + (torch.exp(-h) + h - 1) * dx0_hat
                + (h**2 / 2 - h + 1 - torch.exp(-h)) * d2x0_hat
            )
            x_old = x
            x = x_old * (c_t / c_s) + x_T * (a_t - a_s * (c_t / c_s)) + c_t * integral

        u.append(s)
        u.pop(0)
        xu_hat.append(x0_hat)
        xu_hat.pop(0)

        path.append(x.detach().cpu())
        pred_x0.append(x0_hat.detach().cpu())

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
