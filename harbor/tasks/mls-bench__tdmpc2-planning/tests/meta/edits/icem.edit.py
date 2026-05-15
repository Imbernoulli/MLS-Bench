"""Improved Cross-Entropy Method (iCEM) baseline -- rigorous codebase edit ops.

Replaces MPPI with iCEM: uses temporally correlated (colored) noise
for smoother action sequences, keeps elite actions across iterations,
and decays the noise magnitude over iterations.

Reference: Pinneri et al., "Sample-efficient Cross-Entropy Method for
Real-time Planning", CoRL 2020 / PMLR 2021.
"""

_FILE = "tdmpc2/tdmpc2/common/custom_planner.py"

_ICEM = """\
@torch.no_grad()
def custom_plan(agent, obs, t0=False, eval_mode=False, task=None):
    \"\"\"iCEM baseline -- improved CEM with colored noise and keep-elites.\"\"\"
    cfg = agent.cfg
    colored_noise_beta = 0.5  # temporal correlation strength
    noise_decay = 0.9  # per-iteration noise decay factor
    keep_fraction = 0.1  # fraction of elites kept across iterations

    # Sample policy trajectories as warm-starts
    z = agent.model.encode(obs, task)
    if cfg.num_pi_trajs > 0:
        pi_actions = torch.empty(
            cfg.horizon, cfg.num_pi_trajs, cfg.action_dim,
            device=agent.device,
        )
        _z = z.repeat(cfg.num_pi_trajs, 1)
        for t in range(cfg.horizon - 1):
            pi_actions[t], _ = agent.model.pi(_z, task)
            _z = agent.model.next(_z, pi_actions[t], task)
        pi_actions[-1], _ = agent.model.pi(_z, task)

    # Initialize sampling distribution
    z = z.repeat(cfg.num_samples, 1)
    mean = torch.zeros(cfg.horizon, cfg.action_dim, device=agent.device)
    std = torch.full(
        (cfg.horizon, cfg.action_dim), cfg.max_std,
        dtype=torch.float, device=agent.device,
    )
    if not t0:
        mean[:-1] = agent._prev_mean[1:]
    actions = torch.empty(
        cfg.horizon, cfg.num_samples, cfg.action_dim,
        device=agent.device,
    )
    if cfg.num_pi_trajs > 0:
        actions[:, :cfg.num_pi_trajs] = pi_actions

    n_keep = max(1, int(cfg.num_elites * keep_fraction))
    kept_actions = None  # elites kept from previous iteration
    noise_scale = 1.0

    # Iterate iCEM
    for iteration in range(cfg.iterations):
        n_new = cfg.num_samples - cfg.num_pi_trajs
        if kept_actions is not None:
            n_new = n_new - kept_actions.shape[1]

        # Generate temporally correlated (colored) noise
        white_noise = torch.randn(
            cfg.horizon, n_new, cfg.action_dim,
            device=agent.device,
        )
        # Apply temporal smoothing: exponential moving average along horizon
        colored_noise = torch.zeros_like(white_noise)
        colored_noise[0] = white_noise[0]
        for t in range(1, cfg.horizon):
            colored_noise[t] = (
                colored_noise_beta * colored_noise[t - 1]
                + (1 - colored_noise_beta) * white_noise[t]
            )

        # Sample actions
        actions_sample = (
            mean.unsqueeze(1)
            + noise_scale * std.unsqueeze(1) * colored_noise
        )
        actions_sample = actions_sample.clamp(-1, 1)

        # Combine: policy trajs + kept elites + new samples
        start_idx = cfg.num_pi_trajs
        if kept_actions is not None:
            actions[:, start_idx : start_idx + kept_actions.shape[1]] = kept_actions
            start_idx += kept_actions.shape[1]
        actions[:, start_idx : start_idx + n_new] = actions_sample

        if cfg.multitask:
            actions = actions * agent.model._action_masks[task]

        # Evaluate trajectories and select elites
        value = agent._estimate_value(z, actions, task).nan_to_num(0)
        elite_idxs = torch.topk(
            value.squeeze(1), cfg.num_elites, dim=0,
        ).indices
        elite_actions = actions[:, elite_idxs]

        # Keep top elites for next iteration
        kept_actions = elite_actions[:, :n_keep]

        # Update distribution (simple CEM-style)
        mean = elite_actions.mean(dim=1)
        std = elite_actions.std(dim=1).clamp(cfg.min_std, cfg.max_std)
        if cfg.multitask:
            mean = mean * agent.model._action_masks[task]
            std = std * agent.model._action_masks[task]

        # Decay noise for refinement
        noise_scale *= noise_decay

    # Select action: use the mean
    a = mean[0]
    if not eval_mode:
        a = a + std[0] * torch.randn(cfg.action_dim, device=agent.device)
    agent._prev_mean.copy_(mean)
    return a.clamp(-1, 1)
"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 15,
        "end_line": 120,
        "content": _ICEM,
    },
]
