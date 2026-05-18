"""MPPI (Model Predictive Path Integral) baseline -- rigorous codebase edit ops.

Keeps the default MPPI planning algorithm as a baseline. MPPI uses
softmax-weighted elite statistics to update the sampling distribution,
and Gumbel-softmax sampling for final action selection.

Reference: Williams et al., "Information Theoretic MPC for Model-Based
Reinforcement Learning", ICRA 2017.
"""

_FILE = "tdmpc2/tdmpc2/common/custom_planner.py"

_MPPI = """\
@torch.no_grad()
def custom_plan(agent, obs, t0=False, eval_mode=False, task=None):
    \"\"\"MPPI baseline -- Model Predictive Path Integral control.\"\"\"
    cfg = agent.cfg

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

    # Iterate MPPI
    for _ in range(cfg.iterations):
        # Sample actions from Gaussian
        r = torch.randn(
            cfg.horizon, cfg.num_samples - cfg.num_pi_trajs,
            cfg.action_dim, device=agent.device,
        )
        actions_sample = mean.unsqueeze(1) + std.unsqueeze(1) * r
        actions_sample = actions_sample.clamp(-1, 1)
        actions[:, cfg.num_pi_trajs:] = actions_sample
        if cfg.multitask:
            actions = actions * agent.model._action_masks[task]

        # Evaluate trajectories and select elites
        value = agent._estimate_value(z, actions, task).nan_to_num(0)
        elite_idxs = torch.topk(
            value.squeeze(1), cfg.num_elites, dim=0,
        ).indices
        elite_value = value[elite_idxs]
        elite_actions = actions[:, elite_idxs]

        # Update sampling distribution (softmax-weighted)
        max_value = elite_value.max(0).values
        score = torch.exp(cfg.temperature * (elite_value - max_value))
        score = score / score.sum(0)
        mean = (
            (score.unsqueeze(0) * elite_actions).sum(dim=1)
            / (score.sum(0) + 1e-9)
        )
        std = (
            (
                score.unsqueeze(0)
                * (elite_actions - mean.unsqueeze(1)) ** 2
            ).sum(dim=1)
            / (score.sum(0) + 1e-9)
        ).sqrt()
        std = std.clamp(cfg.min_std, cfg.max_std)
        if cfg.multitask:
            mean = mean * agent.model._action_masks[task]
            std = std * agent.model._action_masks[task]

    # Select action from elites via Gumbel sampling
    rand_idx = math.gumbel_softmax_sample(score.squeeze(1))
    actions = torch.index_select(elite_actions, 1, rand_idx).squeeze(1)
    a, std_final = actions[0], std[0]
    if not eval_mode:
        a = a + std_final * torch.randn(cfg.action_dim, device=agent.device)
    agent._prev_mean.copy_(mean)
    return a.clamp(-1, 1)
"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 15,
        "end_line": 120,
        "content": _MPPI,
    },
]
