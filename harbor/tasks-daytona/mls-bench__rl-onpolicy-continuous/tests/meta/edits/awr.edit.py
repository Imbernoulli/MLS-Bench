"""AWR (Advantage Weighted Regression) baseline — rigorous codebase edit ops.

Replaces PPO's clipped surrogate with advantage-weighted regression.
Policy is trained via supervised learning on actions reweighted by
exponentiated advantages.

Reference: Peng et al., "Advantage-Weighted Regression: Simple and Scalable
Off-Policy Reinforcement Learning", 2019.

Ops are ordered bottom-to-top so line numbers stay stable.
"""

_FILE = "cleanrl/cleanrl/custom_onpolicy_continuous.py"

_AWR_CODE = """\
    def get_action_and_value(self, obs, action=None):
        action_mean = self.actor_mean(obs)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(obs)


def compute_losses(agent, mb_obs, mb_actions, mb_logprobs, mb_advantages, mb_returns, mb_values, args):
    \"\"\"AWR: advantage-weighted regression loss.\"\"\"
    _awr_beta = 0.05
    _awr_max_weight = 20.0

    _, newlogprob, entropy, newvalue = agent.get_action_and_value(mb_obs, mb_actions)
    logratio = newlogprob - mb_logprobs
    ratio = logratio.exp()

    with torch.no_grad():
        approx_kl = ((ratio - 1) - logratio).mean()
        clipfrac = ((ratio - 1.0).abs() > args.clip_coef).float().mean().item()

    # Compute advantage weights: exp(advantage / beta), clamped for stability
    with torch.no_grad():
        weights = torch.exp(mb_advantages / _awr_beta)
        weights = torch.clamp(weights, max=_awr_max_weight)
        weights = weights / (weights.sum() + 1e-8) * weights.numel()

    # Policy loss — advantage-weighted regression (supervised)
    pg_loss = -(newlogprob * weights).mean()

    # Value loss — simple MSE
    newvalue = newvalue.view(-1)
    v_loss = 0.5 * ((newvalue - mb_returns) ** 2).mean()

    entropy_loss = entropy.mean()
    loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

    return loss, pg_loss, v_loss, entropy_loss, approx_kl, clipfrac
"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 175,
        "end_line": 221,
        "content": _AWR_CODE,
    },
]
