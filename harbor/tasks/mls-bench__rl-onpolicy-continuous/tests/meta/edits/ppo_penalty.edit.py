"""PPO-Penalty baseline — rigorous codebase edit ops.

PPO with adaptive KL penalty instead of clipped surrogate objective.
From the original PPO paper (Schulman et al., 2017), Section 4.

Same agent architecture as PPO-Clip. Differs only in loss computation:
uses adaptive KL divergence penalty instead of ratio clipping.

Ops are ordered bottom-to-top so line numbers stay stable.
"""

_FILE = "cleanrl/cleanrl/custom_onpolicy_continuous.py"

_PPO_PENALTY_CODE = """\
    def get_action_and_value(self, obs, action=None):
        action_mean = self.actor_mean(obs)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(obs)


def compute_losses(agent, mb_obs, mb_actions, mb_logprobs, mb_advantages, mb_returns, mb_values, args):
    \"\"\"PPO-Penalty: adaptive KL penalty instead of clipped surrogate.\"\"\"
    if not hasattr(agent, '_kl_beta'):
        agent._kl_beta = 0.5
        agent._target_kl = 0.01

    _, newlogprob, entropy, newvalue = agent.get_action_and_value(mb_obs, mb_actions)
    logratio = newlogprob - mb_logprobs
    ratio = logratio.exp()

    # KL divergence — WITH gradient for the penalty term
    kl = ((ratio - 1) - logratio).mean()

    with torch.no_grad():
        approx_kl = kl.detach()
        clipfrac = ((ratio - 1.0).abs() > args.clip_coef).float().mean().item()

    # Policy loss — KL-penalized (no clipping)
    pg_loss = -(mb_advantages * ratio).mean() + agent._kl_beta * kl

    # Adapt KL penalty coefficient
    with torch.no_grad():
        if approx_kl > 1.5 * agent._target_kl:
            agent._kl_beta = min(agent._kl_beta * 2.0, 100.0)
        elif approx_kl < agent._target_kl / 1.5:
            agent._kl_beta = max(agent._kl_beta / 2.0, 1e-4)

    # Value loss — simple MSE (no clipping)
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
        "content": _PPO_PENALTY_CODE,
    },
]
