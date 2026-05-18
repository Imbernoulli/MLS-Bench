"""A2C (Advantage Actor-Critic) baseline — rigorous codebase edit ops.

Vanilla advantage actor-critic without any trust region mechanism.
Policy gradient uses raw advantages without importance sampling or clipping.

Ops are ordered bottom-to-top so line numbers stay stable.
"""

_FILE = "cleanrl/cleanrl/custom_onpolicy_continuous.py"

_A2C_CODE = """\
    def get_action_and_value(self, obs, action=None):
        action_mean = self.actor_mean(obs)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(obs)


def compute_losses(agent, mb_obs, mb_actions, mb_logprobs, mb_advantages, mb_returns, mb_values, args):
    \"\"\"A2C: vanilla policy gradient with advantage baseline.\"\"\"
    _, newlogprob, entropy, newvalue = agent.get_action_and_value(mb_obs, mb_actions)
    logratio = newlogprob - mb_logprobs
    ratio = logratio.exp()

    with torch.no_grad():
        approx_kl = ((ratio - 1) - logratio).mean()
        clipfrac = ((ratio - 1.0).abs() > args.clip_coef).float().mean().item()

    # Policy loss — vanilla REINFORCE with advantage baseline (no clipping, no IS ratio)
    pg_loss = -(newlogprob * mb_advantages).mean()

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
        "content": _A2C_CODE,
    },
]
