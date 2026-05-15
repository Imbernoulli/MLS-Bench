"""RPO (Robust Policy Optimization) baseline — rigorous codebase edit ops.

Same as PPO but adds uniform noise to action mean during updates.

Reference: cleanrl/cleanrl/rpo_continuous_action.py

Ops are ordered bottom-to-top so line numbers stay stable.
"""

_FILE = "cleanrl/cleanrl/custom_onpolicy_continuous.py"

_RPO_CODE = """\
    def get_action_and_value(self, obs, action=None):
        rpo_alpha = 0.5
        action_mean = self.actor_mean(obs)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        else:
            # RPO: add uniform noise to action mean during update
            z = torch.FloatTensor(action_mean.shape).uniform_(-rpo_alpha, rpo_alpha).to(obs.device)
            action_mean = action_mean + z
            probs = Normal(action_mean, action_std)
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(obs)


def compute_losses(agent, mb_obs, mb_actions, mb_logprobs, mb_advantages, mb_returns, mb_values, args):
    \"\"\"PPO clipped surrogate objective + clipped value loss (same as PPO).\"\"\"
    _, newlogprob, entropy, newvalue = agent.get_action_and_value(mb_obs, mb_actions)
    logratio = newlogprob - mb_logprobs
    ratio = logratio.exp()

    with torch.no_grad():
        approx_kl = ((ratio - 1) - logratio).mean()
        clipfrac = ((ratio - 1.0).abs() > args.clip_coef).float().mean().item()

    # Policy loss — clipped surrogate
    pg_loss1 = -mb_advantages * ratio
    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

    # Value loss — clipped
    newvalue = newvalue.view(-1)
    if args.clip_vloss:
        v_loss_unclipped = (newvalue - mb_returns) ** 2
        v_clipped = mb_values + torch.clamp(
            newvalue - mb_values,
            -args.clip_coef,
            args.clip_coef,
        )
        v_loss_clipped = (v_clipped - mb_returns) ** 2
        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
        v_loss = 0.5 * v_loss_max.mean()
    else:
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
        "content": _RPO_CODE,
    },
]
