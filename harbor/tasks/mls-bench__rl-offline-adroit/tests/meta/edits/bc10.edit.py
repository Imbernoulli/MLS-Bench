"""BC-10% baseline — rigorous codebase edit ops.

Only replaces the OfflineAlgorithm class. Network definitions (DeterministicActor,
Actor, Critic, ValueFunction) are kept as-is from the template.

Note: The 10% filtering is handled by the baseline script calling
any_percent_bc.py with --frac 0.1. This edit file implements standard BC
in the custom_adroit.py template framework.
"""

_FILE = "CORL/algorithms/offline/custom_adroit.py"

_BC_ALGORITHM = """\
class OfflineAlgorithm:
    \"\"\"BC-10%: Behavior Cloning on top 10% trajectories by return.\"\"\"

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        max_action: float,
        replay_buffer=None,
        discount: float = 0.99,
        tau: float = 5e-3,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        alpha_lr: float = 3e-4,
        orthogonal_init: bool = True,
        device: str = "cuda",
    ):
        self.device = device
        self.discount = discount
        self.tau = tau
        self.max_action = max_action
        self.total_it = 0

        # Filter replay buffer to top 10% trajectories by return
        if replay_buffer is not None:
            self._filter_top_frac(replay_buffer, frac=0.1)

        # BC only needs a deterministic actor — no critics
        self.actor = DeterministicActor(state_dim, action_dim, max_action).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)

    def _filter_top_frac(self, buf, frac: float = 0.1):
        \"\"\"Keep only the top `frac` fraction of trajectories by total return.\"\"\"
        n = buf._size
        rewards = buf._rewards[:n].squeeze(-1).cpu().numpy()
        dones = buf._dones[:n].squeeze(-1).cpu().numpy()

        # Identify episode boundaries and compute returns
        episodes = []
        ep_start = 0
        for t in range(n):
            if dones[t] > 0.5 or t == n - 1:
                ep_return = rewards[ep_start:t+1].sum()
                episodes.append((ep_start, t + 1, ep_return))
                ep_start = t + 1

        # Sort by return descending and keep top frac
        episodes.sort(key=lambda x: x[2], reverse=True)
        n_keep = max(1, int(len(episodes) * frac))
        keep_episodes = episodes[:n_keep]

        # Gather indices of transitions to keep
        keep_indices = []
        for start, end, _ in keep_episodes:
            keep_indices.extend(range(start, end))
        keep_indices = np.array(keep_indices, dtype=np.int64)

        # Rebuild buffer with filtered data
        new_size = len(keep_indices)
        buf._states[:new_size] = buf._states[keep_indices]
        buf._actions[:new_size] = buf._actions[keep_indices]
        buf._rewards[:new_size] = buf._rewards[keep_indices]
        buf._next_states[:new_size] = buf._next_states[keep_indices]
        buf._next_actions[:new_size] = buf._next_actions[keep_indices]
        buf._dones[:new_size] = buf._dones[keep_indices]
        buf._size = new_size
        buf._pointer = new_size
        print(f"BC-10%: Filtered to {n_keep}/{len(episodes)} episodes, {new_size} transitions")

    def train(self, batch: TensorBatch) -> Dict[str, float]:
        self.total_it += 1
        states, actions, rewards, next_states, dones, *_ = batch

        # MSE between predicted and dataset actions
        pi = self.actor(states)
        actor_loss = F.mse_loss(pi, actions)

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        return {"actor_loss": actor_loss.item()}
"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 319,
        "end_line": 416,
        "content": _BC_ALGORITHM,
    },
]
