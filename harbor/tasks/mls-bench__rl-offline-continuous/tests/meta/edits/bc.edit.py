"""BC (Behavior Cloning) baseline — rigorous codebase edit ops.

10%-BC: keeps only the top 10% of trajectories (sorted by discounted return)
and trains a deterministic actor via MSE loss on the filtered data.

Reference: CORL/algorithms/offline/any_percent_bc.py
Key: keep_best_trajectories(dataset, frac=0.1, discount=0.99)

Only replaces the OfflineAlgorithm class.  Network definitions (DeterministicActor,
Actor, Critic, ValueFunction) are kept as-is from the template.
"""

_FILE = "CORL/algorithms/offline/custom.py"

_BC_ALGORITHM = """\
class OfflineAlgorithm:
    \"\"\"10%-BC — Behavior Cloning on top 10% of trajectories by discounted return.\"\"\"

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

        # Filter replay buffer to top 10% of trajectories
        if replay_buffer is not None:
            self._keep_best_trajectories(replay_buffer, frac=0.1, discount=discount)

        # BC only needs a deterministic actor — no critics
        self.actor = DeterministicActor(state_dim, action_dim, max_action).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)

    @staticmethod
    def _keep_best_trajectories(buf, frac: float, discount: float, max_episode_steps: int = 1000):
        \"\"\"Filter replay buffer in-place to keep only the top `frac` trajectories.\"\"\"
        n = buf._size
        rewards = buf._rewards[:n].cpu().numpy().flatten()
        dones = buf._dones[:n].cpu().numpy().flatten()

        # Identify trajectories and compute discounted returns
        ids_by_trajectories = []
        returns = []
        cur_ids = []
        cur_return = 0.0
        reward_scale = 1.0
        for i in range(n):
            cur_return += reward_scale * rewards[i]
            cur_ids.append(i)
            reward_scale *= discount
            if dones[i] == 1.0 or len(cur_ids) == max_episode_steps:
                ids_by_trajectories.append(list(cur_ids))
                returns.append(cur_return)
                cur_ids = []
                cur_return = 0.0
                reward_scale = 1.0
        if cur_ids:
            ids_by_trajectories.append(list(cur_ids))
            returns.append(cur_return)

        # Sort by return descending, keep top frac
        sort_ord = np.argsort(returns)[::-1]
        top_trajs = sort_ord[: max(1, int(frac * len(sort_ord)))]

        order = []
        for i in top_trajs:
            order += ids_by_trajectories[i]
        order = np.array(order)

        # Reorder buffer in-place
        buf._states[:len(order)] = buf._states[order]
        buf._actions[:len(order)] = buf._actions[order]
        buf._rewards[:len(order)] = buf._rewards[order]
        buf._next_states[:len(order)] = buf._next_states[order]
        buf._next_actions[:len(order)] = buf._next_actions[order]
        buf._dones[:len(order)] = buf._dones[order]
        buf._size = len(order)
        buf._pointer = len(order)

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
        "start_line": 306,
        "end_line": 397,
        "content": _BC_ALGORITHM,
    },
]
