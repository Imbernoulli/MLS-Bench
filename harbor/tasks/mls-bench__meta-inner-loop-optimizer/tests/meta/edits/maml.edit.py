"""MAML baseline — rigorous codebase edit ops.

Vanilla MAML (Finn et al., 2017): fixed learning rate SGD applied to
all model parameters in the inner loop.

Reference: learn2learn/algorithms/maml.py
Paper: "Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks"
       (Finn, Abbeel, Levine, ICML 2017)

Literature reports vary with ConvNet details, data splits, and adaptation
steps; this benchmark ranks methods using the local task leaderboard.
"""

_FILE = "learn2learn/custom_maml.py"

_MAML = """\
class InnerLoopOptimizer:
    \"\"\"MAML inner-loop optimizer (Finn et al., 2017).

    Vanilla SGD with a fixed learning rate applied uniformly to all
    model parameters. This is the standard MAML inner loop.

    Shot-aware LR override: the global INNER_LR=0.5 destabilizes
    full-network adaptation at 1-shot in the local harness. At 5-shot the larger
    support set buffers gradient noise so 0.5 is fine (matches
    learn2learn benchmark default). Use the common 1-shot recipe
    (0.01) only when N_SHOT=1, keep 0.5 for 5-shot.
    \"\"\"

    def __init__(self, model: nn.Module, inner_lr: float = INNER_LR):
        self.inner_lr = 0.01 if N_SHOT == 1 else 0.5

    def adapt(self, model: nn.Module, support_x: Tensor, support_y: Tensor,
              n_steps: int) -> nn.Module:
        model.train()
        for _ in range(n_steps):
            loss = F.cross_entropy(model(support_x), support_y)
            grads = torch.autograd.grad(
                loss, model.parameters(), create_graph=True
            )
            model = l2l.algorithms.maml.maml_update(
                model, lr=self.inner_lr, grads=grads
            )
        return model

    def meta_parameters(self) -> List[Tensor]:
        return []

"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 177,
        "end_line": 254,
        "content": _MAML,
    },
]
