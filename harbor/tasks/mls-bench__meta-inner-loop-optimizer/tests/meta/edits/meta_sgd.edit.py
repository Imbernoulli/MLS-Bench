"""Meta-SGD baseline — rigorous codebase edit ops.

Meta-SGD (Li et al., 2017): learns a per-parameter learning rate vector
that is meta-optimized by the outer loop. Each parameter gets its own
learnable scalar learning rate, initialized to inner_lr.

Reference: learn2learn/algorithms/meta_sgd.py
Paper: "Meta-SGD: Learning to Learn Quickly for Few-Shot Learning"
       (Li, Zhou, Chen, Li, 2017, arXiv:1707.09835)

Reported accuracies vary with backbone, splits, and adaptation protocol; this
benchmark ranks methods using the local task leaderboard.
"""

_FILE = "learn2learn/custom_maml.py"

_META_SGD = """\
class InnerLoopOptimizer:
    \"\"\"Meta-SGD inner-loop optimizer (Li et al., 2017).

    Learns a per-parameter learning rate vector that is meta-optimized
    by the outer loop. Each model parameter gets a corresponding learnable
    learning rate tensor of the same shape, initialized to inner_lr.
    \"\"\"

    def __init__(self, model: nn.Module, inner_lr: float = INNER_LR):
        self.inner_lr = inner_lr
        # Create per-parameter learnable learning rates
        self.lrs = nn.ParameterList([
            nn.Parameter(torch.ones_like(p) * inner_lr)
            for p in model.parameters()
        ])

    def adapt(self, model: nn.Module, support_x: Tensor, support_y: Tensor,
              n_steps: int) -> nn.Module:
        model.train()
        for _ in range(n_steps):
            loss = F.cross_entropy(model(support_x), support_y)
            grads = torch.autograd.grad(
                loss, model.parameters(), create_graph=True
            )
            updates = [-lr * g for g, lr in zip(grads, self.lrs)]
            l2l.update_module(model, updates=updates)
        return model

    def meta_parameters(self) -> List[Tensor]:
        return list(self.lrs.parameters())

"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 177,
        "end_line": 254,
        "content": _META_SGD,
    },
]
