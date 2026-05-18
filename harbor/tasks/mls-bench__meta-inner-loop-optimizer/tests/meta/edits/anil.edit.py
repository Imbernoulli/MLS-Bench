"""ANIL baseline — rigorous codebase edit ops.

ANIL (Raghu et al., 2019): Almost No Inner Loop — only adapts the
classification head parameters during inner-loop adaptation, freezing
the feature extractor backbone. This is based on the finding that
feature reuse (not rapid learning) is the dominant factor in MAML.

Reference: Raghu et al., "Rapid Learning or Feature Reuse? Towards
           Understanding the Effectiveness of MAML" (ICLR 2020)
Paper: arXiv:1909.09157

Reported accuracies vary with backbone, splits, and adaptation protocol; this
benchmark ranks methods using the local task leaderboard.
"""

_FILE = "learn2learn/custom_maml.py"

_ANIL = """\
class InnerLoopOptimizer:
    \"\"\"ANIL inner-loop optimizer (Raghu et al., 2019).

    Almost No Inner Loop: only adapts the final classification head
    during inner-loop adaptation. The feature extractor backbone is
    frozen, relying on feature reuse from the meta-initialization.
    \"\"\"

    def __init__(self, model: nn.Module, inner_lr: float = INNER_LR):
        self.inner_lr = inner_lr
        # Identify head parameters (the last linear layer: classifier)
        # CNN4 structure: features (CNN4Backbone) -> classifier (Linear)
        self._head_param_names = set()
        for name, _ in model.named_parameters():
            if "classifier" in name:
                self._head_param_names.add(name)

    def adapt(self, model: nn.Module, support_x: Tensor, support_y: Tensor,
              n_steps: int) -> nn.Module:
        model.train()
        for _ in range(n_steps):
            # Re-identify head params each step because l2l.update_module
            # replaces parameter objects (new ids), so stale references
            # from a previous step would cause all updates to be zero.
            head_params = []
            head_ids = set()
            for name, p in model.named_parameters():
                if name in self._head_param_names:
                    head_params.append(p)
                    head_ids.add(id(p))

            loss = F.cross_entropy(model(support_x), support_y)
            grads = torch.autograd.grad(
                loss, head_params, create_graph=True
            )
            grad_map = {id(p): g for p, g in zip(head_params, grads)}
            updates = [
                -self.inner_lr * grad_map[id(p)] if id(p) in head_ids
                else torch.zeros_like(p)
                for p in model.parameters()
            ]
            l2l.update_module(model, updates=updates)
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
        "content": _ANIL,
    },
]
