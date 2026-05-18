"""PCGrad baseline (Yu et al., NeurIPS 2020).

Projects conflicting task gradients to reduce interference between tasks.
When two task gradients conflict (negative cosine similarity), each is
projected onto the normal plane of the other.

Adapted to the MultiTaskLoss interface: uses torch.autograd.grad to compute
per-task gradients inside forward, performs conflict resolution, and returns
a loss whose backward pass applies the corrected gradients via a hook.

Reference: Yu et al., "Gradient Surgery for Multi-Task Learning"
(NeurIPS 2020)

Ops ordered bottom-to-top for line stability.
"""

_FILE = "pytorch-vision/custom_mtl.py"

_CONTENT = """\
class MultiTaskLoss(nn.Module):
    \"\"\"PCGrad: Gradient Surgery for Multi-Task Learning (Yu et al., 2020).

    Projects conflicting task gradients onto the normal plane of the
    other when their cosine similarity is negative, reducing gradient
    interference between tasks.
    \"\"\"

    def __init__(self, num_tasks=2):
        super().__init__()
        self._shared_params = None

    def _get_shared_params(self, loss):
        \"\"\"Extract shared model parameters from the computation graph.\"\"\"
        if self._shared_params is not None:
            return self._shared_params
        # Walk the computation graph to find leaf parameters
        params = []
        seen = set()
        def _walk(grad_fn):
            if grad_fn is None:
                return
            for child, _ in grad_fn.next_functions:
                if child is None:
                    continue
                cid = id(child)
                if cid in seen:
                    continue
                seen.add(cid)
                if hasattr(child, 'variable'):
                    p = child.variable
                    if p.requires_grad:
                        params.append(p)
                _walk(child)
        _walk(loss.grad_fn)
        self._shared_params = params
        return params

    def forward(self, fine_loss, coarse_loss, epoch, total_epochs):
        params = self._get_shared_params(fine_loss)
        if len(params) == 0:
            return fine_loss + coarse_loss

        # Compute per-task gradients
        grads_fine = torch.autograd.grad(
            fine_loss, params, retain_graph=True, allow_unused=True,
        )
        grads_coarse = torch.autograd.grad(
            coarse_loss, params, retain_graph=True, allow_unused=True,
        )

        # Flatten gradients into vectors
        g0 = torch.cat([
            g.flatten() if g is not None else torch.zeros_like(p).flatten()
            for g, p in zip(grads_fine, params)
        ])
        g1 = torch.cat([
            g.flatten() if g is not None else torch.zeros_like(p).flatten()
            for g, p in zip(grads_coarse, params)
        ])

        # PCGrad: project conflicting gradients when cosine similarity < 0
        dot = torch.dot(g0, g1)
        if dot < 0:
            # Project each gradient onto the normal plane of the other.
            # Use originals for both projections (symmetric dot product).
            g0_norm_sq = torch.dot(g0, g0) + 1e-12
            g1_norm_sq = torch.dot(g1, g1) + 1e-12
            g0_proj = g0 - (dot / g1_norm_sq) * g1
            g1_proj = g1 - (dot / g0_norm_sq) * g0
            g0 = g0_proj
            g1 = g1_proj

        # Combined projected gradient
        g_pcgrad = g0 + g1

        # Construct a surrogate loss whose gradient equals g_pcgrad.
        # loss = sum_i (g_pcgrad_i * param_i), so grad w.r.t. param_i = g_pcgrad_i
        offset = 0
        surrogate = torch.tensor(0.0, device=fine_loss.device)
        for p in params:
            numel = p.numel()
            chunk = g_pcgrad[offset:offset + numel].reshape(p.shape).detach()
            surrogate = surrogate + (chunk * p).sum()
            offset += numel
        return surrogate
"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 195,
        "end_line": 216,
        "content": _CONTENT,
    },
]
