"""Random baseline: uniform random position selection, argmax tokens, linear schedule.

This is the default strategy (identical to the template default).
"""

_FILE = "LLaDA/custom_demask_eval.py"

_CONTENT = """\
class DemaskStrategy:
    def __init__(self, prompt_length: int = 0, temperature: float = 1.0):
        self.prompt_length = prompt_length
        self.temperature = temperature
        self.prev_top1_prob = None
        self.prev_top1_id = None

    def step(self, logits, x, mask, step, total_steps):
        B, L, V = logits.shape
        n_masked = mask.sum(dim=-1)
        remaining = max(total_steps - step, 1)
        n_to_unmask = (n_masked.float() / remaining).ceil().clamp(min=1).long()

        noise = torch.rand(B, L, device=mask.device)
        noise[~mask] = 2.0
        ranked = noise.argsort(dim=-1)

        unmask = torch.zeros_like(mask)
        for b in range(B):
            k = min(int(n_to_unmask[b].item()), int(n_masked[b].item()))
            if k > 0:
                unmask[b, ranked[b, :k]] = True

        tokens = logits.argmax(dim=-1)
        return unmask, tokens
"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 59,
        "end_line": 122,
        "content": _CONTENT,
    },
]
