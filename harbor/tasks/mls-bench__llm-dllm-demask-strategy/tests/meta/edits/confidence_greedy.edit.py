"""Confidence-greedy baseline (LLaDA default "low_confidence" remasking).

Reference: Nie et al., "Large Language Diffusion Models" (LLaDA, arXiv 2502.09992).
At each step, unmask the top-k positions by max-probability confidence —
with k given by the uniform schedule num_transfer_tokens. No KL tracking.
"""

_FILE = "LLaDA/custom_demask_eval.py"

_CONTENT = '''\
class DemaskDecoder:
    """low_confidence remasking: unmask top-k positions by confidence."""

    def __init__(self, mask_id: int, temperature: float = 0.0,
                 conf_threshold: float = 0.9, kl_threshold: float = 0.01,
                 history_length: int = 2):
        self.mask_id = mask_id
        self.temperature = temperature

    @torch.no_grad()
    def decode(self, model, input_ids, gen_length: int, steps: int,
               block_length: int):
        mid = self.mask_id
        x = torch.full((1, input_ids.shape[1] + gen_length), mid,
                       dtype=torch.long, device=model.device)
        x[:, :input_ids.shape[1]] = input_ids.clone()
        assert gen_length % block_length == 0
        num_blocks = gen_length // block_length
        assert steps % num_blocks == 0
        steps_per_block = steps // num_blocks
        used = 0
        for b in range(num_blocks):
            bs = input_ids.shape[1] + b * block_length
            be = bs + block_length
            num_xfer = get_num_transfer_tokens(
                (x[:, bs:be] == mid), steps_per_block)
            for step in range(steps_per_block):
                mask_idx = (x == mid)
                block_m = torch.zeros_like(mask_idx)
                block_m[:, bs:be] = True
                mask_idx = mask_idx & block_m
                if not mask_idx.any():
                    break
                logits = model(x).logits
                p_curr = F.softmax(logits.to(torch.float64), dim=-1)
                x0 = torch.argmax(p_curr, dim=-1)
                conf = torch.gather(p_curr, -1, x0.unsqueeze(-1)).squeeze(-1)
                xfer = torch.zeros_like(x0, dtype=torch.bool)
                for j in range(conf.shape[0]):
                    c = conf[j].clone()
                    c[~mask_idx[j]] = -float("inf")
                    _, topk = torch.topk(c, int(num_xfer[j, step].item()))
                    xfer[j, topk] = True
                x = torch.where(xfer, x0, x)
                used += 1
        return x, used
'''

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 59,
        "end_line": 151,
        "content": _CONTENT,
    },
]
