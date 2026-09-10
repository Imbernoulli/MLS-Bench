"""KLASS baseline (SOTA): stability + confidence, KL-adaptive.

Reference: Kim et al., "KLASS: KL-Guided Fast Inference in Masked Diffusion
Models", NeurIPS 2025 (arXiv:2511.05664).

Faithful port of stable_confident_decode from
https://github.com/shkim0116/KLASS/blob/main/src/model/llada_klass.py
"""

_FILE = "LLaDA/custom_demask_eval.py"

_CONTENT = '''\
class DemaskDecoder:
    """KLASS: stability + confidence, KL-adaptive (Kim et al., NeurIPS 2025)."""

    def __init__(self, mask_id: int, temperature: float = 0.0,
                 conf_threshold: float = 0.9, kl_threshold: float = 0.01,
                 history_length: int = 2):
        self.mask_id = mask_id
        self.temperature = temperature
        self.conf_threshold = conf_threshold
        self.kl_threshold = kl_threshold
        self.history_length = history_length

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
        V = model.lm_head.out_features if hasattr(model, "lm_head") \\
                                       else model.config.vocab_size
        kl_hist = torch.zeros((1, x.shape[1], self.history_length),
                              dtype=torch.float64, device=x.device)
        p_prev = torch.zeros((1, x.shape[1], V), dtype=torch.float64,
                             device=x.device)
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
                eps = 1e-12
                kl = (p_curr * (torch.log(p_curr + eps)
                                - torch.log(p_prev + eps))).sum(-1)
                kl_hist = torch.roll(kl_hist, -1, dims=-1)
                kl_hist[..., -1] = kl
                p_prev = p_curr.clone()
                if step >= self.history_length - 1:
                    stable = torch.all(kl_hist < self.kl_threshold, dim=-1)
                else:
                    stable = torch.zeros_like(conf, dtype=torch.bool)
                ready = stable & (conf > self.conf_threshold) & mask_idx
                xfer = torch.zeros_like(x0, dtype=torch.bool)
                for j in range(ready.shape[0]):
                    rdy = torch.where(ready[j])[0]
                    if len(rdy) > 0:
                        xfer[j, rdy] = True
                    else:
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
