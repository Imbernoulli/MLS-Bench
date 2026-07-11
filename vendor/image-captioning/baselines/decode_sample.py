"""Pure-sampling decoder baseline (weak — the shipped default).

Sample each token from the full softmax (temperature 1.0, no truncation).
High-variance, frequently off-topic tokens -> low n-gram overlap -> low CIDEr.
The weak baseline for the caption-decoding-strategy task.
Measured: CIDEr 0.1883, BLEU-4 0.048 (seed 42).
"""
import torch


def decode(fns, image_embeds, cfg):
    eos = fns["eos_id"]
    max_len = cfg.get("max_len", 20)
    g = torch.Generator(device="cpu").manual_seed(0)
    M = image_embeds.shape[0]
    caps = []
    for i in range(M):
        emb = image_embeds[i:i + 1]
        ids = []
        cur = torch.empty(1, 0, dtype=torch.long)
        for _ in range(max_len):
            logits = fns["next_logits"](emb, cur if cur.numel() else None)
            probs = torch.softmax(logits[0].float().cpu(), dim=-1)
            nxt = int(torch.multinomial(probs, 1, generator=g).item())
            if nxt == eos:
                break
            ids.append(nxt)
            cur = torch.tensor([ids], dtype=torch.long)
        caps.append(fns["decode_ids"](ids))
    return caps
