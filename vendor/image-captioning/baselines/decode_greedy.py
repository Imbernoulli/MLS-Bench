"""Greedy decoding baseline.

Take the argmax token at every step until EOS / max length. Short, and prone to
repetition and premature EOS, so it under-covers the reference n-grams -> lower
CIDEr/BLEU. The weak baseline for the caption-decoding task.
"""
import torch


def decode(fns, image_embeds, cfg):
    eos = fns["eos_id"]
    max_len = cfg.get("max_len", 20)
    M = image_embeds.shape[0]
    caps = []
    for i in range(M):
        emb = image_embeds[i:i + 1]
        ids = []
        cur = torch.empty(1, 0, dtype=torch.long)
        for _ in range(max_len):
            logits = fns["next_logits"](emb, cur if cur.numel() else None)
            nxt = int(logits[0].argmax().item())
            if nxt == eos:
                break
            ids.append(nxt)
            cur = torch.tensor([ids], dtype=torch.long)
        caps.append(fns["decode_ids"](ids))
    return caps
