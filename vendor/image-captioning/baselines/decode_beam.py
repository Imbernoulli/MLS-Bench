"""Beam-search decoding baseline (with length penalty + no-repeat-ngram).

Beam width 5, GNMT length penalty (alpha=1.0), a no-repeat-2gram block and a
minimum length. Explores multiple hypotheses and avoids the greedy decoder's
short/repetitive captions, covering more reference n-grams -> higher CIDEr/BLEU.
The strong baseline for the caption-decoding task. Uses the harness's own beam
primitive so the search is exactly the reference implementation.
"""


def decode(fns, image_embeds, cfg):
    return fns["beam_decode"](
        image_embeds, beam=5, max_len=cfg.get("max_len", 20),
        length_penalty=1.0, no_repeat_ngram=2, min_len=4,
    )
