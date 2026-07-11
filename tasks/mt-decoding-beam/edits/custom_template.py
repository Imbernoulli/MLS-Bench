"""Machine-translation beam / repetition surface (agent-editable).

A FROZEN OPUS-MT MarianMT model translates each complete pinned OPUS-100 test
split with a FIXED length
policy (length_penalty 1.0, max_new_tokens 128); you control ONLY the BEAM SEARCH
and REPETITION control. The translations are scored on corpus sacreBLEU (higher
is better) against FIXED English references.

Implement:

    def build_beam_config() -> dict:
        return {"num_beams": 5, "no_repeat_ngram_size": 0}

The two knobs (transformers `model.generate`):
  num_beams            : beam width (1 == greedy; hard-capped at 12). Greedy
                         under-searches and leaves ~1-2 BLEU on the table vs a
                         tuned beam. Beam 4-5 is the standard strong MT decode;
                         very large beams trigger the "beam search curse"
                         (Stahlberg & Byrne 2019) — higher model-probability but
                         worse, shorter hypotheses — so bigger is NOT always
                         better.
  no_repeat_ngram_size : forbid repeating any n-gram of this size in the output.
                         0 disables it. MT rarely loops on clean short sentences,
                         so this is a minor lever here (3 is the standard value
                         if you want it); the main lever is the beam width.

Background:
  Greedy decoding (num_beams=1) is the weak baseline — it under-searches. The
  standard strong MT decode is beam search (num_beams 4-5). Koehn & Knowles 2017
  ("Six Challenges for NMT") show a clear BLEU rise from greedy to a small beam,
  then a plateau/decline at large beams. The DEFAULT here is the WEAK greedy
  config.

Notes:
  * Inference-only and deterministic. Each complete direction is evaluated on
    one GPU; all directions contribute to the score.
"""
from __future__ import annotations


# ================================================================
# EDITABLE REGION — return your beam / repetition decode config below
# ================================================================
def build_beam_config() -> dict:
    # Default (weak): greedy decoding, no repetition control.
    return {"num_beams": 1, "no_repeat_ngram_size": 0}
# ================================================================
# END EDITABLE REGION
# ================================================================
