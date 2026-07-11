"""Machine-translation sampling-vs-beam surface (agent-editable).

For each complete pinned OPUS-100 source-to-English test split, a FROZEN OPUS-MT model produces a translation; you
choose the decode MODE. Scored on corpus sacreBLEU (higher is better).

Implement:

    def build_mode() -> str:
        return "beam"

Modes:
  "sample_t1" : ancestral sampling, temperature 1.0  — weakest (high variance).
  "topp"      : nucleus sampling, top_p 0.9, temp 0.7 — better than pure sampling.
  "greedy"    : greedy argmax decode                  — deterministic, decent.
  "beam"      : beam-5 MAP search                     — strong (best for MT).

Background:
  For a peaked, well-trained MT model, MAP-style search (beam) beats stochastic
  sampling on BLEU. Sampling helps OPEN-ENDED generation (Holtzman et al. 2020),
  not MT: high-temperature sampling injects noise and drops BLEU. Order:
  sample_t1 < topp < greedy <= beam.

Notes:
  * Inference-only. Aggregated over three directions (de/fr/ru -> en). Minute-scale.
"""
from __future__ import annotations


# ================================================================
# EDITABLE REGION — return your decode mode below
# ================================================================
def build_mode() -> str:
    # Default (weak): pure ancestral sampling at temperature 1.0.
    return "sample_t1"
# ================================================================
# END EDITABLE REGION
# ================================================================
