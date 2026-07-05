"""Self-contained SARI (Xu et al. 2016) — the standard reference-based text
simplification metric. Pure Python, no external deps.

SARI needs the SOURCE sentence, the SYSTEM output, AND multiple human REFERENCES
(unlike BLEU/ROUGE which ignore the source). It rewards correct ADD / KEEP /
DELETE n-gram edits relative to the source:

    SARI = (F1_add + F1_keep + P_del) / 3   averaged over n = 1..4

Faithfully ported from the HuggingFace `evaluate` SARI metric (which is itself
adapted from the tensor2tensor implementation, Xu et al. 2016). Returns 0..100.
"""
from __future__ import annotations

from collections import Counter
from typing import List


def _sari_ngram(sgrams, cgrams, rgramslist, numref):
    rgramsall = [rgram for rgrams in rgramslist for rgram in rgrams]
    rgramcounter = Counter(rgramsall)

    sgramcounter = Counter(sgrams)
    sgramcounter_rep = Counter()
    for sgram, scount in sgramcounter.items():
        sgramcounter_rep[sgram] = scount * numref

    cgramcounter = Counter(cgrams)
    cgramcounter_rep = Counter()
    for cgram, ccount in cgramcounter.items():
        cgramcounter_rep[cgram] = ccount * numref

    # KEEP
    keepgramcounter_rep = sgramcounter_rep & cgramcounter_rep
    keepgramcountergood_rep = keepgramcounter_rep & rgramcounter
    keepgramcounterall_rep = sgramcounter_rep & rgramcounter

    keeptmpscore1 = 0
    keeptmpscore2 = 0
    for keepgram in keepgramcountergood_rep:
        keeptmpscore1 += keepgramcountergood_rep[keepgram] / keepgramcounter_rep[keepgram]
        keeptmpscore2 += keepgramcountergood_rep[keepgram]
    keepscore_precision = 1.0
    keepscore_recall = 1.0
    if len(keepgramcounter_rep) > 0:
        keepscore_precision = keeptmpscore1 / len(keepgramcounter_rep)
    if len(keepgramcounterall_rep) > 0:
        keepscore_recall = keeptmpscore2 / sum(keepgramcounterall_rep.values())
    keepscore = 0.0
    if keepscore_precision > 0 or keepscore_recall > 0:
        keepscore = 2 * keepscore_precision * keepscore_recall / (
            keepscore_precision + keepscore_recall)

    # DELETE (precision only)
    delgramcounter_rep = sgramcounter_rep - cgramcounter_rep
    delgramcountergood_rep = delgramcounter_rep - rgramcounter
    delgramcounterall_rep = sgramcounter_rep - rgramcounter
    deltmpscore1 = 0
    deltmpscore2 = 0
    for delgram in delgramcountergood_rep:
        deltmpscore1 += delgramcountergood_rep[delgram] / delgramcounter_rep[delgram]
        deltmpscore2 += delgramcountergood_rep[delgram] / delgramcounterall_rep[delgram]
    delscore_precision = 1.0
    if len(delgramcounter_rep) > 0:
        delscore_precision = deltmpscore1 / len(delgramcounter_rep)

    # ADD
    addgramcounter = set(cgramcounter) - set(sgramcounter)
    addgramcountergood = set(addgramcounter) & set(rgramcounter)
    addgramcounterall = set(rgramcounter) - set(sgramcounter)
    addtmpscore = 0
    for addgram in addgramcountergood:
        addtmpscore += 1
    addscore_precision = 1.0
    addscore_recall = 1.0
    if len(addgramcounter) > 0:
        addscore_precision = addtmpscore / len(addgramcounter)
    if len(addgramcounterall) > 0:
        addscore_recall = addtmpscore / len(addgramcounterall)
    addscore = 0.0
    if addscore_precision > 0 or addscore_recall > 0:
        addscore = 2 * addscore_precision * addscore_recall / (
            addscore_precision + addscore_recall)

    return (keepscore, delscore_precision, addscore)


def sari_sentence(source: str, prediction: str, references: List[str]) -> float:
    numref = len(references)

    s1grams = source.lower().split(" ")
    c1grams = prediction.lower().split(" ")
    r1gramslist = [r.lower().split(" ") for r in references]

    def grams(tokens, n):
        return [" ".join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]

    keep, dele, add = [], [], []
    for n in range(1, 5):
        sg = grams(s1grams, n)
        cg = grams(c1grams, n)
        rgl = [grams(r, n) for r in r1gramslist]
        k, d, a = _sari_ngram(sg, cg, rgl, numref)
        keep.append(k)
        dele.append(d)
        add.append(a)

    avgkeepscore = sum(keep) / max(len(keep), 1)
    avgdelscore = sum(dele) / max(len(dele), 1)
    avgaddscore = sum(add) / max(len(add), 1)
    finalscore = (avgkeepscore + avgdelscore + avgaddscore) / 3
    return finalscore * 100.0


def corpus_sari(sources: List[str], predictions: List[str],
                references: List[List[str]]) -> float:
    assert len(sources) == len(predictions) == len(references)
    tot = 0.0
    for s, p, refs in zip(sources, predictions, references):
        tot += sari_sentence(s, p, refs)
    return tot / max(len(sources), 1)
