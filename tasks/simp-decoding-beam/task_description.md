# Simplification Beam Configuration

## Research Question
Evaluate beam width and repetition controls for a frozen text-simplification pipeline. The same implementation
is evaluated on complete official test partitions; human references are
available only to the verifier.

## Data and Model Visibility
The action workspace contains the complete source-only ASSET (359), TurkCorpus
(359), and WikiAuto (720) partitions so implementations can inspect the sentences
they must simplify. Human reference simplifications are never present in the
action workspace; they are mounted only for verifier scoring. Revision-pinned
frozen checkpoint files are staged offline and are readable because they are the
inference targets, not scoring labels. The verifier checks their file and
architecture manifests before accepting a completion proof.
The shared runtime, SARI implementation, and this task's active harness are also
agent-readable so the execution path can be inspected. Other sibling harnesses,
human references, parser, calibration evidence, and scores are not action inputs.

## Implementation Contract
Edit `text-simplification/solution/beam.py` and implement:

```python
def build_beam_config() -> dict:
    ...
```

Return exactly `num_beams`, `no_repeat_ngram_size`, and `repetition_penalty` with valid numeric types. Invalid values, missing keys, non-finite metrics, generation errors,
or incomplete outputs fail verification. The harness does not clamp or repair an
editable configuration and does not substitute another policy.

## Evaluation
The verifier reports corpus SARI from complete predictions and verifier-only references. Every configured evaluation must complete with finite metrics.
