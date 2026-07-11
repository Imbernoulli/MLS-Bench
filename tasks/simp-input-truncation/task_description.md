# Simplification Input Budget

## Research Question
Evaluate the encoder-side input token budget for a frozen text-simplification pipeline. The same implementation
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
Edit `text-simplification/solution/truncation.py` and implement:

```python
def build_max_input_tokens() -> int:
    ...
```

Return an integer within the documented input bound. Invalid values, missing keys, non-finite metrics, generation errors,
or incomplete outputs fail verification. The harness does not clamp or repair an
editable configuration and does not substitute another policy.

## Evaluation
The verifier reports corpus SARI from complete predictions and verifier-only references. Every configured evaluation must complete with finite metrics.
