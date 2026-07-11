# Machine Translation: Diverse Beam Search

## Research Question
With beam width fixed at eight, how should beams be partitioned into groups and how
much inter-group diversity pressure should be applied for single-best translation?
Diversity can expose different hypotheses, but excessive pressure can displace
high-probability translations.

## Implementation Contract
Modify `machine-translation/solution/divbeam.py` so
`build_divbeam_config() -> dict` returns exactly
`num_beam_groups` and `diversity_penalty`. The group count must be a positive
divisor of eight, and the finite penalty must lie in `[0, 5]`. A one-group policy
must use a zero penalty; a multi-group policy must use a strictly positive penalty.

## Fixed Evaluation
- Three pinned frozen OPUS-MT MarianMT checkpoints (~75M parameters each).
- The complete official 2000-pair OPUS-100 test split is used for each of three
  source languages, without filtering, sampling, shuffling, or head slicing.
- Corpus sacreBLEU is primary and chrF is secondary.
- Every direction executes serially on one GPU and contributes to the task score.
