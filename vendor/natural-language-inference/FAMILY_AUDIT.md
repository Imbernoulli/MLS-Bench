# Natural-Language-Inference Family Audit

Audit date: 2026-07-11

## Inventory

- Repository family: `natural-language-inference`
- Siblings: 10
- Training corpus: all 549,367 labeled SNLI training rows in source order
- Training budget: 3 epochs, batch 32, exactly 51,504 optimizer steps
- Scored evaluations: complete SNLI test (9,824), MultiNLI matched dev
  (9,815), and MultiNLI mismatched dev (9,832)
- Seed: 42
- GPU allocation: exactly one visible GPU per task command
- Runtime image: `sha256:3413891ea22deecf213026a9c34403d65133702286042175057dcb88f329e7e6`

All three evaluation settings participate in the geometric-mean score. There
is no public/hidden setting distinction. Missing, malformed, non-finite,
duplicated, trailing, failed, nonzero-return-code, or incomplete verification
produces no metrics and therefore an exact zero score.

## Representative Evidence

Mangrove task `96642` / container `4950830` used one NVIDIA H20. The verifier
ran for 994.677 seconds; environment setup was 51.276 seconds, agent time was
0.284 seconds, and the complete trial was 1,047.003 seconds (1,048.760 seconds
platform time). The measured score was 0.36685953030730406. This runtime is
retained as the family-scale representative; the identity and fail-closed
changes below do not alter the data, model, batch size, epochs, optimizer-step
inventory, or evaluation inventory.

## Re-Audit Corrections

The earlier accepted state was not fully reproducible or sibling-bound:

- The package config described an immutable repository image but did not set
  `mangrove_base_image`. It now pins the measured runtime image by digest.
- Clean preparation now stages and authenticates the exact config, safetensors,
  fast-tokenizer, tokenizer-config, and vocabulary files from the pinned model
  revision. All five assets and the four canonical corpus outputs are required
  before the dependency is ready, baked into future repository images, and
  re-authenticated by the verifier.
- Ten byte-identical parsers did not bind a task or research surface and
  accepted output after the command-completion record. Each sibling now embeds
  its literal task, surface, and legal policy domain, requires one terminal
  `NLI_COMMAND_DONE rc=0`, and rejects infrastructure failure markers.
- Runtime scripts overwrote `CUDA_VISIBLE_DEVICES`. They now inherit the worker
  assignment, while the verifier requires exactly one visible CUDA GPU.
- The training parser previously ignored malformed records in the reserved
  `NLI_TRAIN*` namespace when they did not contain the expected ASCII space.
  Every sibling now rejects those lookalikes, duplicate train/completion records,
  and any weight decay other than the verifier-owned policy value.
- The truncation parser now binds `lenN` exactly to the protocol and optimizer
  proof's `max_length=N`. The regularization harness separately proves dropout
  and weight decay, and its parser binds `none`, `standard`, and `heavy` to the
  exact verifier-owned values.
- Protocol and model proofs now bind the actual tokenizer cap (128 joint tokens
  for cross encoders or 64 tokens per sentence for siamese encoders), architecture,
  encoder/head/total parameter counts, and all pinned runtime-asset hashes.
- The old class-weighting comparison used inverse-frequency weights on an
  almost perfectly balanced corpus (less than 0.4 percent total weight spread).
  It now exposes a bounded, mean-one three-class cost vector. This preserves all
  rows and optimizer steps while making the research axis materially distinct.

The agent-failure rule is unchanged: an agent that fails before editing may
leave the native solution for evaluation. Once verification starts, no
candidate exception, invalid surface, missing proof, or failed command may be
replaced with a fallback implementation or positive score.

## Known Design Boundaries

- `frozen` is the conventional frozen-feature-extractor arm: its encoder remains
  in evaluation mode, while `finetune` trains the encoder with its configured
  dropout. This is disclosed as a bundled update policy, not a pure gradient-only
  ablation.
- The augmentation `negation` arm is a deterministic lexical diagnostic and can
  introduce label noise; it is not claimed to guarantee semantic contradiction.
- Pair encoding compares a complete joint cross-encoder readout with a complete
  mean-pooled InferSent-style bi-encoder readout. It is not presented as isolating
  only one internal architecture primitive.
