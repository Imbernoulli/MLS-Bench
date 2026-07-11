# MDN Density Re-audit Handoff

> **SUPERSEDED / DO NOT USE THE 2026-07-09 HANDOFF FOR SHIP DECISIONS.**
> That analysis described obsolete tasks, overclaimed the provenance of the
> synthetic targets, trusted parsers that accepted forged or failed logs, and
> missed shell and baseline-edit syntax failures. The executable contract in
> `vendor/mdn-density/PROTOCOL.md` is authoritative.

## Current scope

The family contains exactly ten active sibling tasks:

- `mdn-activation`
- `mdn-component-balance`
- `mdn-covariance`
- `mdn-density-bench`
- `mdn-initialization`
- `mdn-learning-rate`
- `mdn-network-width`
- `mdn-num-components`
- `mdn-trunk-depth`
- `mdn-variance-floor`

`point-vs-mixture`, `variance-stability`, and `mixture-temperature` are not
active siblings. The latter two were removed because their research surfaces
were either mixed or redundant with the retained questions.

## Scientific protocol

The evaluation is a Bishop-inspired conditional-density benchmark, not an
exact reproduction of a canonical benchmark suite. `inverse_sine` is the
Bishop-family target; `two_branch`, `spiral`, and `rot_bimodal` are local
synthetic extensions. Each configured command uses seed 42, 20,000 training
examples, 20,000 held-out test examples, batch size 512, 4,000 optimizer
updates, and one CUDA GPU. Settings run serially.

Training archives are agent-visible. Test archives, the data-generating
implementation, the trusted harness, and model-building code are verifier-only.
The editable solution is a bounded five-line literal configuration; trusted
code parses it with `ast.literal_eval` and never imports or executes candidate
Python.

## Failure and scoring semantics

An agent failure that leaves the native solution untouched may be evaluated.
Training, evaluation, parser, process, timeout, OOM, cancellation, node, or
other verification failure produces no metric and therefore an exact zero
reward. The parser requires a unique, task-bound, surface-bound, target-bound,
seed-bound, budget-bound, data-hash-bound CUDA proof and a terminal completion.

Pending siblings have header-only leaderboards and ordinary score expressions;
there is no explicit score writeback, impossible floor, sentinel anchor, or
fallback implementation. Until fresh final-protocol anchors exist, valid rows
naturally score zero under the scorer's missing-calibration semantics.

The explicit-midpoint logistic score specs require the fail-closed scoring
implementation at commit `147ead243` (including its `dd0` parent). This family
commit is intentionally based directly on `670dcd12`; it must not be rendered
or shipped without merging that scoring dependency first. The older scorer
misinterprets `ref` plus `scale` as a shifted-floor sigmoid.

## Representative runtime evidence

Mangrove task `96377`, container `4927284`, dataset version `18734`, ran the
representative full workload. Setup took 49.541 seconds, verification took
36.854 seconds, and the whole trial took 87.068 seconds. The native held-out
NLLs were `-0.073271`, `1.578776`, and `-0.084000`. A measured stronger recipe
at the same H20 protocol produced `-1.169754`, `-1.023576`, and `-1.806954`.
Those anchors map to 0.1 and 0.5 respectively.

This runtime evidence may be reused only while the numerical workload, data
digests, seed, update count, batch size, and model path remain unchanged. It
does not replace a fresh rendered verifier check for the final commit.

## Remaining ship gate

Before publication, run the complete static/adversarial replay on a worker,
render the final task bundle, verify the image is digest-pinned and contains all
runtime dependencies and data, and confirm the shared nonzero-command-return
gate is present. Do not install packages or prepare data during verification.
