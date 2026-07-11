# qa-question-inclusion

## Research objective

Measure the combined effect of question removal by comparing the real question with the fixed one-token placeholder `what`. This intervention removes question semantics but also reallocates input tokens to the context, so it is not presented as a pure conditioning-only effect.

## Editable contract

Edit only `extractive-qa/solution/question_inclusion.py`. The module must contain
exactly one zero-argument function named `build_question_mode` whose body is one
literal `return`. The accepted value is the string "real" or "drop". Imports, decorators,
annotations, computations, additional statements, and additional functions are
invalid; the verifier parses this surface with a restricted AST and never executes
agent-authored Python.

## Evaluation protocol

The frozen `deepset/roberta-base-squad2` checkpoint and all tokenizer files are
pinned at revision `adc3b06f79f797d1c575d5479d6f5efe54a9e3b4` and verified against a complete six-file
SHA-256 manifest before inference. The full checkpoint contains 124,056,578 model
parameters. Its documented upstream recipe
trained RoBERTa-base on SQuAD 2.0 for two epochs with batch size 96. Questions are capped
at 64 tokens, long contexts use complete overflow-window coverage, and each example
must produce exactly one prediction. The verifier evaluates four complete answerable validation domains from the official MRQA unified validation data. Every source example is loaded and must produce exactly one prediction; task-specific window-use interventions are disclosed above. Every configured command contributes to the task score.

The primary metric is official SQuAD token-overlap F1 on a 0-100 scale. Exact
match is reported as a secondary metric. Each per-command official F1 maps directly from 0 to score 0 and from 100 to score 1; the task score is their geometric mean. This fixed, baseline-free mapping does not substitute a representative result for the current run. Dataset loading, model loading, CUDA
execution, feature construction, inference, prediction completeness, metric
calculation, terminal completion proof emission, or output parsing failure yields no metric and
therefore a score of exactly zero; there is no fallback metric or default score.
