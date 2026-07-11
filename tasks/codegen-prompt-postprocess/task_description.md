# Prompt and Postprocess Interaction for Code Generation

## Research Question

How do prompt wording and output postprocessing interact when a frozen code model
is evaluated under multiple deterministic output-format conditions? The same
prompt policy produces one greedy completion per MBPP problem, and the same
postprocessor must turn three views of that completion into executable source.

## Editable Contract

Edit `code-generation-lab/solution/policy_postprocess.py` and implement both
`build_prompt(problem) -> str` and
`postprocess(raw_text, entry_point) -> str`.

- `build_prompt` receives a dictionary with exactly `prompt` and `entry_point`.
- Its return value must be non-empty and no longer than 1024 tokens after the
  pinned chat template is applied. Oversized values are rejected, not truncated.
- `postprocess` receives one completion view and the required function name. It
  receives no condition label, problem prompt, assertions, setup code, task
  identifier, or scoring results.
- Both functions are loaded only after the verifier's private files are removed.

The evaluator deliberately does not prescribe either implementation.

## Scored Conditions

One model completion is generated for each problem, then the same postprocessor
is called on all three deterministic views:

- `direct`: the model completion without modification.
- `fenced_wrapper`: the completion placed inside a deterministic Markdown fence
  with fixed surrounding prose.
- `trailing_text`: the completion followed by deterministic non-code text.

Each condition has its own execution pass@1 and compilation diagnostic. All three
pass@1 values participate equally in the task score; no condition is excluded.
Per-item proof records contain a pass bit and parse bit for every condition, so
the parser recomputes every terminal aggregate.

## Evaluation Protocol

- Model: `Qwen/Qwen2.5-Coder-1.5B-Instruct`, revision
  `357b899b4714bf46d935fb9911e8139b5b9efc29`.
- Dataset: `google-research-datasets/mbpp`, revision
  `4bb6404fdc6cacfda99d4ac4205087b89d32030c`.
- Inventory: all 257 sanitized test problems in source order.
- Generation: one greedy completion per problem, seed 42, at most 512 new tokens.
- Scored settings: `direct`, `fenced_wrapper`, and `trailing_text`.
