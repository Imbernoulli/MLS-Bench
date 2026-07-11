# Model-Output Extraction for Code Generation

## Research Question

How much does output extraction affect executable pass@1 when the prompt, model,
and greedy decoder are fixed? The task evaluates one extraction policy over the
complete pinned MBPP sanitized test inventory.

## Editable Contract

Edit `code-generation-lab/solution/policy_extract.py` and implement
`extract(raw_text, entry_point) -> str`.

- `raw_text` is the unmodified model completion.
- `entry_point` is the required function name.
- The function must return Python source text.
- The policy receives no problem prompt, assertions, setup code, task identifier,
  or scoring results.

The evaluator deliberately does not prescribe an extraction algorithm. Each
returned program is compiled for diagnostics and executed for scoring.

## Evaluation Protocol

- Model: `Qwen/Qwen2.5-Coder-1.5B-Instruct`, revision
  `357b899b4714bf46d935fb9911e8139b5b9efc29`.
- Dataset: `google-research-datasets/mbpp`, revision
  `4bb6404fdc6cacfda99d4ac4205087b89d32030c`.
- Inventory: all 257 sanitized test problems in source order.
- Generation: one greedy completion per problem, seed 42, at most 512 new tokens.
- Scored metric: execution `pass_at_1_mbpp`; compilation rate is diagnostic.
