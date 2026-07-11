# Run-Wide Few-Shot Priming for Code Generation

## Research Question

Can one fixed in-context demonstration prefix improve greedy code generation
across the complete MBPP sanitized test inventory? The model, base instruction,
decoder, output extractor, and execution protocol remain fixed.

## Editable Contract

Edit `code-generation-lab/solution/policy_fewshot.py` and implement
`fewshot() -> str`.

- The verifier calls `fewshot()` exactly once for the complete run.
- The function receives no arguments and therefore cannot inspect or adapt to the
  current benchmark problem.
- It returns one run-wide text prefix, which may be empty.
- The prefix is capped at 256 tokens under the pinned model tokenizer. The
  verifier rejects an oversized prefix rather than truncating it.
- A non-empty prefix must be generic and must not encode solutions or assertions
  from any benchmark problem.

For every problem, the harness appends the same fixed base instruction and the
current problem prompt after this one prefix. The evaluator does not prescribe
the contents of the prefix.

## Evaluation Protocol

- Model: `Qwen/Qwen2.5-Coder-1.5B-Instruct`, revision
  `357b899b4714bf46d935fb9911e8139b5b9efc29`.
- Dataset: `google-research-datasets/mbpp`, revision
  `4bb6404fdc6cacfda99d4ac4205087b89d32030c`.
- Inventory: all 257 sanitized test problems in source order.
- Generation: one greedy completion per problem, seed 42, at most 512 new tokens.
- Scored metric: execution `pass_at_1_mbpp`; compilation rate is diagnostic.
