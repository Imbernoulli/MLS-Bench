# Prompt Specification Design for Code Generation

## Research Question

How much can the wording of a problem specification change greedy code
generation when the model, decoder, output extractor, and execution protocol are
fixed? This task evaluates a prompt policy over the complete pinned MBPP
sanitized test inventory.

## Editable Contract

Edit `code-generation-lab/solution/policy_docstring.py` and implement
`build_prompt(problem) -> str`.

- `problem` is a dictionary with exactly two fields: `prompt` and `entry_point`.
- The function must return a non-empty user message.
- The verifier applies the model's frozen chat template and rejects any returned
  message longer than 1024 tokenizer tokens. It does not truncate the message.
- The policy receives no assertions, setup code, task identifier, or scoring
  results.

The evaluator deliberately does not prescribe a prompt implementation. Generated
text is processed by the same fixed extractor for every submission.

## Evaluation Protocol

- Model: `Qwen/Qwen2.5-Coder-1.5B-Instruct`, revision
  `357b899b4714bf46d935fb9911e8139b5b9efc29`.
- Dataset: `google-research-datasets/mbpp`, revision
  `4bb6404fdc6cacfda99d4ac4205087b89d32030c`.
- Inventory: all 257 sanitized test problems in source order.
- Generation: one greedy completion per problem, seed 42, at most 512 new tokens.
- Scored metric: execution `pass_at_1_mbpp`; compilation rate is diagnostic.
