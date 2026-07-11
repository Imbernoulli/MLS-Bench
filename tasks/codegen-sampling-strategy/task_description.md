# Sampling Diversity for a Fixed Code-Candidate Pool

## Research Question
For a frozen code model and a fixed pool of eight completions per problem, which
temperature and nucleus cutoff produce candidates that a fixed PROVIDED-test
selector can turn into the highest RESERVED-test pass@1?

Pool size, prompt wording, token cap, extraction, and selection are held fixed.
This isolates diversity parameters from candidate-compute allocation and prompt
engineering; no parameter pair is assumed to win before measurement.

## Implementation Contract
Modify `code-generation-lab/solution/policy_sampling.py`:

```python
def sampling_parameters(problem):
    ...
```

Return exactly `(temperature, top_p)` as two finite real numbers. Temperature
must be in `(0, 2]` and top-p in `(0, 1]`. `problem` contains only `task_id`,
`entry_point`, and the natural-language prompt. Invalid types, non-finite values,
or values outside the envelope fail verification.

## Fixed Evaluation
- Model: `Qwen/Qwen2.5-Coder-1.5B-Instruct`, revision
  `357b899b4714bf46d935fb9911e8139b5b9efc29`, fp16, seed 42.
- Data: all 257 problems in the MBPP sanitized test split at revision
  `4bb6404fdc6cacfda99d4ac4205087b89d32030c`, under the derived protocol that
  exposes one assertion to the selector and reserves the remaining assertions.
- Generation: exactly eight sampled candidates per problem, fixed prompt and
  `max_new_tokens=512`.
- Selection: choose the first candidate passing the PROVIDED assertion; if no
  candidate passes, choose candidate zero.
- Metric: `pass_at_1_mbpp`, the fraction of chosen programs passing every
  RESERVED assertion. `visible_solve_rate_mbpp` is diagnostic only.
- One GPU; full-split runtime must be measured before release.

Private scoring assertions are removed before the editable policy is imported.
