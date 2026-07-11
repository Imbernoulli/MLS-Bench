# Global Candidate-Budget Allocation for Code Generation

## Research Question
At a fixed total of 1,028 sampled candidates across 257 problems, can a global
hardness policy improve RESERVED-test pass@1 over uniform compute allocation?
The total candidate count, sampling parameters, prompt, extraction, and
PROVIDED-test selector are fixed.

## Implementation Contract
Modify `code-generation-lab/solution/policy_budget.py`:

```python
def allocation_weights(problems):
    ...
```

The function is called exactly once. Return a list or tuple containing one
finite non-negative real weight for each of the 257 policy-visible problems;
at least one weight must be positive. Each record contains only `task_id`,
`entry_point`, and the natural-language prompt. The verifier deterministically
maps the weights to integer allocations in `[1, 8]` using capped
largest-remainder allocation and verifies that their sum is exactly 1,028.

Calling model generation from the allocation policy fails verification.

## Fixed Evaluation
- Frozen model revision `357b899b4714bf46d935fb9911e8139b5b9efc29`, fp16,
  seed 42; fixed temperature 0.6, top-p 0.95, and 512-token cap.
- All 257 MBPP sanitized test problems at revision
  `4bb6404fdc6cacfda99d4ac4205087b89d32030c`, under the derived
  one-PROVIDED/rest-RESERVED assertion protocol.
- Candidate zero is chosen when no candidate passes the PROVIDED assertion;
  otherwise the first passing candidate is chosen.
- Metric: `pass_at_1_mbpp`; exact average candidate count is reported as
  `avg_samples_mbpp` and recomputed from per-problem proof.
- One GPU; full-split runtime must be measured before release.

Private scoring assertions are removed before the editable policy is imported.
