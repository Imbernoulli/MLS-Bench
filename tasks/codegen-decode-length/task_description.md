# Fixed-Budget Generation-Length Allocation

## Research Question
At a fixed total generation cap, how should token capacity be distributed across
the full problem inventory? The experiment tests whether a global estimate of
solution-length need improves RESERVED-test pass@1 over uniform caps without
increasing the total candidate or token-cap budget.

## Implementation Contract
Modify `code-generation-lab/solution/policy_decode.py`:

```python
def token_cap_weights(problems):
    ...
```

The function is called once and receives all 257 records, each containing only
`task_id`, `entry_point`, and the natural-language prompt. Return one finite
non-negative real weight per record, with at least one positive weight. The
verifier deterministically converts the weights to integer token caps in
`[64, 640]`; the caps sum exactly to `257 * 256`. Model generation is not
available while this policy runs.

## Fixed Evaluation
- Frozen model revision `357b899b4714bf46d935fb9911e8139b5b9efc29`, fp16,
  seed 42; four candidates per problem at temperature 0.6 and top-p 0.95.
- All 257 MBPP sanitized test problems at revision
  `4bb6404fdc6cacfda99d4ac4205087b89d32030c`, under the derived
  one-PROVIDED/rest-RESERVED assertion protocol.
- Fixed PROVIDED-test first-survivor selection and fixed code extraction.
- Metric: `pass_at_1_mbpp`. `parse_rate_mbpp` and `avg_token_cap_mbpp` are
  proof-recomputed diagnostics.
- One GPU; full-split runtime must be measured before release.

Private scoring assertions are removed before the editable policy is imported.
