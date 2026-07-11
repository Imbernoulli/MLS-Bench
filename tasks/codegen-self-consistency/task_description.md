# Canonicalization for Code Self-Consistency

## Research Question
When several independently sampled programs pass the PROVIDED assertion, can a
canonicalization key identify non-trivial agreement that improves RESERVED-test
pass@1 over choosing the first survivor?

## Implementation Contract
Modify `code-generation-lab/solution/policy_consensus.py`:

```python
def canonical(program):
    ...
```

Return a hashable key for one candidate source string. Generation is disabled
while the editable canonicalizer runs. The harness groups the fixed pool's
PROVIDED-test survivors by key, chooses a representative of the largest group,
and resolves ties by earliest candidate index.

## Fixed Evaluation
- Frozen model revision `357b899b4714bf46d935fb9911e8139b5b9efc29`, fp16,
  seed 42; exactly eight candidates per problem at temperature 0.6, top-p 0.95,
  and a 512-token cap.
- All 257 MBPP sanitized test problems at revision
  `4bb6404fdc6cacfda99d4ac4205087b89d32030c`, under the derived
  one-PROVIDED/rest-RESERVED assertion protocol.
- Metric: `pass_at_1_mbpp`. Per-problem proof also recomputes the pool oracle,
  survivor and cluster counts, winning-cluster size, non-trivial agreement rate,
  and how often clustering changes the first-survivor choice.
- One GPU; full-split runtime must be measured before release.

Private scoring assertions are removed before the editable policy is imported.
