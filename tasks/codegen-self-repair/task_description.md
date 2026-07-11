# Compute-Matched Error-Driven Code Repair

## Research Question
When an initial frozen-model program fails a PROVIDED assertion, does the
failure message improve a fixed-budget reprompt compared with a compute-matched
reprompt that sees the same task and candidate but not the error?

The editable policy controls only repair-prompt construction. The verifier owns
all generation calls: one greedy candidate with `max_new_tokens=512` per failed
round, for at most two rounds. Returning a prompt cannot increase the candidate
count or change decoding.

## Implementation Contract
Modify `code-generation-lab/solution/policy_repair.py`:

```python
def build_repair_prompt(problem, program, error, round_index):
    ...
```

Return non-empty text. `problem` is the policy-visible task record, `program` is
the current candidate, `error` is the PROVIDED-test failure string, and
`round_index` is `0` or `1`. The chat-formatted repair input may contain at most
1,536 tokens. Calling generation from the editable surface fails verification.

## Fixed Evaluation
- Model revision: `357b899b4714bf46d935fb9911e8139b5b9efc29`; seed 42.
- Data revision: `4bb6404fdc6cacfda99d4ac4205087b89d32030c`; all 257 MBPP
  sanitized test problems under the derived one-PROVIDED/rest-RESERVED protocol.
- Initial generation and each repair generation are greedy, one candidate, and
  capped at 512 new tokens.
- Metric: RESERVED-test `pass_at_1_mbpp`. `visible_solve_rate_mbpp` and
  `repair_help_rate_mbpp` are diagnostics recomputed from per-problem proof.
- One GPU; full-split runtime must be measured before release.

Private scoring assertions are removed before the editable policy is imported.
