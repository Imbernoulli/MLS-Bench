# Best-of-N Candidate Selection for Code Generation (pass@1)

## Research Question
Given a FIXED pool of N sampled candidate programs per problem, which one do you
submit? A naive "take the first sample" ignores the other candidates. Design the
candidate **selection / reranking** policy for a **frozen** small code LM so that
the fraction of problems whose SELECTED program passes ALL RESERVED unit tests
(pass@1) is maximized. The research variable is the pool-level evidence used to
choose an index, including syntax, program structure, agreement, length, and
outcomes on the PROVIDED example tests.

## Background
Reranking correct programs among many candidates is the crux of turning high
pass@k into high pass@1. n-best reranking uses a feature to score candidates:
sequence log-likelihood (length-normalized), Coder-Reviewer likelihood, trained
verifiers (LEVER), execution results, and agreement across samples. This task
measures which signals transfer from the candidate pool and PROVIDED assertion
to correctness on the RESERVED assertions; it does not assume a winning policy.

## Implementation Contract
Modify `code-generation-lab/solution/policy_select.py`:

```python
def select_candidate(candidates, problem, tok):
    ...
```

- You receive `candidates` (list of program strings), `problem` (with
  `visible_tests`, `entry_point`, `prompt`, `test_setup`) and `tok`. You may
  execute candidates against the PROVIDED tests via
  `common.passes_all(prog, problem['visible_tests'], problem.get('test_setup',''))`
  or `common.run_tests(...)`. Return the chosen candidate INDEX.
- You NEVER receive the reserved tests. The candidate pool (8 samples @ T=0.6),
  sampling, code extraction, and reserved-test scoring are all FIXED in the harness.

## Fixed Pipeline & Evaluation
- Model: `Qwen2.5-Coder-1.5B-Instruct` (frozen), fp16, seed 42.
- Setting `mbpp`: all 257 MBPP sanitized test problems under the derived
  one-PROVIDED/rest-RESERVED assertion protocol. Pool: 8 candidates per problem
  at temperature 0.6, top-p 0.95.
- Metric (higher is better): `pass_at_1_mbpp` = (# SELECTED programs passing ALL
  RESERVED tests) / 257. `oracle_pass_at_1_mbpp` (any pool candidate passes) is the
  ceiling, reported for diagnosis. Selecting a candidate that overfits the
  visible tests still cannot exceed the pool's true-correct fraction; a broken
  selector fails verification instead of being replaced by candidate 0.
- Deterministic; runs on one GPU; full-split runtime must be measured before release.

## Pinned Protocol
- Model revision: `357b899b4714bf46d935fb9911e8139b5b9efc29`; MBPP revision:
  `4bb6404fdc6cacfda99d4ac4205087b89d32030c`; source-order inventory: 257.
- Policy input contains the prompt, entry point, setup, and one PROVIDED assertion,
  never the RESERVED scoring assertions. Private files are removed before policy import.
