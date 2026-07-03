# Documented residuals (known, accepted, or out-of-scope)

The fixes close the in-scope leak: the model cannot **name the eval target** nor
**read/regenerate the held-out answer** from its editable process. A few residual
vectors remain by design. They are recorded here for honesty and for a future
maintainer's judgement — none lets the model trivially read the answer key it was
handed; each is either information-theoretically inherent to the task or requires a
deeper (data-level) change than a per-task scaffold edit.

## 1. Inherent reconstruct-from-observable (tolerated "reconstruct-by-shape")

For some tasks the held-out optimum is a deterministic function of the **observable
inputs the model legitimately receives**. Recovering it is doing the task, not reading
the answer:

| Task | Residual | Why inherent |
|---|---|---|
| optimization-variance-reduction | the ridge minimizer is recomputable from `(X_train, y_train)` (self-documented in dgp.py) | the model is given the training problem it must optimize |

These match the project's stated tolerance: *reconstruct-by-shape is a tolerated
"邪门技巧"*. The fix still removes the **explicit** `w_star`/`y_test` + the DGP +
the scorer from the scaffold, so an honest solver computes the metric the intended
way and a lazy "return the stored answer" path no longer exists.

**Update (2026-07-03).** optimization-diagonal-net and optimization-parity were
previously listed here, but a Codex re-review showed their "observable" actually
carried the held-out answer in a form the editable code could read off disk:
diagonal-net's blob shipped `y_test` (so `w_star` was directly least-squares-
solvable), and parity's blob shipped the full **noiseless** pool labels (so the
secret `S` was GF(2)-solvable — a strong model exploited exactly this on the
leaderboard). Both are now **closed**, not tolerated: the fixed driver loads the
arrays into memory and **deletes the input blobs before any editable hook runs**,
and parity's `make_dataset` now sees only the UNLABELED pool and returns row
indices. The editable code can no longer read the answer off disk; the only
residual is a **frame-walk** into the fixed driver's locals — the same tolerated
class as §6 (commits 6cc0c2f, 8e9bce9).

## 2. optimization-multi-objective — marshalled evaluator

The agent receives a **name-free, marshalled** code object for the objective (so it
cannot read the problem name). A determined agent could `marshal.loads` + `dis` it to
recognise the exact ZDT/DTLZ arithmetic and re-derive the analytic Pareto front, then
hand-set fitness values (the fixed DEAP loop re-evaluates only `not fitness.valid`).
Name-free marshalling mitigates but does not eliminate this. Closing it fully would
need an opaque/black-box evaluator service (Recipe B), which was out of scope for this
batch.

## 3. quant — `$close` reconstruction (data-level, out of scope)

The qlib label `Ref($close,-2)/Ref($close,-1)-1` is a deterministic function of the
on-disk `$close` series that the features are also built from. An adversarial
`predict()` can bypass the dataset object and recompute the label from the global
`qlib.data.D` provider against the bind-mounted price data. Closing this requires a
data-level change (restricting the on-disk calendar visible during inference), not a
per-task dataset guard. The `_label_guard` closes the "read the answer key out of the
object you were handed" vector for **both predict and fit** (see `per-task.md`); the
`$close` reconstruction is explicitly left as a documented residual.

## 4. ml-clustering-algorithm — instruction.md geometry hint

The agent-visible `instruction.md` (a DBSCAN baseline reference, outside `_scaffold`)
discloses data **geometry** ("convex blobs, non-convex shapes, high-dimensional
embeddings") and a generator-tuned `eps=0.22`. This reveals neither the env→token
map, the seed, nor the held-out assignments — the answer stays host-side and scoring
is host-side. It telegraphs the dataset *families*, which the task framing arguably
needs (the task is to cluster across varied geometries). Flagged for a future call on
whether to genericise the baseline-reference prose; it is **not** an answer leak.

## 5. Opaque token: salted vs unsalted

ml-symbolic-regression uses a secret host-only salt (`"mlsbench-sr-v2"`) so its token
is irreversible. The other opaque-token tasks (causal-treatment-effect, ml-anomaly,
ml-clustering, ml-ensemble-boosting, ml-missing-data) use an **unsalted namespaced**
hash `sha1("<task>::dataset-token::v1::"+label)[:12]`. The namespace template lives
only in host-only `dgp.py` (not agent-visible). A maximally adversarial agent that
both guessed the exact template **and** read its own `ENV` token could brute-force a
small candidate dataset list. Per the rubric ("an opaque sha1 token … is OK") these
are LEAK-CLOSED; the salted/unsalted asymmetry is noted for a future hardening pass
(adding a salt would require regenerating the token-named input files + re-render).

## 6. Harbor phase isolation (why a generic "read /tests" worry is moot)

In Harbor the agent's editable program runs in the **work phase**, when only
`environment/_scaffold` is on disk; `tests/` (with `holdout/<task>/dgp.py`, the answer,
the scorer) is mounted **only at verification time**, in a separate process, after the
agent is done. So an agent cannot `import dgp` / call `truth()` during its run — the
held-out material is not present in its filesystem. (The native harness scores in the
host process and never bind-mounts the holdout into the container either.)
