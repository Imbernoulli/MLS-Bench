# Oracle-Leakage Fixes for MLS-Bench Harbor Tasks

> Scope of this document: the 26 tasks fixed on branch `fix/oracle-leakage`
> (public PR #54). It records **what the leak was**, **how each task was fixed**,
> **why the fix preserves honest results**, and **how to reproduce every check**
> so a future maintainer can re-verify or extend the work without re-deriving it.

---

## 1. The leak, in one paragraph

Under the Harbor harness an evaluated model edits a slice of a task's code, then a
**separate verifier process** scores its work. The leak is this: for many tasks the
held-out answer (or the means to reconstruct it) lived **inside the slice the model
can see and run**. Concretely, a model could reach a perfect/near-perfect score
**without doing the science** by one of: reading the ground-truth answer that was
handed to its editable function; re-running a data-generating process (DGP) with a
known seed to regenerate the eval instance and thus the answer; or simply learning
**which dataset/config** is scored (the data here is often *synthetic*, so naming the
dataset and handing over the synthesis script is equivalent to handing over the
answer). None of these require adversarial intent — the information was simply in
scope.

**The fix principle (one line):** *break the co-location.* The model's editable
piece only ever sees **inputs**; the dataset identity, the DGP, the held-out answer,
and the scorer all move **out of the model's process** into the verifier-only
`tests/` tree. Honest predictions are byte-for-byte unchanged, so every existing
result reproduces.

---

## 2. Threat model (Harbor-specific)

What we defend against, and what we deliberately do not:

- **DEFEND:** the evaluated model must not be able to (a) **learn which
  dataset/config** is scored, or (b) **read or reconstruct the held-out answer**,
  from within its editable process. This is the real risk on Harbor, where the model
  gets repeated trial-and-error attempts against the eval.
- **OUT OF SCOPE (tolerated):** exotic "邪门技巧" such as walking Python stack frames
  to read a caller's locals, or reconstructing a public dataset purely from its
  shape. A model will not naturally do these, and fixing them would mean risky eval
  surgery. The native (non-Harbor) harness is also low-stakes (no trial-and-error
  loop) and is fixed only as a by-product.
- **HARD CONSTRAINT:** never break the eval. Every fix keeps the eval runnable and
  the numbers identical to the pre-fix baseline (exact for deterministic tasks,
  within seed noise otherwise). The goal is to **close the leak, not to make tasks
  unrunnable.**

---

## 3. Scope & isolation guarantee

This branch touches **exactly 26 tasks and nothing else.** Every changed file lives
under one of three per-task prefixes:

```
tasks/<task>/…            (native task source)
holdout/<task>/…          (host-only DGP/answer; never bind-mounted to the model)
harbor/tasks/mls-bench__<task>/…   (rendered Harbor bundle)
```

Verify it yourself from the branch root:

```bash
# Anything outside the three per-task prefixes? (expect: none)
git diff --name-only origin/main...HEAD | grep -vE '^(tasks/|holdout/|harbor/)'
# Any change under src/ ? (expect: none)
git diff --name-only origin/main...HEAD | grep -E '(^|/)src/'
```

Both return empty. **`src/mlsbench/` is untouched**, and no file shared across tasks
is modified, so the change set **cannot affect any task other than the 26**. This is
the rigorous basis for "the other tasks are unaffected": not a re-run of 140 tasks,
but a proof that the blast radius is the 26 by construction.

The 26 tasks:

| # | task | leak class | fix pattern |
|---|------|-----------|-------------|
| 1 | causal-treatment-effect | regenerable DGP (synthetic) | A: out-of-process + opaque token |
| 2 | causal-observational-linear-gaussian | regenerable DGP | A: out-of-process |
| 3 | causal-observational-linear-non-gaussian | regenerable DGP | A: out-of-process |
| 4 | causal-observational-nonlinear | regenerable DGP | A: out-of-process |
| 5 | causal-discovery-discrete | regenerable DGP | A: out-of-process |
| 6 | ml-anomaly-detection | dataset identity (synthetic/public) | A: out-of-process + opaque token |
| 7 | ml-clustering-algorithm | dataset identity | A: out-of-process + opaque token |
| 8 | ml-ensemble-boosting | dataset identity | A: out-of-process + opaque token |
| 9 | ml-missing-data-imputation | dataset identity | A: out-of-process |
| 10 | ml-symbolic-regression | regenerable DGP | A: out-of-process |
| 11 | optimization-diagonal-net | regenerable problem | A: out-of-process (pre-gen inputs) |
| 12 | optimization-multi-objective | regenerable problem | A: out-of-process |
| 13 | optimization-variance-reduction | regenerable problem | A: out-of-process (artifact) |
| 14 | optimization-parity | regenerable problem | A: out-of-process + parser tag-normalize |
| 15 | quant-stock-prediction | test label handed over | D: label-guard |
| 16 | quant-graph-stock | test label handed over | D: label-guard |
| 17 | quant-concept-drift | test label handed over | D: label-guard |
| 18 | ai4sci-pla-binding-affinity | answer as fn argument | B: mask-arg |
| 19 | ai4sci-mol-property-prediction | answer as fn argument | B: mask-arg |
| 20 | graph-signal-propagation | answer as fn argument | B: mask-arg |
| 21 | ai4sci-climate-emulation | answer as `__main__` global | C: wrap-main |
| 22 | ai4sci-weather-forecast-aggregation | answer as `__main__` global | C: wrap-main |
| 23 | graph-link-prediction | answer as `__main__` global | C: wrap-main |
| 24 | graph-node-classification | answer as `__main__` global | C: wrap-main |
| 25 | ml-calibration | answer as `__main__` global | C: wrap-main |
| 26 | ml-selective-deferral | answer as `__main__` global | C: wrap-main |

---

## 4. The fix patterns

Each pattern is modeled on a task that already did it right, and is chosen to be the
*least* invasive change that closes the leak without shifting line numbers (so the
edit-range guard and the model's editable window stay valid).

### Pattern A — out-of-process (predict-then-score)
*For: regenerable DGP / synthetic data / dataset-identity leaks (tasks 1–14).*

- Move the DGP + ground-truth answer + metric into `holdout/<task>/dgp.py`, a
  **host-only** module that is **never** bind-mounted into the model's container.
- The model's editable template is restructured to **read pre-generated inputs** and
  **emit a prediction artifact** (stdout line or file). It no longer contains the
  DGP, the answer, or the scorer.
- Inputs are pre-generated **deterministically** at workspace-setup time
  (`mid_edit.py`, host-side, ~0 eval-time cost): same seed → byte-identical inputs →
  honest results preserved.
- A fixed host-side scorer (native: `parser.py`; Harbor: `tests/meta/` + the
  verifier) reads the predictions and the held-out answer and computes the **same
  metric** as before.
- **Opaque token add-on** (tasks 1, 6, 7, 8): where a dataset *name* would otherwise
  reveal the eval target, the run scripts pass a host-only **salted sha1 token**
  (`sha1("<task>::dataset-token::v1::<label>")[:12]`) as `--dataset`/`ENV`. Listing
  the workspace reveals only opaque tokens; the host parser maps the test-command
  label back to the real dataset.

### Pattern B — mask-arg
*For: the held-out answer arrives as a function ARGUMENT (tasks 18, 19, 20).*

The editable `forward`/`predict` was handed the answer (e.g. `batch.labels`,
`batch.y`, `targets`). At **eval** time the fixed wrapper **zeroes the answer values
(shape preserved)** before calling the editable code, and keeps the true answer only
in the fixed eval scope for the metric. Honest predictions are unchanged (an honest
model never reads the labels it is asked to predict), so **no re-run is needed**. The
masking lives **outside** the editable range so the model cannot undo it.

### Pattern C — wrap-main
*For: the answer is reachable as a MODULE GLOBAL under `if __name__=='__main__':`
(tasks 21–26).*

The eval block ran at module top level, so `data`, `targets`, etc. became module
globals that the editable function could read via `globals()`. The fix moves the
block into `def _main(): …` plus `if __name__=='__main__': _main()`. The body is
already indented, so **line numbers do not shift**. Now those names are locals of
`_main`, invisible to the editable function. Honest code is unaffected (it never
legitimately reads a `__main__` global), so **no re-run is needed**.

### Pattern D — quant label-guard
*For: the qlib TEST-split label is handed to the editable model (tasks 15, 16, 17).*

The editable model code was given the test-split label column. The fix withholds the
test label from the editable scope; only train features/labels are exposed, and the
test target is revealed only to the verifier for scoring.

---

## 5. Verification methodology

Two independent properties are checked **per task**:

1. **Runnability / results preserved** — build the task's Harbor image, apply the
   reference baseline (`solution/solve.sh`), run the verifier (`tests/test.sh`), and
   confirm the produced metric matches the committed **anchor** (the reference
   baseline row in `tests/meta/leaderboard.csv`). Deterministic tasks must match
   exactly; stochastic tasks within seed noise. See §6 for the exact commands.
2. **Leak closed** — static + empirical:
   - *Static:* the agent-visible `environment/_scaffold/` tree contains **no**
     dataset identity, **no** held-out answer, **no** regenerable DGP, **no**
     self-scorer; those live only under `tests/`.
   - *Empirical:* a "cheat" solution that tries to read/reconstruct the answer from
     within the scaffold scores ~chance (or errors), **not** perfect.

**Do not trust a green checkmark from a sub-agent or a script — read the actual eval
log and the actual reward.** Every number in §8 is from a real verifier run whose log
was inspected by hand.

---

## 6. Reproduction recipe (run a Harbor verifier locally)

For any task `<t>` (bundle `B=harbor/tasks/mls-bench__<t>`):

```bash
B=harbor/tasks/mls-bench__<t>
# 1. Build the per-task image (FROM <base> + COPY _scaffold/ → /workspace)
docker build -q -t mls-verify-<t> "$B/environment/"
LOGS=$(mktemp -d)
# 2. Apply the reference baseline, then run the verifier.
#    --ipc=host: PyTorch DataLoader workers need >64MB /dev/shm (else a bus error).
#    --gpus '"device=N"': give GPU tasks a *dedicated* GPU — score_task places every
#    same-group benchmark on the first visible GPU (cuda:0), so multiple group=1
#    benchmarks sharing one container OOM each other; one GPU per container avoids it.
docker run --rm --gpus '"device=0"' --ipc=host \
  -v "$PWD/$B/tests":/tests:ro -v "$PWD/$B/solution":/solution:ro -v "$LOGS":/logs \
  mls-verify-<t> \
  bash -c "mkdir -p /logs/verifier && bash /solution/solve.sh; bash /tests/test.sh"
# 3. Read the result + the actual eval log (don't trust the number blindly).
cat "$LOGS/verifier/reward.txt"
cat "$LOGS/verifier/score_error.txt" 2>/dev/null
ls  "$LOGS/verifier/"*.log
```

The anchor to compare against:

```bash
grep -i baseline "$B/tests/meta/leaderboard.csv" | head
```

### Known **verification-harness** artifacts (NOT task bugs)

These cause a spurious `reward=0` in a naive local run; they are properties of the
local reproduction, not of the fixes:

- **CPU oversubscription** — running several CPU-bound tasks in parallel on a shared
  box starves each; CPU-parallel algorithms (missForest, GES) then time out and a
  gmean over configs zeroes the reward. *Run CPU-heavy tasks one at a time with full
  cores.*
- **`/dev/shm` too small** — Docker's default 64 MB shared memory is too small for
  PyTorch DataLoader workers → bus error. *Always pass `--ipc=host`.*
- **GPU packing** — `score_task` puts every same-`group` benchmark on the first
  visible GPU. Three `group=1` benchmarks in one container OOM each other. *Give each
  benchmark its own dedicated GPU/container when reproducing.*
- **Short timeout** — some evals are legitimately long (e.g. optimization-diagonal-net
  runs 19 instances of long gradient descent). A short docker timeout cuts them off →
  `reward=NONE`. *Use a generous timeout.*

---

## 7. Per-task fix notes

See `per-task.md` for the file-by-file change of every task (editable template,
`mid_edit.py`, `holdout/dgp.py`, `parser.py`, `config.json`, scripts) plus the exact
commit that introduced it.

---

## 8. Results (anchor vs post-fix)

See `results.md` for the live table (task / anchor / post-fix metric / match?). It is
filled from real verifier runs as each completes; the methodology is §5–§6.
