# GPyTorch GP Re-audit Handoff

> **SUPERSEDED / DO NOT USE EARLIER GP HANDOFFS FOR SHIP DECISIONS.**
> Old sibling anchors used different solver or budget protocols, old parsers did
> not authenticate the executed workload, and the old deep-kernel budget claim
> did not match the code. This handoff records the current `openml_full_v2`
> contract and its unresolved publication gates.

## Current scope and scale

The family contains ten active sibling tasks: ARD lengthscale, deep-kernel
width, deep-kernel extractor, ExactGP learning rate, kernel design, kernel
smoothness, likelihood noise, mean function, sparse inducing points, and SVGP
learning rate. All three configured OpenML datasets participate in every task's
score and execute as distinct serial groups on one CUDA GPU.

ExactGP and deep-kernel settings use 200 full-batch optimizer updates. SVGP
settings use 20 epochs with batch size 1,024, producing exactly 20, 160, and 300
optimizer updates for Concrete, Kin8nm, and Elevators. The proof binds task,
surface, dataset, seed, CUDA device, split digest, train/test counts, budget
kind/value, batch size, and actual update count.

## Runtime evidence

Mangrove task `96379`, container `4927422`, dataset version `18736`, used one
H20 and the complete three-dataset workload. Setup took 73.348 seconds,
verification took 234.023 seconds, and the trial took 308.261 seconds. Harness
times were approximately 11.1, 60.0, and 155.3 seconds. Native NLLs were
`3.114257`, `-0.004929`, and `-3.019634`; the measured ARD-Matern recipe gave
`2.993942`, `-0.507105`, and `-3.305446`.

That artifact emitted `openml_full_v1` metrics. It proves the workload and
runtime scale, but it is not an online pass of the current `openml_full_v2`
completion and parser contract. A final rendered v2 replay remains required.

The repaired Spectral Mixture path received a separate one-GPU smoke in the
digest-pinned repo image on worker `dev-tszg9-34612-worker-0`. The device was an
NVIDIA H200. CUDA initialization and an 8-by-8 kernel forward were finite, all
kernel parameters were on CUDA, and the unsupported multidimensional
`ard=False` plan was rejected before training. This smoke checks the repaired
builder path only; it is not a replacement for a full task anchor.

## Failure and scoring semantics

An agent failure that leaves the native solution untouched may be evaluated.
Training, evaluation, parser, process, timeout, OOM, cancellation, node, or
other verification failure produces no metric and an exact zero reward. The
task parser independently rejects runner failure prefixes even if a plausible
terminal proof follows them.

The representative kernel-design score uses a genuine logistic midpoint and
scale: native RBF maps to 0.1 and ARD-Matern maps to 0.5. This requires the
fail-closed scoring chain `dd0c8df53 -> 147ead243`; the direct `670dcd12`
scorer incorrectly treats the same expression as a shifted-floor sigmoid and
maps the rows to 0 and 0.8. Do not render or publish the family commit without
merging the shared scoring chain.

The other nine tasks have header-only leaderboards and no final-protocol
anchors. Their ordinary score expressions contain no impossible floor, explicit
writeback, sentinel, or fallback, and therefore return zero while uncalibrated.
They are static-ready but not publishable research tasks until task-specific
weak and strong anchors are measured.

## Render dependency and remaining gate

The configs require adapter support for `agent_data_prune`,
`verifier_data_deps`, verifier-only package files, and automatic removal of
unrelated sibling solutions. That support is not present in the direct base or
the scoring commits and must be committed and integrated separately.

Before publication, a fresh render must prove that the agent tree contains no
GP data, trusted common/harness modules, sibling solutions, anchors, leaderboard,
or score spec; the verifier tree must contain the six trusted modules and three
checksum-bound NPZ files. The digest-pinned repo image must already contain all
dependencies and data. Verification must not install packages, download, or
prepare data.
