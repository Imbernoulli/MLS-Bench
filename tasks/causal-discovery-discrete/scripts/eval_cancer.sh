#!/bin/bash
# Evaluate on a held-out discrete Bayesian-network case.
# The integer-encoded samples are pre-generated into the workspace at setup time;
# the network identity and the ground-truth DAG are held out off-machine. The
# driver selects the pre-generated input by the OPAQUE case token below (which
# reveals nothing about the network) and emits a CAUSAL_PRED line that the
# host-side scorer grades against the regenerated truth.

ENV=828d54650232 python -u bench/run_eval.py --seed "${SEED:-42}"
