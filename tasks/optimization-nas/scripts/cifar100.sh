#!/bin/bash
# The harness re-stages this run's validation table before every evaluation,
# so the runner treats it as ephemeral (deleted right after loading, before
# any editable code runs).
export MLSBENCH_EPHEMERAL_INPUTS=1
ENV=cifar100 NAS_EPOCHS=30 python custom_nas_search.py
