#!/bin/bash
# Opaque task token (host-side salted hash of the benchmark); reveals nothing
# about which target function is in use.
SR_TASK=3ea1edc2332c python custom_sr.py --seed ${SEED:-42} --pop-size 500 --generations 50 --max-depth 6
