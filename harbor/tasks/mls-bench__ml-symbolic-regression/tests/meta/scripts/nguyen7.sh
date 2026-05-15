#!/bin/bash
python custom_sr.py --benchmark nguyen7 --seed ${SEED:-42} --pop-size 500 --generations 50 --max-depth 6
