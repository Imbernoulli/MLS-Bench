#!/bin/bash
python custom_sr.py --benchmark koza3 --seed ${SEED:-42} --pop-size 500 --generations 50 --max-depth 6
