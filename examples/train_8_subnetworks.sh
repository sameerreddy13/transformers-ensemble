#!/bin/bash
if [ $# -gt 0 ]; then
	python3 ensemble_basic.py --num-models 8 --batch-size 32 --lr 5e-3 --save-dir examples/subnet_checkpoints --extract-subnetwork --num-epochs 20 --gpus $@
fi