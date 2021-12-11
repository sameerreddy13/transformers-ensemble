#!/bin/bash
if [ $# -gt 0 ]; then
	python3 --exp-dir examples/subnet_checkpoints --weighted-vote --device $1 --batch-size 64
fi