#!/bin/bash
cd "$(dirname "$0")/../.." || exit 1
until [ -f benchmark_results/_queue_logs/queue12_phaseB.log ]; do sleep 60; done
.venv/bin/python benchmarks/gpu_lra_subset.py > benchmark_results/paper/long_range_arena/gpu_lra_subset.log 2>&1
echo QUEUE13 DONE > benchmark_results/_queue_logs/queue13_lra.log
