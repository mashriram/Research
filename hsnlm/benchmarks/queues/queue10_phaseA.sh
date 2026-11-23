#!/bin/bash
# Phase A reviewer-response runs, sequential on the single GPU. Each script resumes from its results JSON.
cd "$(dirname "$0")/../.." || exit 1
P=.venv/bin/python
$P benchmarks/gpu_overhead_profile.py  > benchmark_results/paper/training_overhead_profile/gpu_overhead_profile.log 2>&1
$P benchmarks/gpu_extrap_16k.py        > benchmark_results/paper/length_extrapolation_16k/gpu_extrap_16k.log 2>&1
$P benchmarks/gpu_wikitext103_3seed.py > benchmark_results/paper/wikitext103_3seed/gpu_wikitext103_3seed.log 2>&1
$P benchmarks/gpu_scaling_law.py       > benchmark_results/paper/scaling_2m_to_38m/gpu_scaling_law.log 2>&1
echo QUEUE DONE
