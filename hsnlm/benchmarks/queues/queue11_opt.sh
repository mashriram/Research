#!/bin/bash
cd "$(dirname "$0")/../.." || exit 1
P=.venv/bin/python
$P benchmarks/gpu_decode_fair.py               >> benchmark_results/paper/decode_and_serving/gpu_decode_fair.log 2>&1
$P benchmarks/gpu_decode_fair.py --flash-alibi  > benchmark_results/paper/decode_and_serving/gpu_decode_fair_flashalibi.log 2>&1
$P benchmarks/gpu_overhead_profile.py --flash-alibi > benchmark_results/paper/training_overhead_profile/gpu_overhead_profile_flashalibi.log 2>&1
echo QUEUE11 DONE > benchmark_results/_queue_logs/queue11_opt.log
