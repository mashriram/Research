#!/bin/bash
# Phase B queue: waits for the running synthetic suite and the domain data build, then runs the rest sequentially.
cd "$(dirname "$0")/../.." || exit 1
P=.venv/bin/python
while pgrep -f "benchmarks/gpu_synthetic_suite.py" >/dev/null; do sleep 30; done
while pgrep -f "data_prep/build_domains.py" >/dev/null; do sleep 30; done
$P benchmarks/gpu_domains_3seed.py        > benchmark_results/paper/multidomain_3seed/gpu_domains_3seed.log 2>&1
$P benchmarks/gpu_domains_3seed.py image code --kinds=rope-hybrid-h6 > benchmark_results/paper/multidomain_3seed/gpu_domains_ropehybrid.log 2>&1  # diagnostic: ALiBi vs RoPE inside DEEPA
$P benchmarks/gpu_wikitext103_fair.py     > benchmark_results/paper/wikitext103_lr_and_horizon/gpu_wikitext103_fair.log 2>&1
$P benchmarks/gpu_extrap_16k.py 1 2       > benchmark_results/paper/length_extrapolation_16k/gpu_extrap_16k_seeds12.log 2>&1
echo QUEUE12 DONE > benchmark_results/_queue_logs/queue12_phaseB.log
