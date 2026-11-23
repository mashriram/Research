#!/bin/bash
# Continuation of queue10 after its parent shell died: wait for the running gpu_extrap_16k.py, then the rest.
cd "$(dirname "$0")/../.." || exit 1
while pgrep -f "benchmarks/gpu_extrap_16k.py" >/dev/null; do sleep 20; done
P=.venv/bin/python
$P benchmarks/gpu_extrap_16k.py        >> benchmark_results/paper/length_extrapolation_16k/gpu_extrap_16k.log 2>&1   # no-op if complete; resumes otherwise
$P benchmarks/gpu_wikitext103_3seed.py > benchmark_results/paper/wikitext103_3seed/gpu_wikitext103_3seed.log 2>&1
$P benchmarks/gpu_scaling_law.py       > benchmark_results/paper/scaling_2m_to_38m/gpu_scaling_law.log 2>&1
echo QUEUE DONE >> benchmark_results/_queue_logs/queue10_phaseA.log
