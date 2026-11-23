#!/bin/bash
# Converged synthetic suite (gpu_synthetic_proper.py) after the LRA queue. Replaces the earlier 12k-step MQAR-only rerun.
cd "$(dirname "$0")/../.." || exit 1
until [ -f benchmark_results/_queue_logs/queue13_lra.log ]; do sleep 120; done
.venv/bin/python benchmarks/gpu_lra_subset.py > benchmark_results/paper/long_range_arena/gpu_lra_subset_resume.log 2>&1  # resumes (skips finished runs); completes text/image if the first pass hit OOM
.venv/bin/python benchmarks/gpu_domains_3seed.py --kinds=sheafdelta-pure > benchmark_results/paper/multidomain_3seed/gpu_domains_sheafpure.log 2>&1
PROPER_ARCHS=alibi-hybrid-h6-beta1,sheafdelta-pure-beta1 PROPER_OUT=benchmark_results/paper/parity_negative_eigenvalue_control/gpu_parity_beta_control.json .venv/bin/python benchmarks/gpu_synthetic_proper.py parity > benchmark_results/paper/parity_negative_eigenvalue_control/gpu_parity_beta_control.log 2>&1  # negative-eigenvalue control
KINDS=alibi-hybrid-h6 .venv/bin/python benchmarks/gpu_chinchilla_15m.py 13300 0,1,2 > benchmark_results/paper/enwik8_equal_wallclock/gpu_isotime_13300.log 2>&1  # iso-wall-clock vs pure ALiBi 18k (folded-kernel speed ratio 117.0k/86.5k = 1.353)
.venv/bin/python benchmarks/gpu_synthetic_proper.py > benchmark_results/paper/synthetic_converged/gpu_synthetic_proper.log 2>&1
echo QUEUE14 DONE > benchmark_results/_queue_logs/queue14_proper.log
