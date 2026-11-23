#!/bin/bash
# (1) iso-STEP control: Pure ALiBi at 13,300 steps (closes the over-repetition loophole in the equal-wall-clock comparison).
# (2) LR-robustness grid for the seed-sensitive tasks: every LR x seeds 1,2, LR then chosen by 3-seed mean for every model.
cd "$(dirname "$0")/../.." || exit 1
KINDS=alibi-pure .venv/bin/python benchmarks/gpu_chinchilla_15m.py 13300 0,1,2 > benchmark_results/paper/enwik8_equal_wallclock/gpu_isostep_pure_13300.log 2>&1
PROPER_FULLGRID=1 .venv/bin/python benchmarks/gpu_synthetic_proper.py copy_paper copy mqar64_T512 > benchmark_results/paper/synthetic_converged/gpu_synthetic_grid.log 2>&1
echo QUEUE15 DONE > benchmark_results/_queue_logs/queue15_robust.log
