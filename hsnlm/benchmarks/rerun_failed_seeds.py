"""Rerun the DEEPA seeds that stalled at a plateau in gpu_synthetic_proper.json, at lr=1e-3 (same protocol otherwise).
copy|alibi-hybrid-h6|seed1 (failed at 3e-3) and mqar64_T512|alibi-hybrid-h6|seed2 (failed at 3e-3).
Writes benchmark_results/development/misc/gpu_synthetic_rerun.json; never touches the main suite JSON."""
import os, sys, json
SRC = open("benchmarks/gpu_synthetic_proper.py").read().split("\nONLY = ")[0]
os.environ["PROPER_OUT"] = "benchmark_results/development/misc/gpu_synthetic_rerun.json"
exec(compile(SRC, "gpu_synthetic_proper.py", "exec"))
for task, seed in [("copy", 1), ("mqar64_T512", 2)]:
    k = f"{task}|alibi-hybrid-h6|seed{seed}|lr0.001"
    if k in R: continue
    R[k] = run(task, "alibi-hybrid-h6", 1e-3, seed); save()
    log("RERUN", k, {a: b for a, b in R[k].items() if a.startswith("acc") or a in ("steps_to_solve", "reached_len")})
log("RERUN DONE")
