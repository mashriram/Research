# YaRN RoPE rescaling for GPT and the RoPE hybrid

**Status:** superseded  ·  **Paper:** not in paper (development history)  ·  **Script(s):** [`gpu_yarn.py`](../../../benchmarks/gpu_yarn.py), [`gpu_yarn_recheck.py`](../../../benchmarks/gpu_yarn_recheck.py), [`gpu_yarn_hybrid.py`](../../../benchmarks/gpu_yarn_hybrid.py)

## What it tests and how (from the script)

```text
Fair-baseline check: both findings reports flagged the GPT baseline as 'plain RoPE, no NoPE/YaRN/
ALiBi' -- i.e. the worst-case Transformer, not the best-known one. This retrains the same 14M-param
GPT (width 384, 8 layers, ctx 1024, same recipe as gpu_scale.py) and evaluates length extrapolation
both with plain RoPE (replicating gpu_scale.py's number as a sanity check) and with YaRN-scaled RoPE
(NTK-by-parts frequency interpolation + attention temperature scaling), applied ONLY at eval time for
contexts beyond the 1024 training length -- no retraining needed for YaRN itself, exactly how it's
used in practice to extend an already-trained model's context. Compares both against the hybrid-fused
3:1 number already measured (1.084 bpb @ 4096, gpu_ratio.json/gpu_scale.json).
```

## Reproduce

From the `hsnlm/` root (GPU runs need CUDA; see the top-level README for the environment):

```bash
python benchmarks/gpu_yarn_recheck.py
python benchmarks/gpu_yarn_hybrid.py
```

## Files

| file | kind | size |
|---|---|---|
| [`gpu_yarn.log`](gpu_yarn.log) | stdout of the run | 0.8 KB |
| [`gpu_yarn_recheck.json`](gpu_yarn_recheck.json) | results (machine-readable) | 0.3 KB |
| [`gpu_yarn_recheck.log`](gpu_yarn_recheck.log) | stdout of the run | 0.0 KB |
| [`gpu_yarn_hybrid.json`](gpu_yarn_hybrid.json) | results (machine-readable) | 0.4 KB |

Archived (invalid, do not use): [`gpu_yarn.json.BROKEN_residualbug`](../../_archive/gpu_yarn.json.BROKEN_residualbug), [`gpu_yarn_recheck.json.BROKEN_residualbug`](../../_archive/gpu_yarn_recheck.json.BROKEN_residualbug)

## Results: `gpu_yarn_recheck.json`

Auto-extracted per-run metrics (bits/byte lower is better; accuracies in [0,1]; `x` = not solved). Paper tables aggregate these over seeds.

| run | train_s | params |
|---|---|---|
| `lm` | 373.3 | 14,260,608 |

## Results: `gpu_yarn_hybrid.json`

Auto-extracted per-run metrics (bits/byte lower is better; accuracies in [0,1]; `x` = not solved). Paper tables aggregate these over seeds.

| run | tok_s | peak_mem_mb | train_s | params |
|---|---|---|---|---|
| `lm` | 39,012.9 | 4,081.4 | 629.9 | 14,468,592 |
