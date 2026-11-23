# Early decode benchmarks (torch.cat KV cache; per-layer cost)

**Status:** superseded by decode_and_serving  ·  **Paper:** not in paper (development history)  ·  **Script(s):** [`gpu_decode.py`](../../../benchmarks/gpu_decode.py), [`gpu_decode_hybrid_recheck.py`](../../../benchmarks/gpu_decode_hybrid_recheck.py)

## Summary

The GPT baseline here grew its KV cache with torch.cat (unfair); use decode_and_serving instead.

## What it tests and how (from the script)

```text
GPU port of experiments/decode_o1.py: per-token decode cost of ONE layer vs context length already
consumed (batch 1, D=1024, 16 heads). CUDA-synced timing; GPU decode is latency-bound by kernel-launch
overhead at batch 1, so this checks whether the CPU-measured O(1) vs O(N) shape survives on GPU.
```

## Reproduce

From the `hsnlm/` root (GPU runs need CUDA; see the top-level README for the environment):

```bash
python benchmarks/gpu_decode_hybrid_recheck.py
```

## Files

| file | kind | size |
|---|---|---|
| [`gpu_decode_hybrid_recheck.json`](gpu_decode_hybrid_recheck.json) | results (machine-readable) | 0.5 KB |

## Results: `gpu_decode_hybrid_recheck.json`

Auto-extracted per-run metrics (bits/byte lower is better; accuracies in [0,1]; `x` = not solved). Paper tables aggregate these over seeds.

| run | step_ms | kv_cache_mb | sheafdelta_state_mb |
|---|---|---|---|
| `256` | 5.2470 | 4.1940 | 1.9660 |
| `1024` | 5.9630 | 16.7770 | 1.9660 |
| `4096` | 7.2750 | 67.1090 | 1.9660 |
| `16384` | 13.6400 | 268.4 | 1.9660 |
| `65536` | 41.6730 | 1,073.7 | 1.9660 |
