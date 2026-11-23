# Hybrid attention-ratio ablation (RoPE hybrids)

**Status:** superseded  ·  **Paper:** not in paper (development history)  ·  **Script(s):** [`gpu_ratio.py`](../../../benchmarks/gpu_ratio.py)

## What it tests and how (from the script)

```text
G5 from findings_report.md's GPU plan: hybrid attention-ratio ablation. Same scale/recipe as
gpu_scale.py (14M params, width 384, 8 layers, ctx 1024, bf16, enwik8) so results are directly
comparable to the already-measured endpoints: 0 attention layers = hsn-fused (1.199 bpb, gpu_scale.py),
8 attention layers = gpt (1.103 bpb, gpu_scale.py), 1 attention layer = hybrid-fused 7:1 (1.021 bpb,
gpu_scale.py). This script fills in the missing ratios: 2, 3, 4 attention layers out of 8.
Attention layers are spread evenly across depth rather than clustered at the end.
```

## Reproduce

From the `hsnlm/` root (GPU runs need CUDA; see the top-level README for the environment):

```bash
python benchmarks/gpu_ratio.py
```

## Files

| file | kind | size |
|---|---|---|
| [`gpu_ratio.json`](gpu_ratio.json) | results (machine-readable) | 0.8 KB |
| [`gpu_ratio.log`](gpu_ratio.log) | stdout of the run | 2.4 KB |

## Results: `gpu_ratio.json`

Auto-extracted per-run metrics (bits/byte lower is better; accuracies in [0,1]; `x` = not solved). Paper tables aggregate these over seeds.

| run | val_bpb | val_bpb_len2048 | val_bpb_len4096 | tok_s | peak_mem_mb | train_s | params |
|---|---|---|---|---|---|---|---|
| `ratio|2` | 1.0136 | 0.9950 | 1.0835 | 34,179.5 | 4,471.6 | 719.0 | 14,468,592 |
| `ratio|3` | 1.0382 | 1.0223 | 1.1157 | 37,384.1 | 4,197.4 | 657.4 | 14,295,688 |
| `ratio|4` | 1.0449 | 1.0299 | 1.1281 | 40,489.4 | 3,922.5 | 607.0 | 14,122,784 |
