# Largest single-GPU LM run (pre-checkpointing)

**Status:** superseded  ·  **Paper:** not in paper (development history)  ·  **Script(s):** [`gpu_scale.py`](../../../benchmarks/gpu_scale.py)

## What it tests and how (from the script)

```text
Larger-than-CPU-suite LM quality/speed run on enwik8, scaled to what one 8GB GPU can hold comfortably
(dim 384, 8 layers, ctx 1024) -- the GPU analogue of cpu_suite's LM stage, but ~15x the parameters and
4x the context. Compares GPT vs HSN-fused (pure fla-fused sheaf-delta) vs Hybrid-fused (7 sheaf-delta + 1
attention). bf16 autocast. Resumable: benchmark_results/development/gpu_scale/gpu_scale.json. usage: gpu_scale.py [smoke]
```

## Reproduce

From the `hsnlm/` root (GPU runs need CUDA; see the top-level README for the environment):

```bash
python benchmarks/gpu_scale.py
```

## Files

| file | kind | size |
|---|---|---|
| [`gpu_scale.json`](gpu_scale.json) | results (machine-readable) | 5.7 KB |
| [`gpu_scale.log`](gpu_scale.log) | stdout of the run | 6.1 KB |

## Results: `gpu_scale.json`

Auto-extracted per-run metrics (bits/byte lower is better; accuracies in [0,1]; `x` = not solved). Paper tables aggregate these over seeds.

| run | val_bpb | val_bpb_len2048 | val_bpb_len4096 | tok_s | peak_mem_mb | train_s | params |
|---|---|---|---|---|---|---|---|
| `sweep|gpt|0.001` | 4.1313 |  |  | 50,585.1 | 2,897.6 | 32.3890 | 14,260,608 |
| `sweep|gpt|0.002` | 3.9044 |  |  | 35,725.2 | 2,896.8 | 45.8612 | 14,260,608 |
| `sweep|hsn-fused|0.001` | 3.6397 |  |  | 12,649.6 | 4,946.5 | 129.5 | 14,224,576 |
| `sweep|hsn-fused|0.002` | 3.2228 |  |  | 30,635.1 | 4,946.5 | 53.4811 | 14,224,576 |
| `sweep|hybrid-fused|0.001` | 3.5880 |  |  | 32,375.0 | 4,737.7 | 50.6070 | 14,641,496 |
| `sweep|hybrid-fused|0.002` | 3.1811 |  |  | 32,407.8 | 4,737.7 | 50.5557 | 14,641,496 |
| `best_lr` |  |  |  |  |  |  |  |
| `lm|gpt` | 1.1030 | 2.1106 | 3.5472 | 66,014.8 | 2,896.8 | 372.3 | 14,260,608 |
| `lm|hsn-fused` | 1.1988 | 1.1865 | 1.2475 | 30,631.1 | 4,946.5 | 802.3 | 14,224,576 |
| `lm|hybrid-fused` | 1.0212 | 1.0035 | 1.0904 | 32,329.4 | 4,737.7 | 760.2 | 14,641,496 |
