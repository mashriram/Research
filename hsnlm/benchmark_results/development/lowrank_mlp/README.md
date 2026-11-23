# Low-rank MLP factorisation

**Status:** negative result  ·  **Paper:** not in paper (development history)  ·  **Script(s):** [`gpu_lowrank_mlp.py`](../../../benchmarks/gpu_lowrank_mlp.py)

## What it tests and how (from the script)

```text
Tests one legitimate, separable idea extracted from the pasted 'GOAT LoSparse' code: does a
FIXED-rank low-rank factorization of the MLP's up/down projections save real parameters without
hurting quality, at the scale we already have a dense baseline for? Deliberately drops the pasted
code's two unvalidated/buggy pieces: (1) adaptive rank growth, which reassigns nn.Parameter objects
post-hoc -- the already-constructed AdamW optimizer keeps references to the OLD tensors, so expanded
rank columns silently never receive a gradient update; (2) the sparse "specialist" path, which adds
its own magnitude-mask machinery under torch.compile with a data-dependent topk -> constant
recompilation. This isolates the one piece worth measuring: plain bottleneck factorization, fixed
rank, ordinary backprop, no exotic machinery. Same recipe as gpu_scale.py (14M-param hybrid-fused
config, enwik8, bf16, 1500 steps) for direct comparison against the dense result (1.021 bpb, 14.64M
params, 760.2s, 4737.7 MB, from benchmark_results/development/gpu_scale/gpu_scale.json).
```

## Reproduce

From the `hsnlm/` root (GPU runs need CUDA; see the top-level README for the environment):

```bash
python benchmarks/gpu_lowrank_mlp.py
```

## Files

| file | kind | size |
|---|---|---|
| [`gpu_lowrank_mlp.json`](gpu_lowrank_mlp.json) | results (machine-readable) | 0.8 KB |
| [`gpu_lowrank_mlp.log`](gpu_lowrank_mlp.log) | stdout of the run | 2.1 KB |

## Results: `gpu_lowrank_mlp.json`

Auto-extracted per-run metrics (bits/byte lower is better; accuracies in [0,1]; `x` = not solved). Paper tables aggregate these over seeds.

| run | val_bpb | val_bpb_len2048 | val_bpb_len4096 | tok_s | peak_mem_mb | train_s | params |
|---|---|---|---|---|---|---|---|
| `rank64|0.002` | 1.0547 | 1.0373 | 1.1316 | 33,833.6 | 4,684.3 | 726.4 | 7,861,592 |
| `rank128|0.002` | 1.0784 | 1.0630 | 1.1643 | 33,635.7 | 4,738.7 | 730.7 | 9,689,432 |
| `rank192|0.002` | 1.0696 | 1.0563 | 1.1623 | 32,945.4 | 4,801.3 | 746.0 | 11,517,272 |
