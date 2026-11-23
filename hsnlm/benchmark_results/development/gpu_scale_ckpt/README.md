# Scaling past 14M with gradient checkpointing (7:1 and 3:1)

**Status:** superseded by scaling_2m_to_38m  ·  **Paper:** not in paper (development history)  ·  **Script(s):** [`gpu_scale_ckpt.py`](../../../benchmarks/gpu_scale_ckpt.py), [`gpu_scale_ckpt_31.py`](../../../benchmarks/gpu_scale_ckpt_31.py)

## Summary

Source of the '206M params fit in 5.3 GB' memory statement.

## What it tests and how (from the script)

```text
Push past the 14M-param VRAM ceiling hit in gpu_scale.py, via gradient checkpointing (trade compute
for activation memory) applied uniformly to all three architectures -- more general than the existing
reversible-block mechanism (hsnlm/reversible.py), which only supports a homogeneous sheaf-mixer stack,
not GPT or the hybrid pattern. Each block's forward is recomputed during backward instead of its
activations being stored, so peak memory becomes closer to O(one block) instead of O(depth x block).
Same recipe/data as gpu_scale.py; width/layers raised to whatever this 8GB card can hold with
checkpointing on. usage: gpu_scale_ckpt.py probe   -- just print peak memory for one step at WIDTH/LAYERS
       gpu_scale_ckpt.py            -- full training run (sweep -> 1500-step main stage)
```

## Reproduce

From the `hsnlm/` root (GPU runs need CUDA; see the top-level README for the environment):

```bash
python benchmarks/gpu_scale_ckpt.py
python benchmarks/gpu_scale_ckpt_31.py
```

## Files

| file | kind | size |
|---|---|---|
| [`gpu_scale_ckpt.json`](gpu_scale_ckpt.json) | results (machine-readable) | 2.0 KB |
| [`gpu_scale_ckpt.log`](gpu_scale_ckpt.log) | stdout of the run | 6.1 KB |
| [`gpu_scale_ckpt_31.json`](gpu_scale_ckpt_31.json) | results (machine-readable) | 0.7 KB |
| [`gpu_scale_ckpt_31.log`](gpu_scale_ckpt_31.log) | stdout of the run | 2.2 KB |

## Results: `gpu_scale_ckpt.json`

Auto-extracted per-run metrics (bits/byte lower is better; accuracies in [0,1]; `x` = not solved). Paper tables aggregate these over seeds.

| run | val_bpb | val_bpb_len2048 | val_bpb_len4096 | tok_s | peak_mem_mb | train_s | params |
|---|---|---|---|---|---|---|---|
| `sweep|gpt|0.001` | 4.4538 |  |  | 7,051.6 | 4,553.5 | 92.9372 | 201,622,528 |
| `sweep|gpt|0.002` | 4.4137 |  |  | 7,046.9 | 4,553.5 | 92.9991 | 201,622,528 |
| `sweep|hsn-fused|0.001` | 4.1301 |  |  | 3,765.6 | 5,181.0 | 174.0 | 200,265,472 |
| `sweep|hsn-fused|0.002` | 3.8737 |  |  | 3,938.5 | 5,181.0 | 166.4 | 200,265,472 |
| `sweep|hybrid-fused|0.001` | 4.0734 |  |  | 4,087.5 | 5,258.0 | 160.3 | 206,276,000 |
| `sweep|hybrid-fused|0.002` | 3.8421 |  |  | 4,087.4 | 5,258.0 | 160.3 | 206,276,000 |
| `best_lr` |  |  |  |  |  |  |  |
| `lm|gpt` | 2.4402 | 2.4586 | 2.5636 | 6,988.6 | 4,553.5 | 703.3 | 201,622,528 |
| `lm|hsn-fused` | 1.5067 | 1.4982 | 1.5552 | 3,936.5 | 5,181.0 | 1,248.6 | 200,265,472 |
| `lm|hybrid-fused` | 1.5138 | 1.5073 | 1.5621 | 4,082.8 | 5,258.0 | 1,203.9 | 206,276,000 |

## Results: `gpu_scale_ckpt_31.json`

Auto-extracted per-run metrics (bits/byte lower is better; accuracies in [0,1]; `x` = not solved). Paper tables aggregate these over seeds.

| run | val_bpb | val_bpb_len2048 | val_bpb_len4096 | tok_s | peak_mem_mb | train_s | params |
|---|---|---|---|---|---|---|---|
| `sweep|0.001` | 3.7537 |  |  | 4,083.8 | 5,247.5 | 160.5 | 203,930,688 |
| `sweep|0.002` | 3.6350 |  |  | 4,303.0 | 5,247.8 | 152.3 | 203,930,688 |
| `lm` | 1.5096 | 1.5045 | 1.5678 | 4,296.6 | 5,247.8 | 1,144.0 | 203,930,688 |
