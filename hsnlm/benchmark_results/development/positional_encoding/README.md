# NoPE / ALiBi / RoPE positional-encoding comparison

**Status:** current (motivates ALiBi)  ·  **Paper:** not in paper (development history)  ·  **Script(s):** [`gpu_posenc.py`](../../../benchmarks/gpu_posenc.py), [`gpu_posenc_recheck.py`](../../../benchmarks/gpu_posenc_recheck.py)

## Summary

Source of the NoPE (2.344 bpb) negative result cited in the ablation section.

## What it tests and how (from the script)

```text
Checks whether NoPE or ALiBi -- the two context-extension techniques flagged as untested in every
caveat so far (they need retraining from scratch with the position-encoding choice baked in, unlike
YaRN which is a post-hoc fix, see gpu_yarn.py) -- beat plain RoPE, YaRN, or the hybrid 3:1 architecture
on length extrapolation. Same 14M-param/width-384/8-layer/ctx-1024 recipe as gpu_scale.py, real enwik8
training from scratch for each variant. Compares against: plain RoPE (1.103/2.111/3.547, gpu_scale.py),
YaRN (1.130/1.370/1.472, gpu_yarn.py), hybrid 3:1 (1.014/0.995/1.084, gpu_ratio.py).
```

## Reproduce

From the `hsnlm/` root (GPU runs need CUDA; see the top-level README for the environment):

```bash
python benchmarks/gpu_posenc.py
python benchmarks/gpu_posenc_recheck.py
```

## Files

| file | kind | size |
|---|---|---|
| [`gpu_posenc.json`](gpu_posenc.json) | results (machine-readable) | 0.6 KB |
| [`gpu_posenc.log`](gpu_posenc.log) | stdout of the run | 3.2 KB |
| [`gpu_posenc_recheck.json`](gpu_posenc_recheck.json) | results (machine-readable) | 0.7 KB |
| [`gpu_posenc_recheck.log`](gpu_posenc_recheck.log) | stdout of the run | 1.8 KB |

Archived (invalid, do not use): [`gpu_posenc.json.BROKEN_residualbug`](../../_archive/gpu_posenc.json.BROKEN_residualbug), [`gpu_posenc_recheck.json.BROKEN_residualbug`](../../_archive/gpu_posenc_recheck.json.BROKEN_residualbug)

## Results: `gpu_posenc.json`

Auto-extracted per-run metrics (bits/byte lower is better; accuracies in [0,1]; `x` = not solved). Paper tables aggregate these over seeds.

| run | val_bpb | val_bpb_len2048 | val_bpb_len4096 | tok_s | train_s | params |
|---|---|---|---|---|---|---|
| `sweep|nope|0.001` | 4.1404 |  |  | 79,736.9 | 20.5476 | 14,260,608 |
| `sweep|nope|0.002` | 4.0885 |  |  | 80,753.0 | 20.2890 | 14,260,608 |
| `best_lr` |  |  |  |  |  |  |
| `lm|nope` | 2.3438 | 2.7299 | 3.6776 | 80,545.6 | 305.1 | 14,260,608 |

## Results: `gpu_posenc_recheck.json`

Auto-extracted per-run metrics (bits/byte lower is better; accuracies in [0,1]; `x` = not solved). Paper tables aggregate these over seeds.

| run | val_bpb_len2048 | val_bpb_len4096 | tok_s | peak_mem_mb | train_s | params |
|---|---|---|---|---|---|---|
| `lm|alibi-pure` | 1.0062 | 1.0872 | 56,970.0 | 2,726.5 | 431.4 | 14,260,608 |
| `lm|alibi-hybrid` | 1.0048 | 1.0875 | 31,912.0 | 4,030.8 | 770.1 | 14,468,592 |
