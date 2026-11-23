# Iso-quality: tokens / wall-clock to reach RoPE GPT's final loss

**Status:** current  ·  **Paper:** Sec. 5.1 (42.4% fewer tokens, 30.3% less time)  ·  **Script(s):** [`gpu_isoquality_compiled.py`](../../../benchmarks/gpu_isoquality_compiled.py)

## What it tests and how (from the script)

```text
Fair iso-quality (time-to-target-loss) comparison: does the RoPE 3:1 hybrid reach GPT's final
quality in less wall-clock time once BOTH are compiled the same way (torch.compile(model), applied
uniformly -- not compiling only the hybrid's mixer while leaving GPT uncompiled, which is not a fair
comparison). Same recipe as gpu_scale.py/gpu_ratio.py (14M params, width 384, 8 layers, ctx 1024,
lr=0.002, enwik8), real training curves saved for both so the iso-quality crossing point is measured,
not guessed from an uncompiled baseline's curve.
```

## Reproduce

From the `hsnlm/` root (GPU runs need CUDA; see the top-level README for the environment):

```bash
python benchmarks/gpu_isoquality_compiled.py
```

## Files

| file | kind | size |
|---|---|---|
| [`gpu_isoquality_compiled.json`](gpu_isoquality_compiled.json) | results (machine-readable) | 3.1 KB |
| [`gpu_isoquality_compiled.log`](gpu_isoquality_compiled.log) | stdout of the run | 1.1 KB |

Archived (invalid, do not use): [`gpu_isoquality_compiled.json.OLD_noextrap`](../../_archive/gpu_isoquality_compiled.json.OLD_noextrap)

## Results: `gpu_isoquality_compiled.json`

Auto-extracted per-run metrics (bits/byte lower is better; accuracies in [0,1]; `x` = not solved). Paper tables aggregate these over seeds.

| run | val_bpb | val_bpb_len2048 | val_bpb_len4096 | tok_s | peak_mem_mb | train_s | params |
|---|---|---|---|---|---|---|---|
| `gpt|compiled` | 1.1014 | 2.0244 | 3.3221 | 97,771.9 | 2,407.9 | 251.4 | 14,260,608 |
| `hybrid|compiled` | 1.0305 | 1.0143 | 1.0981 | 80,874.1 | 2,866.2 | 303.9 | 14,468,592 |
