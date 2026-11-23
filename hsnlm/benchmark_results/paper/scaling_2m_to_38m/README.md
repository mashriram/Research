# Parameter scaling 2M -> 38M at 5 tokens/param

**Status:** current  ·  **Paper:** Table 5 (tab:scaling), Fig. 2b  ·  **Script(s):** [`gpu_scaling_law.py`](../../../benchmarks/gpu_scaling_law.py)

## Summary

Seed 0 only. At 38M DEEPA used activation checkpointing (its tok/s there is not like-for-like).

## What it tests and how (from the script)

```text
Reviewer item 6 (scalability): alibi-hybrid-h6 (3:1 SheafDelta:ALiBi) vs pure ALiBi at four sizes, each trained at a
matched 5 tokens/parameter (5 x the hybrid's param count; both models at a scale get the identical step count, so
they are compute/token-matched to within the hybrid's +1.2-1.4% params), seed 0, torch.compile.
   scale  dim layers attn_heads mixer_heads  pattern
   ~2M    192   4      4          3          ddda
   ~6M    256   8      4          4          dddaddda
   ~14M   384   8      8          6          dddaddda      (the paper's model)
   ~38M   512  12      8          8          dddadddaddda
Mixer head dim is 64 at every scale. 5 tok/param needs up to ~190M training bytes, which would be ~4.7 epochs of the
usual 40M-byte slice; to avoid a repeated-data confound the training set here is ALL of enwik8 except the standard
validation slice: bytes [0,40M) + [41M,100M) = 99M bytes (<= 1.95 epochs at the largest scale). Validation is the
SAME bytes [40M,41M) used everywhere else in the paper.
LR: 2e-3 (the tuned value at 14M) for the three smaller scales. At ~38M, each architecture picks its own LR from
{1e-3, 2e-3} by a 1,000-step sweep run (seed 100), so neither model is handicapped by a shared untuned LR.
Activation checkpointing is turned on automatically only if a B=16 step OOMs (recorded per run).
```

## Reproduce

From the `hsnlm/` root (GPU runs need CUDA; see the top-level README for the environment):

```bash
python benchmarks/gpu_scaling_law.py
```

## Files

| file | kind | size |
|---|---|---|
| [`gpu_scaling_law.json`](gpu_scaling_law.json) | results (machine-readable) | 11.5 KB |
| [`gpu_scaling_law.log`](gpu_scaling_law.log) | stdout of the run | 10.9 KB |

## Results: `gpu_scaling_law.json`

Auto-extracted per-run metrics (bits/byte lower is better; accuracies in [0,1]; `x` = not solved). Paper tables aggregate these over seeds.

| run | val_bpb_full | val_bpb | val_bpb_len2048 | val_bpb_len4096 | tok_s | peak_mem_mb | train_s |
|---|---|---|---|---|---|---|---|
| `2M|alibi-hybrid-h6` | 2.1117 | 1.7638 | 1.7554 | 1.7979 | 392,707.0 | 703.4 | 23.4887 |
| `2M|alibi-pure` | 2.4094 | 2.0407 | 2.0278 | 2.0708 | 402,245.3 | 585.8 | 22.9318 |
| `6M|alibi-hybrid-h6` | 1.5983 | 1.0523 | 1.0282 | 1.1010 | 148,485.8 | 1,773.4 | 216.9 |
| `6M|alibi-pure` | 1.6196 | 1.0749 | 1.0436 | 1.1138 | 164,210.2 | 1,443.6 | 196.2 |
| `14M|alibi-hybrid-h6` | 1.3938 | 0.8765 | 0.8534 | 0.9130 | 80,515.8 | 2,719.2 | 896.6 |
| `14M|alibi-pure` | 1.4301 | 0.9058 | 0.8757 | 0.9344 | 85,494.3 | 2,240.6 | 844.4 |
| `sweep|38M|alibi-hybrid-h6|0.001` |  | 1.1230 |  |  | 14,961.4 | 1,634.6 | 1,095.1 |
| `sweep|38M|alibi-hybrid-h6|0.002` |  | 1.1143 |  |  | 14,953.8 | 1,633.8 | 1,095.6 |
| `38M|alibi-hybrid-h6` | 1.2349 | 0.7304 | 0.7085 | 0.7567 | 14,947.8 | 1,633.8 | 12,825.2 |
| `sweep|38M|alibi-pure|0.001` |  | 1.1649 |  |  | 43,128.4 | 4,480.5 | 379.9 |
| `sweep|38M|alibi-pure|0.002` |  | 1.1382 |  |  | 43,032.8 | 4,480.1 | 380.7 |
| `38M|alibi-pure` | 1.2624 | 0.7523 | 0.7313 | 0.7773 | 43,090.7 | 4,480.2 | 4,449.0 |
