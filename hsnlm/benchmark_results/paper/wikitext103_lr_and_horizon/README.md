# WikiText-103: LR sweep + 5 tokens/param (4,406 steps)

**Status:** current  ·  **Paper:** Table 6 (WikiText-103 4.4k row)  ·  **Script(s):** [`gpu_wikitext103_fair.py`](../../../benchmarks/gpu_wikitext103_fair.py)

## Summary

Both models' best LR is 2e-3; at 4,406 steps DEEPA leads on both seeds (1.4250 vs 1.4404 full-val bpb).

## What it tests and how (from the script)

```text
B4 -- is the WikiText-103 loss (gpu_wikitext103_3seed.json: pure ALiBi beat the hybrid by ~0.012 bpb on the full val set
at 1,500 steps / lr 2e-3) real, or an artefact of (a) a learning rate tuned on enwik8 or (b) a short training horizon?
(a) LR sweep for BOTH models: lr in {1e-3, 2e-3 (existing runs), 3e-3}, 1,500 steps, seed 0.
(b) Each model at its own best LR from (a), trained to 5 tokens/param (4,406 steps, the gpu_scaling_law.py horizon), seeds 0, 1.
Same data / eval as gpu_wikitext103_3seed.py. ALiBi uses the exact fused path for both models (deepa_common.ALIBI_IMPL).
```

## Reproduce

From the `hsnlm/` root (GPU runs need CUDA; see the top-level README for the environment):

```bash
python benchmarks/gpu_wikitext103_fair.py
```

## Files

| file | kind | size |
|---|---|---|
| [`gpu_wikitext103_fair.json`](gpu_wikitext103_fair.json) | results (machine-readable) | 6.8 KB |
| [`gpu_wikitext103_fair.log`](gpu_wikitext103_fair.log) | stdout of the run | 3.3 KB |

## Results: `gpu_wikitext103_fair.json`

Auto-extracted per-run metrics (bits/byte lower is better; accuracies in [0,1]; `x` = not solved). Paper tables aggregate these over seeds.

| run | val_bpb_full | val_bpb | val_bpb_len2048 | val_bpb_len4096 | tok_s | peak_mem_mb | train_s |
|---|---|---|---|---|---|---|---|
| `sweep|alibi-hybrid-h6|0.002` | 1.6283 | 1.6029 | 1.5929 | 1.6381 | 80,828.8 | 2,718.8 | 304.1 |
| `sweep|alibi-hybrid-h6|0.001` | 1.6390 | 1.6237 | 1.6155 | 1.6566 | 86,699.6 | 2,750.9 | 283.5 |
| `sweep|alibi-hybrid-h6|0.003` | 1.6283 | 1.6067 | 1.5958 | 1.6405 | 86,527.3 | 2,749.7 | 284.0 |
| `long|alibi-hybrid-h6|seed0` | 1.4266 | 1.3757 | 1.3520 | 1.4102 | 86,524.3 | 2,749.7 | 834.3 |
| `long|alibi-hybrid-h6|seed1` | 1.4233 | 1.3726 | 1.3469 | 1.4024 | 86,564.1 | 2,749.7 | 833.9 |
| `sweep|alibi-pure|0.002` | 1.6154 | 1.5788 | 1.5496 | 1.5949 | 85,656.5 | 2,240.7 | 286.9 |
| `sweep|alibi-pure|0.001` | 1.6490 | 1.6178 | 1.5922 | 1.6295 | 117,117.2 | 2,284.8 | 209.8 |
| `sweep|alibi-pure|0.003` | 1.6179 | 1.5834 | 1.5549 | 1.5956 | 116,791.6 | 2,284.8 | 210.4 |
| `long|alibi-pure|seed0` | 1.4410 | 1.3902 | 1.3596 | 1.4186 | 116,839.1 | 2,284.8 | 617.8 |
| `long|alibi-pure|seed1` | 1.4398 | 1.3875 | 1.3567 | 1.4140 | 116,770.4 | 2,284.8 | 618.2 |
