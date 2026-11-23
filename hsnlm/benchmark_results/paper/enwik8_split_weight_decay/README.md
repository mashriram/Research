# Uniform vs split weight decay (1,500 steps, 3 seeds)

**Status:** current  ·  **Paper:** Table 4 (1.5k rows)  ·  **Script(s):** [`gpu_splitwd_3seed.py`](../../../benchmarks/gpu_splitwd_3seed.py)

## What it tests and how (from the script)

```text
Checks whether train_deepa_phase1.py's split weight-decay optimizer (decay on 2D+ params
excluding the Householder '.V', no decay on 1D params/biases -- so SheafDeltaMixer's s.ba.bias
decay-gate bias, init'd to +3.0 for a long memory horizon, isn't dragged toward 0 every step the
way a single uniform weight_decay=0.05 group would) is a real improvement over the uniform-WD
recipe used for every other number in this project, or just single-run noise (Stage 1 of
queue7_chinchilla.log showed h6 at 1.0149 bpb @1024 under split-WD vs h6's uniform-WD 3-seed mean
of 1.0291 -- a bigger gap than h6's own uniform-WD seed noise (std 0.0016), but still one run).
Reruns h6 (ALiBi-hybrid 3:1 + mixer_heads=6) AND pure ALiBi at 3 seeds each, both under the exact
same split-WD optimizer (imported from train_deepa_phase1.py, not reimplemented) -- giving pure
ALiBi the identical treatment keeps this a fair, single-variable (optimizer only) comparison.
Same 1,500-step/14.3-14.4M-param recipe as gpu_v2_ablation.py's h6 and gpu_multiseed_compiled.py's
alibi-pure, so directly comparable to both.
```

## Reproduce

From the `hsnlm/` root (GPU runs need CUDA; see the top-level README for the environment):

```bash
python benchmarks/gpu_splitwd_3seed.py
```

## Files

| file | kind | size |
|---|---|---|
| [`gpu_splitwd_3seed.json`](gpu_splitwd_3seed.json) | results (machine-readable) | 1.8 KB |

## Results: `gpu_splitwd_3seed.json`

Auto-extracted per-run metrics (bits/byte lower is better; accuracies in [0,1]; `x` = not solved). Paper tables aggregate these over seeds.

| run | val_bpb | val_bpb_len2048 | val_bpb_len4096 | tok_s | peak_mem_mb | train_s | params |
|---|---|---|---|---|---|---|---|
| `alibi-hybrid-h6|seed0` | 1.0321 | 1.0115 | 1.0882 | 81,408.4 | 2,720.0 | 301.9 | 14,436,348 |
| `alibi-hybrid-h6|seed1` | 1.0286 | 1.0101 | 1.0897 | 81,280.9 | 2,720.4 | 302.4 | 14,436,348 |
| `alibi-hybrid-h6|seed2` | 1.0364 | 1.0149 | 1.0881 | 81,299.3 | 2,720.4 | 302.3 | 14,436,348 |
| `alibi-pure|seed0` | 1.0314 | 1.0066 | 1.0854 | 86,577.3 | 2,152.5 | 283.9 | 14,260,608 |
| `alibi-pure|seed1` | 1.0530 | 1.0253 | 1.0973 | 86,635.7 | 2,152.5 | 283.7 | 14,260,608 |
| `alibi-pure|seed2` | 1.0377 | 1.0080 | 1.0836 | 86,543.3 | 2,152.5 | 284.0 | 14,260,608 |
