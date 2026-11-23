# Chinchilla-horizon convergence on enwik8 (18,000 steps, 3 seeds)

**Status:** current  ·  **Paper:** Table 4 (tab:chinchilla)  ·  **Script(s):** [`gpu_chinchilla_15m.py`](../../../benchmarks/gpu_chinchilla_15m.py)

## Summary

DEEPA beats Pure ALiBi on 3/3 seeds (0.7794 vs 0.7863 bpb, 64 KB slice). 18,000 steps = 7.4 epochs over the 40 MB training pool (sampled with replacement). ALiBi here uses the dense-mask path (tok/s are pre-folding).

## What it tests and how (from the script)

```text
Closes the single most-repeated open caveat in this project: every 15m-scale quality number so
far (gpu_multiseed_compiled.py, gpu_v2_ablation.py) was measured at 1,500 steps = 24.6M sampled
bytes = ~1.73 "tokens"/param -- far below a Chinchilla-ish ~20 tokens/param, and squarely in the
"early-training advantage that might vanish with more training" risk zone the CPU report's own
threats-to-validity section warned about from the start. This reruns the h6 winner (ALiBi-hybrid
3:1, mixer_heads=6) and pure ALiBi at 18,000 steps instead of 1,500 -- ~295M sampled bytes, ~20.5
"tokens"/param -- same code path, same recipe, same uniform weight_decay=0.05 as every other number
in DEEPA_V2_ABLATION.md, so step count is the only variable changed. Note: enwik8's n_train=40M-byte
pool is fixed and sampled WITH replacement (hsnlm/data.py), so 18,000 steps is ~7.4 epochs over the
same slice, not 295M units of fresh unique data -- reported honestly as such, not oversold.
```

## Reproduce

From the `hsnlm/` root (GPU runs need CUDA; see the top-level README for the environment):

```bash
python benchmarks/gpu_chinchilla_15m.py 18000 0,1,2
```

## Files

| file | kind | size |
|---|---|---|
| [`gpu_chinchilla_15m_18000.json`](gpu_chinchilla_15m_18000.json) | results (machine-readable) | 17.4 KB |

## Results: `gpu_chinchilla_15m_18000.json`

Auto-extracted per-run metrics (bits/byte lower is better; accuracies in [0,1]; `x` = not solved). Paper tables aggregate these over seeds.

| run | val_bpb | val_bpb_len2048 | val_bpb_len4096 | tok_s | peak_mem_mb | train_s | params |
|---|---|---|---|---|---|---|---|
| `alibi-hybrid-h6|seed0` | 0.7820 | 0.7637 | 0.8276 | 80,686.3 | 2,719.9 | 3,655.0 | 14,436,348 |
| `alibi-pure|seed0` | 0.7924 | 0.7720 | 0.8307 | 85,625.6 | 2,152.5 | 3,444.2 | 14,260,608 |
| `alibi-hybrid-h6|seed1` | 0.7761 | 0.7590 | 0.8198 | 80,552.5 | 2,719.9 | 3,661.1 | 14,436,348 |
| `alibi-hybrid-h6|seed2` | 0.7801 | 0.7631 | 0.8264 | 80,413.0 | 2,720.3 | 3,667.5 | 14,436,348 |
| `alibi-pure|seed1` | 0.7861 | 0.7665 | 0.8260 | 85,199.1 | 2,152.5 | 3,461.4 | 14,260,608 |
| `alibi-pure|seed2` | 0.7803 | 0.7605 | 0.8206 | 85,205.4 | 2,152.5 | 3,461.2 | 14,260,608 |
