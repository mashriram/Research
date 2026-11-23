# Equal wall-clock and equal-steps control (13,300 steps, 3 seeds)

**Status:** current  ·  **Paper:** Table 4 (13.3k rows)  ·  **Script(s):** [`gpu_chinchilla_15m.py`](../../../benchmarks/gpu_chinchilla_15m.py)

## Summary

Folded-kernel Pure ALiBi trains 1.353x faster than DEEPA (117.0k vs 86.5k tok/s), so its 18,000-step budget buys DEEPA 13,300 steps. The Pure ALiBi 13,300-step rows are the equal-steps control for the over-repetition hypothesis (both models score better at 13.3k than at 18k).

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
KINDS=alibi-hybrid-h6 python benchmarks/gpu_chinchilla_15m.py 13300 0,1,2   # DEEPA, equal wall-clock
KINDS=alibi-pure python benchmarks/gpu_chinchilla_15m.py 13300 0,1,2        # Pure ALiBi, equal steps
```

## Files

| file | kind | size |
|---|---|---|
| [`gpu_chinchilla_15m_13300.json`](gpu_chinchilla_15m_13300.json) | results (machine-readable) | 17.3 KB |
| [`gpu_isotime_13300.log`](gpu_isotime_13300.log) | stdout of the run | 10.4 KB |
| [`gpu_isostep_pure_13300.log`](gpu_isostep_pure_13300.log) | stdout of the run | 9.3 KB |

## Results: `gpu_chinchilla_15m_13300.json`

Auto-extracted per-run metrics (bits/byte lower is better; accuracies in [0,1]; `x` = not solved). Paper tables aggregate these over seeds.

| run | val_bpb | val_bpb_len2048 | val_bpb_len4096 | tok_s | peak_mem_mb | train_s | params |
|---|---|---|---|---|---|---|---|
| `alibi-hybrid-h6|seed0` | 0.7776 | 0.7579 | 0.8200 | 80,923.4 | 2,718.8 | 2,692.8 | 14,436,348 |
| `alibi-hybrid-h6|seed1` | 0.7756 | 0.7575 | 0.8153 | 80,922.1 | 2,721.2 | 2,692.8 | 14,436,348 |
| `alibi-hybrid-h6|seed2` | 0.7779 | 0.7602 | 0.8194 | 80,938.0 | 2,721.2 | 2,692.3 | 14,436,348 |
| `alibi-pure|seed0` | 0.7891 | 0.7699 | 0.8256 | 86,141.9 | 2,239.7 | 2,529.6 | 14,260,608 |
| `alibi-pure|seed1` | 0.7851 | 0.7659 | 0.8262 | 86,161.9 | 2,239.5 | 2,529.0 | 14,260,608 |
| `alibi-pure|seed2` | 0.7921 | 0.7706 | 0.8249 | 86,209.9 | 2,239.5 | 2,527.6 | 14,260,608 |
