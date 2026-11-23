# Training step / prefill latency and memory vs length (1k-16k)

**Status:** current  ·  **Paper:** Table 7 (tab:train_speed), Fig. 4a, Table 3  ·  **Script(s):** [`gpu_overhead_profile.py`](../../../benchmarks/gpu_overhead_profile.py)

## Summary

The paper reports the faster implementation per cell; *_flashalibi files are the folded-kernel ALiBi runs.

## What it tests and how (from the script)

```text
Reviewer items 5 & 8 (training speed, memory, computational overhead): measured and analytical cost of the 14.4M
models (width 384) as the sequence length grows, ctx in {1024, 2048, 4096, 8192, 16384}.
Measured, per (kind, ctx), B=1, bf16 autocast, torch.compile, AdamW:
  train  : full training step (fwd + bwd + clip + AdamW) with per-block activation checkpointing -> ms/step, tok/s,
           peak VRAM.  Also the same step WITHOUT checkpointing where it fits (OOM recorded otherwise).
  prefill: no-grad forward over the whole sequence -> ms, tok/s, peak VRAM.
  3 warm-up steps, then the median of 10 timed steps (cuda-synchronized).
Analytical, per kind: parameter count/bytes, training-state bytes (fp32 params+grads+2 Adam moments = 16 B/param),
and the per-sequence inference cache at each ctx in bf16: attention layer = K and V, 2*T*D values; SheafDelta layer =
matrix memory H*dh*dh + dyadic ring buffer 16*D values, independent of T.
Note: the model architecture is held fixed at the paper's ctx=1024 config at every length (no ctx-dependent params).
ALiBi at T > 4096 uses the exact query-chunked path (deepa_common.py).
```

## Reproduce

From the `hsnlm/` root (GPU runs need CUDA; see the top-level README for the environment):

```bash
python benchmarks/gpu_overhead_profile.py                # dense-mask ALiBi
python benchmarks/gpu_overhead_profile.py --flash-alibi  # folded (FlashAttention) ALiBi
```

## Files

| file | kind | size |
|---|---|---|
| [`gpu_overhead_profile.json`](gpu_overhead_profile.json) | results (machine-readable) | 7.5 KB |
| [`gpu_overhead_profile.log`](gpu_overhead_profile.log) | stdout of the run | 36.7 KB |
| [`gpu_overhead_profile_flashalibi.json`](gpu_overhead_profile_flashalibi.json) | results (machine-readable) | 5.0 KB |
| [`gpu_overhead_profile_flashalibi.log`](gpu_overhead_profile_flashalibi.log) | stdout of the run | 4.3 KB |

## Results: `gpu_overhead_profile.json`

Auto-extracted per-run metrics (bits/byte lower is better; accuracies in [0,1]; `x` = not solved). Paper tables aggregate these over seeds.

| run | ms | tok_s | peak_mem_mb | params |
|---|---|---|---|---|
| `analytic|alibi-hybrid-h6` |  |  |  | 14,436,348 |
| `alibi-hybrid-h6|1024|train_ckpt` | 59.7819 | 17,128.9 | 316.5 |  |
| `alibi-hybrid-h6|1024|train_nockpt` | 22.7912 | 44,929.7 | 414.8 |  |
| `alibi-hybrid-h6|1024|prefill` | 4.9666 | 206,177.9 | 97.4689 |  |
| `alibi-hybrid-h6|2048|train_ckpt` | 78.9183 | 25,950.9 | 544.5 |  |
| `alibi-hybrid-h6|2048|train_nockpt` | 36.0531 | 56,805.1 | 669.0 |  |
| `alibi-hybrid-h6|2048|prefill` | 6.3636 | 321,828.0 | 153.3 |  |
| `alibi-hybrid-h6|4096|train_ckpt` | 205.2 | 19,965.5 | 1,444.5 |  |
| `alibi-hybrid-h6|4096|train_nockpt` | 74.6737 | 54,852.0 | 1,375.5 |  |
| `alibi-hybrid-h6|4096|prefill` | 22.3123 | 183,575.6 | 365.7 |  |
| `alibi-hybrid-h6|8192|train_ckpt` | 568.3 | 14,415.2 | 951.3 |  |
| `alibi-hybrid-h6|8192|train_nockpt` | 181.2 | 45,198.6 | 1,529.4 |  |
| `alibi-hybrid-h6|8192|prefill` | 52.6379 | 155,629.4 | 253.6 |  |
| `alibi-hybrid-h6|16384|train_ckpt` | 1,936.5 | 8,460.6 | 1,702.4 |  |
| `alibi-hybrid-h6|16384|train_nockpt` | 530.6 | 30,880.8 | 2,933.0 |  |
| `alibi-hybrid-h6|16384|prefill` | 136.8 | 119,728.9 | 432.0 |  |
| `analytic|alibi-pure` |  |  |  | 14,260,608 |
| `alibi-pure|1024|train_ckpt` | 25.6428 | 39,933.3 | 309.0 |  |
| `alibi-pure|1024|train_nockpt` | 21.1396 | 48,440.0 | 471.8 |  |
| `alibi-pure|1024|prefill` | 4.5133 | 226,886.8 | 105.3 |  |
| `alibi-pure|2048|train_ckpt` | 64.1106 | 31,944.8 | 495.4 |  |
| `alibi-pure|2048|train_nockpt` | 48.4808 | 42,243.5 | 995.5 |  |
| `alibi-pure|2048|prefill` | 15.2415 | 134,369.6 | 166.6 |  |
| `alibi-pure|4096|train_ckpt` | 179.1 | 22,870.2 | 1,331.2 |  |
| `alibi-pure|4096|train_nockpt` | 135.0 | 30,333.5 | 2,847.5 |  |
| `alibi-pure|4096|prefill` | 44.5914 | 91,856.4 | 390.0 |  |
| `alibi-pure|8192|train_ckpt` | 416.5 | 19,668.8 | 1,580.3 |  |
| `alibi-pure|8192|train_nockpt` | 404.4 | 20,256.8 | 2,068.0 |  |
| `alibi-pure|8192|prefill` | 102.5 | 79,886.4 | 776.3 |  |
| `alibi-pure|16384|train_ckpt` | 1,506.2 | 10,878.0 | 3,321.3 |  |
| `alibi-pure|16384|train_nockpt` | 1,475.9 | 11,100.8 | 4,594.6 |  |
| `alibi-pure|16384|prefill` | 366.2 | 44,737.6 | 2,498.2 |  |
| `analytic|rope-gpt` |  |  |  | 14,260,608 |
| `rope-gpt|1024|train_ckpt` | 18.6430 | 54,926.9 | 309.6 |  |
| `rope-gpt|1024|train_nockpt` | 18.2700 | 56,048.2 | 359.2 |  |
| `rope-gpt|1024|prefill` | 3.6099 | 283,662.4 | 92.7749 |  |
| `rope-gpt|2048|train_ckpt` | 32.3498 | 63,308.0 | 310.8 |  |
| `rope-gpt|2048|train_nockpt` | 31.1392 | 65,769.3 | 494.9 |  |
| `rope-gpt|2048|prefill` | 8.5652 | 239,107.8 | 107.8 |  |
| `rope-gpt|4096|train_ckpt` | 64.4013 | 63,601.2 | 385.2 |  |
| `rope-gpt|4096|train_nockpt` | 62.4875 | 65,549.1 | 774.7 |  |
| `rope-gpt|4096|prefill` | 18.7275 | 218,715.9 | 132.6 |  |
| `rope-gpt|8192|train_ckpt` | 171.5 | 47,766.0 | 540.8 |  |
| `rope-gpt|8192|train_nockpt` | 150.9 | 54,301.3 | 1,324.3 |  |
| `rope-gpt|8192|prefill` | 46.3979 | 176,559.8 | 186.3 |  |
| `rope-gpt|16384|train_ckpt` | 525.7 | 31,165.1 | 872.9 |  |
| `rope-gpt|16384|train_nockpt` | 440.4 | 37,200.0 | 2,408.6 |  |
| `rope-gpt|16384|prefill` | 135.0 | 121,361.9 | 293.7 |  |

## Results: `gpu_overhead_profile_flashalibi.json`

Auto-extracted per-run metrics (bits/byte lower is better; accuracies in [0,1]; `x` = not solved). Paper tables aggregate these over seeds.

| run | ms | tok_s | peak_mem_mb | params |
|---|---|---|---|---|
| `analytic|alibi-hybrid-h6` |  |  |  | 14,436,348 |
| `alibi-hybrid-h6|1024|train_ckpt` | 70.1209 | 14,603.3 | 315.7 |  |
| `alibi-hybrid-h6|1024|train_nockpt` | 25.2780 | 40,509.6 | 383.0 |  |
| `alibi-hybrid-h6|1024|prefill` | 5.1053 | 200,574.6 | 87.7455 |  |
| `alibi-hybrid-h6|2048|train_ckpt` | 70.1120 | 29,210.4 | 326.8 |  |
| `alibi-hybrid-h6|2048|train_nockpt` | 31.8171 | 64,367.9 | 543.1 |  |
| `alibi-hybrid-h6|2048|prefill` | 6.3173 | 324,187.7 | 100.4 |  |
| `alibi-hybrid-h6|4096|train_ckpt` | 100.5 | 40,759.7 | 403.8 |  |
| `alibi-hybrid-h6|4096|train_nockpt` | 54.0209 | 75,822.5 | 855.9 |  |
| `alibi-hybrid-h6|4096|prefill` | 14.4962 | 282,557.5 | 125.7 |  |
| `alibi-hybrid-h6|8192|train_ckpt` | 226.1 | 36,232.2 | 579.9 |  |
| `alibi-hybrid-h6|8192|train_nockpt` | 115.2 | 71,084.5 | 1,487.4 |  |
| `alibi-hybrid-h6|8192|prefill` | 31.2440 | 262,194.1 | 176.3 |  |
| `alibi-hybrid-h6|16384|train_ckpt` | 577.5 | 28,370.5 | 953.9 |  |
| `alibi-hybrid-h6|16384|train_nockpt` | 281.3 | 58,246.7 | 2,749.3 |  |
| `alibi-hybrid-h6|16384|prefill` | 76.6247 | 213,821.4 | 277.5 |  |
| `analytic|alibi-pure` |  |  |  | 14,260,608 |
| `alibi-pure|1024|train_ckpt` | 19.1442 | 53,488.8 | 307.2 |  |
| `alibi-pure|1024|train_nockpt` | 17.0435 | 60,081.6 | 349.4 |  |
| `alibi-pure|1024|prefill` | 2.8397 | 360,596.3 | 91.2364 |  |
| `alibi-pure|2048|train_ckpt` | 34.4371 | 59,470.7 | 320.4 |  |
| `alibi-pure|2048|train_nockpt` | 28.1125 | 72,850.1 | 482.6 |  |
| `alibi-pure|2048|prefill` | 5.7571 | 355,733.9 | 106.5 |  |
| `alibi-pure|4096|train_ckpt` | 66.4741 | 61,618.0 | 398.3 |  |
| `alibi-pure|4096|train_nockpt` | 57.3947 | 71,365.5 | 746.7 |  |
| `alibi-pure|4096|prefill` | 15.6371 | 261,941.7 | 136.5 |  |
| `alibi-pure|8192|train_ckpt` | 174.8 | 46,861.2 | 585.5 |  |
| `alibi-pure|8192|train_nockpt` | 137.9 | 59,407.9 | 1,263.2 |  |
| `alibi-pure|8192|prefill` | 42.8295 | 191,270.0 | 196.4 |  |
| `alibi-pure|16384|train_ckpt` | 530.4 | 30,890.6 | 961.1 |  |
| `alibi-pure|16384|train_nockpt` | 410.5 | 39,912.4 | 2,283.8 |  |
| `alibi-pure|16384|prefill` | 125.1 | 130,997.5 | 312.2 |  |
