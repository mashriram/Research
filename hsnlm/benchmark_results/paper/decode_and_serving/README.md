# CUDA-graph decoding latency and 8 GB serving capacity

**Status:** current  ·  **Paper:** Table 7b (tab:decode_serve), Fig. 4b  ·  **Script(s):** [`gpu_decode_fair.py`](../../../benchmarks/gpu_decode_fair.py)

## Summary

Every model uses a preallocated KV cache + CUDA graphs. DEEPA B=32 numbers use the dense-mask path, B=1 the folded kernel (best per regime). Includes forward-vs-step correctness checks.

## What it tests and how (from the script)

```text
B1 -- fair autoregressive-decode benchmark (Reviewer items 5, 8). Replaces the torch.cat-grown KV cache of
gpu_decode.py / gpu_decode_hybrid_recheck.py (which copies the whole cache every step and so penalises attention)
with what a real inference engine does:
  * attention layers (RoPE and ALiBi) use a PREALLOCATED bf16 KV cache [B,h,Tmax,dh]; one in-place write per step
  * SheafDelta layers carry their fixed state (matrix memory S [B,H,dh,dh] + 16-step dyadic buffer), updated in place
  * the whole one-token step (embed -> 8 blocks -> head -> greedy argmax) is captured as ONE CUDA graph and replayed,
    for all three models alike, so Python/kernel-launch overhead does not decide the comparison
  * models in bf16 (inference dtype); the 14.4M paper configs from deepa_common.build()
Measured at each (context T, batch B): ms per decode step (median of 100 graph replays), aggregate tok/s = B/step,
peak VRAM. Also: "serving capacity" = the largest batch (concurrent sequences) that fits in this 8 GB GPU at T, and its
throughput. The cache is filled with random values up to position T-1 (timing does not depend on cache contents);
correctness of the decode path is checked separately: step-by-step logits vs. the parallel forward() on real text.
```

## Reproduce

From the `hsnlm/` root (GPU runs need CUDA; see the top-level README for the environment):

```bash
python benchmarks/gpu_decode_fair.py
python benchmarks/gpu_decode_fair.py --flash-alibi
```

## Files

| file | kind | size |
|---|---|---|
| [`gpu_decode_fair.json`](gpu_decode_fair.json) | results (machine-readable) | 13.3 KB |
| [`gpu_decode_fair.log`](gpu_decode_fair.log) | stdout of the run | 14.8 KB |
| [`gpu_decode_fair_flashalibi.json`](gpu_decode_fair_flashalibi.json) | results (machine-readable) | 8.3 KB |
| [`gpu_decode_fair_flashalibi.log`](gpu_decode_fair_flashalibi.log) | stdout of the run | 8.0 KB |

## Results: `gpu_decode_fair.json`

Auto-extracted per-run metrics (bits/byte lower is better; accuracies in [0,1]; `x` = not solved). Paper tables aggregate these over seeds.

| run | ms_per_step | tok_s | max_batch | cache_mb | peak_mem_mb |
|---|---|---|---|---|---|
| `check|alibi-hybrid-h6` |  |  |  |  |  |
| `check|alibi-pure` |  |  |  |  |  |
| `check|rope-gpt` |  |  |  |  |  |
| `alibi-hybrid-h6|1024|B1|eager` | 4.0859 | 244.7 |  | 3.5144 | 68.2460 |
| `alibi-hybrid-h6|1024|B1|graph` | 0.9239 | 1,082.4 |  | 3.5144 | 68.2460 |
| `alibi-hybrid-h6|1024|B8|graph` | 1.2203 | 6,555.9 |  | 28.1149 | 95.0323 |
| `alibi-hybrid-h6|1024|B32|graph` | 2.1624 | 14,798.7 |  | 112.5 | 193.5 |
| `alibi-hybrid-h6|1024|B128|graph` | 6.5180 | 19,637.9 |  | 449.8 | 562.0 |
| `alibi-hybrid-h6|4096|B1|eager` | 4.1584 | 240.5 |  | 12.9516 | 110.8 |
| `alibi-hybrid-h6|4096|B1|graph` | 1.2318 | 811.8 |  | 12.9516 | 110.8 |
| `alibi-hybrid-h6|4096|B8|graph` | 1.7205 | 4,649.7 |  | 103.6 | 204.7 |
| `alibi-hybrid-h6|4096|B32|graph` | 4.1130 | 7,780.1 |  | 414.4 | 529.6 |
| `alibi-hybrid-h6|4096|B128|graph` | 14.2660 | 8,972.4 |  | 1,657.8 | 1,804.1 |
| `alibi-hybrid-h6|16384|B1|eager` | 4.9051 | 203.9 |  | 50.7003 | 168.1 |
| `alibi-hybrid-h6|16384|B1|graph` | 2.5418 | 393.4 |  | 50.7003 | 176.7 |
| `alibi-hybrid-h6|16384|B8|graph` | 4.4569 | 1,795.0 |  | 405.6 | 540.9 |
| `alibi-hybrid-h6|16384|B32|graph` | 12.7073 | 2,518.2 |  | 1,622.4 | 1,771.8 |
| `alibi-hybrid-h6|16384|B128|graph` | 47.9577 | 2,669.0 |  | 6,489.6 | 6,670.2 |
| `alibi-hybrid-h6|65536|B1|eager` | 9.3700 | 106.7 |  | 201.7 | 355.0 |
| `alibi-hybrid-h6|65536|B1|graph` | 7.4637 | 134.0 |  | 201.7 | 363.5 |
| `alibi-hybrid-h6|65536|B8|graph` | 13.6215 | 587.3 |  | 1,613.6 | 1,784.3 |
| `alibi-hybrid-h6|65536|B32|graph` | 47.4730 | 674.1 |  | 6,454.2 | 6,638.5 |
| `alibi-hybrid-h6|65536|B128|graph` |  |  |  |  |  |
| `alibi-pure|1024|B1|eager` | 1.3316 | 751.0 |  | 12.5829 | 202.7 |
| `alibi-pure|1024|B1|graph` | 0.7839 | 1,275.7 |  | 12.5829 | 202.7 |
| `alibi-pure|1024|B8|graph` | 1.1948 | 6,695.7 |  | 100.7 | 292.5 |
| `alibi-pure|1024|B32|graph` | 3.3341 | 9,597.8 |  | 402.7 | 603.3 |
| `alibi-pure|1024|B128|graph` | 11.7228 | 10,918.9 |  | 1,610.6 | 1,821.0 |
| `alibi-pure|4096|B1|eager` | 2.3414 | 427.1 |  | 50.3316 | 259.5 |
| `alibi-pure|4096|B1|graph` | 2.1287 | 469.8 |  | 50.3316 | 268.0 |
| `alibi-pure|4096|B8|graph` | 3.5556 | 2,250.0 |  | 402.7 | 628.7 |
| `alibi-pure|4096|B32|graph` | 12.1790 | 2,627.5 |  | 1,610.6 | 1,845.4 |
| `alibi-pure|4096|B128|graph` | 45.8753 | 2,790.2 |  | 6,442.5 | 6,686.9 |
| `alibi-pure|16384|B1|eager` | 7.4123 | 134.9 |  | 201.3 | 445.0 |
| `alibi-pure|16384|B1|graph` | 7.1957 | 139.0 |  | 201.3 | 453.5 |
| `alibi-pure|16384|B8|graph` | 12.9217 | 619.1 |  | 1,610.6 | 1,871.3 |
| `alibi-pure|16384|B32|graph` | 46.7481 | 684.5 |  | 6,442.5 | 6,711.8 |
| `alibi-pure|16384|B128|graph` |  |  |  |  |  |
| `alibi-pure|65536|B1|eager` | 27.6445 | 36.1735 |  | 805.3 | 1,076.9 |
| `alibi-pure|65536|B1|graph` | 27.3590 | 36.5511 |  | 805.3 | 1,085.4 |
| `alibi-pure|65536|B8|graph` | 51.4454 | 155.5 |  | 6,442.5 | 6,731.1 |
| `alibi-pure|65536|B32|graph` |  |  |  |  |  |
| `alibi-pure|65536|B128|graph` |  |  |  |  |  |
| `rope-gpt|1024|B1|eager` | 3.5395 | 282.5 |  | 12.5829 | 313.4 |
| `rope-gpt|1024|B1|graph` | 0.8920 | 1,121.0 |  | 12.5829 | 313.4 |
| `rope-gpt|1024|B8|graph` | 1.7504 | 4,570.4 |  | 100.7 | 403.3 |
| `rope-gpt|1024|B32|graph` | 4.2259 | 7,572.4 |  | 402.7 | 714.1 |
| `rope-gpt|1024|B128|graph` | 14.4694 | 8,846.3 |  | 1,610.6 | 1,923.4 |
| `rope-gpt|4096|B1|eager` | 3.5165 | 284.4 |  | 50.3316 | 361.6 |
| `rope-gpt|4096|B1|graph` | 1.0878 | 919.3 |  | 50.3316 | 361.6 |
| `rope-gpt|4096|B8|graph` | 4.3312 | 1,847.0 |  | 402.7 | 713.8 |
| `rope-gpt|4096|B32|graph` | 14.3361 | 2,232.1 |  | 1,610.6 | 1,922.1 |
| `rope-gpt|4096|B128|graph` | 55.8458 | 2,292.0 |  | 6,442.5 | 6,755.2 |
| `rope-gpt|16384|B1|eager` | 3.5525 | 281.5 |  | 201.3 | 512.4 |
| `rope-gpt|16384|B1|graph` | 1.9138 | 522.5 |  | 201.3 | 512.4 |
| `rope-gpt|16384|B8|graph` | 15.0950 | 530.0 |  | 1,610.6 | 1,921.7 |
| `rope-gpt|16384|B32|graph` | 56.0455 | 571.0 |  | 6,442.5 | 6,753.9 |
| `rope-gpt|16384|B128|graph` |  |  |  |  |  |
| `rope-gpt|65536|B1|eager` | 5.6688 | 176.4 |  | 805.3 | 1,116.3 |
| `rope-gpt|65536|B1|graph` | 5.5140 | 181.4 |  | 805.3 | 1,116.3 |
| `rope-gpt|65536|B8|graph` | 58.1152 | 137.7 |  | 6,442.5 | 6,753.6 |
| `rope-gpt|65536|B32|graph` |  |  |  |  |  |
| `rope-gpt|65536|B128|graph` |  |  |  |  |  |
| `capacity|alibi-hybrid-h6|4096` | 67.4711 | 8,537.0 | 576 | 7,460.1 | 7,728.9 |
| `capacity|alibi-hybrid-h6|16384` | 54.2830 | 2,652.8 | 144 | 7,300.8 | 7,549.8 |
| `capacity|alibi-hybrid-h6|65536` | 52.3064 | 688.3 | 36 | 7,261.0 | 7,546.5 |
| `capacity|alibi-pure|4096` | 51.7717 | 2,781.4 | 144 | 7,247.8 | 7,553.5 |
| `capacity|alibi-pure|16384` | 51.4243 | 700.1 | 36 | 7,247.8 | 7,552.6 |
| `capacity|alibi-pure|65536` | 55.2246 | 163.0 | 9 | 7,247.8 | 7,554.8 |
| `capacity|rope-gpt|4096` | 63.0806 | 2,282.8 | 144 | 7,247.8 | 7,553.7 |
| `capacity|rope-gpt|16384` | 62.2997 | 577.9 | 36 | 7,247.8 | 7,552.2 |
| `capacity|rope-gpt|65536` | 62.4681 | 144.1 | 9 | 7,247.8 | 7,551.8 |

## Results: `gpu_decode_fair_flashalibi.json`

Auto-extracted per-run metrics (bits/byte lower is better; accuracies in [0,1]; `x` = not solved). Paper tables aggregate these over seeds.

| run | ms_per_step | tok_s | max_batch | cache_mb | peak_mem_mb |
|---|---|---|---|---|---|
| `check|alibi-hybrid-h6` |  |  |  |  |  |
| `check|alibi-pure` |  |  |  |  |  |
| `alibi-hybrid-h6|1024|B1|eager` | 4.5377 | 220.4 |  | 4.0387 | 68.2465 |
| `alibi-hybrid-h6|1024|B1|graph` | 0.8999 | 1,111.3 |  | 4.0387 | 68.2465 |
| `alibi-hybrid-h6|1024|B8|graph` | 1.3042 | 6,133.9 |  | 32.3092 | 99.5686 |
| `alibi-hybrid-h6|1024|B32|graph` | 2.4807 | 12,899.6 |  | 129.2 | 210.3 |
| `alibi-hybrid-h6|1024|B128|graph` | 6.9603 | 18,390.1 |  | 516.9 | 629.5 |
| `alibi-hybrid-h6|4096|B1|eager` | 4.7865 | 208.9 |  | 15.0487 | 110.8 |
| `alibi-hybrid-h6|4096|B1|graph` | 0.9497 | 1,053.0 |  | 15.0487 | 110.8 |
| `alibi-hybrid-h6|4096|B8|graph` | 1.9988 | 4,002.3 |  | 120.4 | 221.4 |
| `alibi-hybrid-h6|4096|B32|graph` | 5.1020 | 6,272.0 |  | 481.6 | 596.7 |
| `alibi-hybrid-h6|4096|B128|graph` | 18.2155 | 7,027.0 |  | 1,926.2 | 2,072.8 |
| `alibi-hybrid-h6|16384|B1|eager` | 4.6424 | 215.4 |  | 59.0889 | 176.6 |
| `alibi-hybrid-h6|16384|B1|graph` | 1.2156 | 822.7 |  | 59.0889 | 185.1 |
| `alibi-hybrid-h6|16384|B8|graph` | 5.5068 | 1,452.7 |  | 472.7 | 607.8 |
| `alibi-hybrid-h6|16384|B32|graph` | 16.2541 | 1,968.7 |  | 1,890.8 | 2,040.1 |
| `alibi-hybrid-h6|16384|B128|graph` |  |  |  |  |  |
| `alibi-hybrid-h6|65536|B1|eager` | 5.1321 | 194.9 |  | 235.2 | 378.0 |
| `alibi-hybrid-h6|65536|B1|graph` | 2.1913 | 456.3 |  | 235.2 | 386.5 |
| `alibi-hybrid-h6|65536|B8|graph` | 16.5007 | 484.8 |  | 1,882.0 | 2,042.6 |
| `alibi-hybrid-h6|65536|B32|graph` |  |  |  |  |  |
| `alibi-hybrid-h6|65536|B128|graph` |  |  |  |  |  |
| `alibi-pure|1024|B1|eager` | 2.5972 | 385.0 |  | 14.6801 | 185.6 |
| `alibi-pure|1024|B1|graph` | 0.6837 | 1,462.6 |  | 14.6801 | 185.6 |
| `alibi-pure|1024|B8|graph` | 1.6689 | 4,793.7 |  | 117.4 | 292.3 |
| `alibi-pure|1024|B32|graph` | 4.3833 | 7,300.4 |  | 469.8 | 653.5 |
| `alibi-pure|1024|B128|graph` | 15.7614 | 8,121.1 |  | 1,879.0 | 2,072.7 |
| `alibi-pure|4096|B1|eager` | 2.6668 | 375.0 |  | 58.7203 | 250.5 |
| `alibi-pure|4096|B1|graph` | 0.9472 | 1,055.8 |  | 58.7203 | 259.0 |
| `alibi-pure|4096|B8|graph` | 4.4629 | 1,792.6 |  | 469.8 | 678.7 |
| `alibi-pure|4096|B32|graph` | 15.6724 | 2,041.8 |  | 1,879.0 | 2,096.8 |
| `alibi-pure|4096|B128|graph` |  |  |  |  |  |
| `alibi-pure|16384|B1|eager` | 2.7791 | 359.8 |  | 234.9 | 452.2 |
| `alibi-pure|16384|B1|graph` | 1.9139 | 522.5 |  | 234.9 | 460.7 |
| `alibi-pure|16384|B8|graph` | 15.9789 | 500.7 |  | 1,879.0 | 2,113.5 |
| `alibi-pure|16384|B32|graph` |  |  |  |  |  |
| `alibi-pure|16384|B128|graph` |  |  |  |  |  |
| `alibi-pure|65536|B1|eager` | 5.9116 | 169.2 |  | 939.5 | 1,173.9 |
| `alibi-pure|65536|B1|graph` | 6.0087 | 166.4 |  | 939.5 | 1,182.4 |
| `alibi-pure|65536|B8|graph` |  |  |  |  |  |
| `alibi-pure|65536|B32|graph` |  |  |  |  |  |
| `alibi-pure|65536|B128|graph` |  |  |  |  |  |
| `capacity|alibi-hybrid-h6|4096` | 65.5258 | 6,837.0 | 448 | 6,741.8 | 7,163.2 |
| `capacity|alibi-hybrid-h6|16384` | 56.4960 | 2,017.8 | 114 | 6,736.1 | 7,075.3 |
| `capacity|alibi-hybrid-h6|65536` | 56.1997 | 516.0 | 29 | 6,822.2 | 7,140.6 |
| `capacity|alibi-pure|4096` | 56.1495 | 2,065.9 | 116 | 6,811.5 | 7,124.3 |
| `capacity|alibi-pure|16384` | 56.0160 | 517.7 | 29 | 6,811.5 | 7,123.0 |
| `capacity|alibi-pure|65536` | 58.3597 | 119.9 | 7 | 6,576.7 | 6,887.8 |
