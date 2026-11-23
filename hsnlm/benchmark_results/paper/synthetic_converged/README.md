# Converged synthetic suite (MQAR, parity, S3, S5, copy, selective copy)

**Status:** current  ·  **Paper:** Table 10 (tab:synth)  ·  **Script(s):** [`gpu_synthetic_proper.py`](../../../benchmarks/gpu_synthetic_proper.py)

## Summary

Up to 20k steps, early stop at >=99% (two consecutive evals). dim 128 / 4 layers except copy_paper (paper model). grid|... keys = LR-robustness pass so the LR can be chosen by 3-seed mean for every model.

## What it tests and how (from the script)

```text
B2-proper -- synthetic capability suite, run to convergence (replaces the 3,000-step screening in gpu_synthetic_suite.py,
whose MQAR / S5 numbers were under-trained: Zoology-style MQAR needs O(10^5) steps, and S5 is normally learned with a
length curriculum).

Protocol (identical for every architecture):
  * models: dim 128, 4 layers, attn heads 4, mixer heads 2 (same as the screening suite), vocab 256, bf16, torch.compile.
  * budget: up to MAX_STEPS (default 20,000) steps, B=64, AdamW(0.9,0.95) wd 0.05, warmup 200 + cosine over MAX_STEPS.
  * LR grid {3e-4, 1e-3, 3e-3} on seed 0; the best LR (final acc, then fewest steps-to-solve) is rerun on seeds 1 and 2.
  * eval every 500 steps on a fixed held-out set at the training length (1,024 sequences); "solved" = acc >= 0.99;
    runs stop early after two consecutive solved evals (so cost tracks difficulty). We log the whole accuracy curve and
    steps_to_solve (sample efficiency), then a final held-out eval (2,048 sequences) at every test length.
  * state-tracking tasks use a length curriculum 8 -> 16 -> 32 -> 64 (advance when the current length is solved);
    a run that never reaches length 64 is reported with the stage it reached.
Tasks:
  mqar8_T64    sanity / easy recall: 8 pairs in 64 tokens.          mqar32_T512, mqar64_T512: Zoology-scale recall.
  parity       Z2 running parity, train 64, test 64 / 256.
  s3           S3 (6 elements, non-abelian, solvable) running product, train 64, test 64 / 256.
  s5           S5 (120 elements, non-solvable, NC1-complete) running product, train 64, test 64 / 256.
  copy_paper   copy again, at the paper model size (dim 384, 8 layers, 2 attention layers in the hybrid).
  copy         copy a random 128-token string after a separator, test 128 / 512.  selcopy: selective copying, 16 of 256.
Usage: python benchmarks/gpu_synthetic_proper.py [task ...]      env: PROPER_MAX_STEPS, PROPER_SMOKE=1
```

## Reproduce

From the `hsnlm/` root (GPU runs need CUDA; see the top-level README for the environment):

```bash
python benchmarks/gpu_synthetic_proper.py                                          # LR sweep on seed 0 + 3 seeds
PROPER_FULLGRID=1 python benchmarks/gpu_synthetic_proper.py copy_paper copy mqar64_T512  # every LR x every seed
```

## Files

| file | kind | size |
|---|---|---|
| [`gpu_synthetic_proper.json`](gpu_synthetic_proper.json) | results (machine-readable) | 373.1 KB |
| [`gpu_synthetic_proper.log`](gpu_synthetic_proper.log) | stdout of the run | 30.4 KB |
| [`gpu_synthetic_grid.log`](gpu_synthetic_grid.log) | stdout of the run | 8.7 KB |

## Results: `gpu_synthetic_proper.json`

Auto-extracted per-run metrics (bits/byte lower is better; accuracies in [0,1]; `x` = not solved). Paper tables aggregate these over seeds.

| run | acc_T64 | acc_T128 | acc_T256 | acc_T512 | steps_to_solve | reached_len | train_s |
|---|---|---|---|---|---|---|---|
| `sweep|copy_paper|alibi-hybrid-h6|0.0003` |  | 0.9983 |  | 0.2455 | 2,000 | 128 | 477.2 |
| `sweep|copy_paper|alibi-hybrid-h6|0.001` |  | 0.9997 |  | 0.9977 | 2,000 | 128 | 468.1 |
| `sweep|copy_paper|alibi-hybrid-h6|0.003` |  | 0.9996 |  | 0.9964 | 2,000 | 128 | 467.6 |
| `copy_paper|alibi-hybrid-h6|seed0` |  | 0.9997 |  | 0.9977 | 2,000 | 128 | 468.1 |
| `copy_paper|alibi-hybrid-h6|seed1` |  | 0.9998 |  | 0.9809 | 2,000 | 128 | 467.3 |
| `copy_paper|alibi-hybrid-h6|seed2` |  | 0.9995 |  | 0.3847 | 4,000 | 128 | 839.2 |
| `sweep|copy_paper|alibi-pure|0.0003` |  | 0.9990 |  | 0.9605 | 500 | 128 | 127.8 |
| `sweep|copy_paper|alibi-pure|0.001` |  | 0.9978 |  | 0.9974 | 500 | 128 | 127.5 |
| `sweep|copy_paper|alibi-pure|0.003` |  | 0.9980 |  | 0.9937 | 1,000 | 128 | 189.6 |
| `copy_paper|alibi-pure|seed0` |  | 0.9990 |  | 0.9605 | 500 | 128 | 127.8 |
| `copy_paper|alibi-pure|seed1` |  | 0.9986 |  | 0.9518 | 500 | 128 | 126.9 |
| `copy_paper|alibi-pure|seed2` |  | 0.9986 |  | 0.8735 | 500 | 128 | 127.3 |
| `sweep|copy_paper|rope-gpt|0.0003` |  | 1.0000 |  | 0.0039 | 500 | 128 | 154.5 |
| `sweep|copy_paper|rope-gpt|0.001` |  | 1.0000 |  | 0.0039 | 500 | 128 | 154.5 |
| `sweep|copy_paper|rope-gpt|0.003` |  | 1.0000 |  | 0.0040 | 500 | 128 | 154.9 |
| `copy_paper|rope-gpt|seed0` |  | 1.0000 |  | 0.0039 | 500 | 128 | 154.5 |
| `copy_paper|rope-gpt|seed1` |  | 1.0000 |  | 0.0040 | 500 | 128 | 154.6 |
| `copy_paper|rope-gpt|seed2` |  | 1.0000 |  | 0.0039 | 500 | 128 | 154.4 |
| `sweep|copy_paper|sheafdelta-pure|0.0003` |  | 0.9961 |  | 0.1366 | 2,500 | 128 | 620.9 |
| `sweep|copy_paper|sheafdelta-pure|0.001` |  | 0.9949 |  | 0.1593 | 5,500 | 128 | 1,239.4 |
| `sweep|copy_paper|sheafdelta-pure|0.003` |  | 0.9958 |  | 0.1808 | 3,500 | 128 | 828.3 |
| `copy_paper|sheafdelta-pure|seed0` |  | 0.9961 |  | 0.1366 | 2,500 | 128 | 620.9 |
| `copy_paper|sheafdelta-pure|seed1` |  | 0.9973 |  | 0.1759 | 2,500 | 128 | 621.0 |
| `copy_paper|sheafdelta-pure|seed2` |  | 0.9945 |  | 0.1492 | 3,500 | 128 | 826.9 |
| `sweep|copy|alibi-hybrid-h6|0.0003` |  | 0.9981 |  | 0.3164 | 3,500 | 128 | 99.7741 |
| `sweep|copy|alibi-hybrid-h6|0.001` |  | 0.9952 |  | 0.3672 | 4,500 | 128 | 112.9 |
| `sweep|copy|alibi-hybrid-h6|0.003` |  | 0.9992 |  | 0.9597 | 6,000 | 128 | 146.1 |
| `copy|alibi-hybrid-h6|seed0` |  | 0.9992 |  | 0.9597 | 6,000 | 128 | 146.1 |
| `copy|alibi-hybrid-h6|seed1` |  | 0.0151 |  | 0.0044 | x | 128 | 440.5 |
| `copy|alibi-hybrid-h6|seed2` |  | 0.9978 |  | 0.5870 | 6,000 | 128 | 146.2 |
| `sweep|copy|alibi-pure|0.0003` |  | 0.9986 |  | 0.9881 | 1,000 | 128 | 24.1800 |
| `sweep|copy|alibi-pure|0.001` |  | 0.9962 |  | 0.9789 | 500 | 128 | 17.3470 |
| `sweep|copy|alibi-pure|0.003` |  | 0.9974 |  | 0.9546 | 2,000 | 128 | 39.3194 |
| `copy|alibi-pure|seed0` |  | 0.9986 |  | 0.9881 | 1,000 | 128 | 24.1800 |
| `copy|alibi-pure|seed1` |  | 0.9982 |  | 0.9811 | 1,000 | 128 | 24.9465 |
| `copy|alibi-pure|seed2` |  | 0.9987 |  | 0.9858 | 1,000 | 128 | 24.1917 |
| `sweep|copy|rope-gpt|0.0003` |  | 1.0000 |  | 0.0039 | 500 | 128 | 19.1688 |
| `sweep|copy|rope-gpt|0.001` |  | 1.0000 |  | 0.0039 | 500 | 128 | 19.8642 |
| `sweep|copy|rope-gpt|0.003` |  | 1.0000 |  | 0.0040 | 500 | 128 | 19.0804 |
| `copy|rope-gpt|seed0` |  | 1.0000 |  | 0.0039 | 500 | 128 | 19.1688 |
| `copy|rope-gpt|seed1` |  | 1.0000 |  | 0.0040 | 500 | 128 | 19.8944 |
| `copy|rope-gpt|seed2` |  | 1.0000 |  | 0.0039 | 500 | 128 | 19.1303 |
| `sweep|copy|sheafdelta-pure|0.0003` |  | 0.9913 |  | 0.2250 | 18,000 | 128 | 444.7 |
| `sweep|copy|sheafdelta-pure|0.001` |  | 0.9930 |  | 0.2411 | 5,500 | 128 | 145.7 |
| `sweep|copy|sheafdelta-pure|0.003` |  | 0.0040 |  | 0.0039 | x | 128 | 480.7 |
| `copy|sheafdelta-pure|seed0` |  | 0.9930 |  | 0.2411 | 5,500 | 128 | 145.7 |
| `copy|sheafdelta-pure|seed1` |  | 0.9945 |  | 0.2592 | 4,500 | 128 | 121.9 |
| `copy|sheafdelta-pure|seed2` |  | 0.9921 |  | 0.1908 | 4,000 | 128 | 109.7 |
| `sweep|mqar8_T64|alibi-hybrid-h6|0.0003` | 0.9997 |  |  |  | 3,500 | 64 | 51.3846 |
| `sweep|mqar8_T64|alibi-hybrid-h6|0.001` | 0.9993 |  |  |  | 1,000 | 64 | 20.4793 |
| `sweep|mqar8_T64|alibi-hybrid-h6|0.003` | 0.9984 |  |  |  | 1,000 | 64 | 21.6901 |
| `mqar8_T64|alibi-hybrid-h6|seed0` | 0.9997 |  |  |  | 3,500 | 64 | 51.3846 |
| `mqar8_T64|alibi-hybrid-h6|seed1` | 0.9988 |  |  |  | 1,500 | 64 | 25.8373 |
| `mqar8_T64|alibi-hybrid-h6|seed2` | 0.9997 |  |  |  | 1,500 | 64 | 28.4868 |
| `sweep|mqar8_T64|alibi-pure|0.0003` | 0.9944 |  |  |  | 7,000 | 64 | 29.8416 |
| `sweep|mqar8_T64|alibi-pure|0.001` | 0.9938 |  |  |  | 6,000 | 64 | 33.4624 |
| `sweep|mqar8_T64|alibi-pure|0.003` | 0.2753 |  |  |  | x | 64 | 76.2275 |
| `mqar8_T64|alibi-pure|seed0` | 0.9944 |  |  |  | 7,000 | 64 | 29.8416 |
| `mqar8_T64|alibi-pure|seed1` | 0.9933 |  |  |  | 6,000 | 64 | 30.7188 |
| `mqar8_T64|alibi-pure|seed2` | 0.9953 |  |  |  | 9,000 | 64 | 40.9802 |
| `sweep|mqar8_T64|rope-gpt|0.0003` | 0.9918 |  |  |  | 9,000 | 64 | 42.8555 |
| `sweep|mqar8_T64|rope-gpt|0.001` | 0.1707 |  |  |  | x | 64 | 88.8603 |
| `sweep|mqar8_T64|rope-gpt|0.003` | 0.1713 |  |  |  | x | 64 | 88.0279 |
| `mqar8_T64|rope-gpt|seed0` | 0.9918 |  |  |  | 9,000 | 64 | 42.8555 |
| `mqar8_T64|rope-gpt|seed1` | 0.9972 |  |  |  | 9,500 | 64 | 53.7136 |
| `mqar8_T64|rope-gpt|seed2` | 0.9922 |  |  |  | 9,000 | 64 | 42.8299 |
| `sweep|mqar8_T64|sheafdelta-pure|0.0003` | 0.9931 |  |  |  | 13,500 | 64 | 199.8 |
| `sweep|mqar8_T64|sheafdelta-pure|0.001` | 0.9909 |  |  |  | 13,500 | 64 | 194.0 |
| `sweep|mqar8_T64|sheafdelta-pure|0.003` | 0.9925 |  |  |  | 11,000 | 64 | 156.3 |
| `mqar8_T64|sheafdelta-pure|seed0` | 0.9931 |  |  |  | 13,500 | 64 | 199.8 |
| `mqar8_T64|sheafdelta-pure|seed1` | 0.9946 |  |  |  | 5,000 | 64 | 81.8945 |
| `mqar8_T64|sheafdelta-pure|seed2` | 0.9951 |  |  |  | 2,500 | 64 | 43.5670 |
| `sweep|mqar32_T512|alibi-hybrid-h6|0.0003` |  |  |  | 0.9999 | 2,500 | 512 | 166.9 |
| `sweep|mqar32_T512|alibi-hybrid-h6|0.001` |  |  |  | 0.9995 | 3,000 | 512 | 183.1 |
| `sweep|mqar32_T512|alibi-hybrid-h6|0.003` |  |  |  | 0.9998 | 5,500 | 512 | 311.9 |
| `mqar32_T512|alibi-hybrid-h6|seed0` |  |  |  | 0.9999 | 2,500 | 512 | 166.9 |
| `mqar32_T512|alibi-hybrid-h6|seed1` |  |  |  | 0.9997 | 2,500 | 512 | 158.6 |
| `mqar32_T512|alibi-hybrid-h6|seed2` |  |  |  | 0.9976 | 2,500 | 512 | 157.4 |
| `sweep|mqar32_T512|alibi-pure|0.0003` |  |  |  | 0.9950 | 2,500 | 512 | 108.5 |
| `sweep|mqar32_T512|alibi-pure|0.001` |  |  |  | 0.9986 | 4,000 | 512 | 156.8 |
| `sweep|mqar32_T512|alibi-pure|0.003` |  |  |  | 0.9975 | 15,500 | 512 | 555.7 |
| `mqar32_T512|alibi-pure|seed0` |  |  |  | 0.9986 | 4,000 | 512 | 156.8 |
| `mqar32_T512|alibi-pure|seed1` |  |  |  | 0.9956 | 6,500 | 512 | 277.5 |
| `mqar32_T512|alibi-pure|seed2` |  |  |  | 0.9960 | 3,000 | 512 | 122.3 |
| `sweep|mqar32_T512|rope-gpt|0.0003` |  |  |  | 0.0847 | x | 512 | 782.2 |
| `sweep|mqar32_T512|rope-gpt|0.001` |  |  |  | 0.0845 | x | 512 | 780.9 |
| `sweep|mqar32_T512|rope-gpt|0.003` |  |  |  | 0.0856 | x | 512 | 780.8 |
| `mqar32_T512|rope-gpt|seed0` |  |  |  | 0.0856 | x | 512 | 780.8 |
| `mqar32_T512|rope-gpt|seed1` |  |  |  | 0.0851 | x | 512 | 780.2 |
| `mqar32_T512|rope-gpt|seed2` |  |  |  | 0.0849 | x | 512 | 778.9 |
| `sweep|mqar32_T512|sheafdelta-pure|0.0003` |  |  |  | 0.9948 | 2,500 | 512 | 171.8 |
| `sweep|mqar32_T512|sheafdelta-pure|0.001` |  |  |  | 0.9951 | 6,000 | 512 | 366.6 |
| `sweep|mqar32_T512|sheafdelta-pure|0.003` |  |  |  | 0.9930 | 4,500 | 512 | 283.1 |
| `mqar32_T512|sheafdelta-pure|seed0` |  |  |  | 0.9951 | 6,000 | 512 | 366.6 |
| `mqar32_T512|sheafdelta-pure|seed1` |  |  |  | 0.9928 | 4,500 | 512 | 282.8 |
| `mqar32_T512|sheafdelta-pure|seed2` |  |  |  | 0.9971 | 4,500 | 512 | 282.6 |
| `sweep|mqar64_T512|alibi-hybrid-h6|0.0003` |  |  |  | 0.8415 | x | 512 | 1,034.0 |
| `sweep|mqar64_T512|alibi-hybrid-h6|0.001` |  |  |  | 0.9968 | 12,500 | 512 | 672.3 |
| `sweep|mqar64_T512|alibi-hybrid-h6|0.003` |  |  |  | 1.0000 | 4,500 | 512 | 260.5 |
| `mqar64_T512|alibi-hybrid-h6|seed0` |  |  |  | 1.0000 | 4,500 | 512 | 260.5 |
| `mqar64_T512|alibi-hybrid-h6|seed1` |  |  |  | 0.9966 | 6,000 | 512 | 339.5 |
| `mqar64_T512|alibi-hybrid-h6|seed2` |  |  |  | 0.0619 | x | 512 | 1,033.9 |
| `sweep|mqar64_T512|alibi-pure|0.0003` |  |  |  | 0.9923 | 3,000 | 512 | 122.8 |
| `sweep|mqar64_T512|alibi-pure|0.001` |  |  |  | 0.9958 | 9,000 | 512 | 330.4 |
| `sweep|mqar64_T512|alibi-pure|0.003` |  |  |  | 0.8653 | x | 512 | 694.3 |
| `mqar64_T512|alibi-pure|seed0` |  |  |  | 0.9958 | 9,000 | 512 | 330.4 |
| `mqar64_T512|alibi-pure|seed1` |  |  |  | 0.0623 | x | 512 | 695.3 |
| `mqar64_T512|alibi-pure|seed2` |  |  |  | 0.9984 | 3,500 | 512 | 139.9 |
| `sweep|mqar64_T512|rope-gpt|0.0003` |  |  |  | 0.0620 | x | 512 | 780.5 |
| `sweep|mqar64_T512|rope-gpt|0.001` |  |  |  | 0.0623 | x | 512 | 781.6 |
| `sweep|mqar64_T512|rope-gpt|0.003` |  |  |  | 0.0627 | x | 512 | 780.8 |
| `mqar64_T512|rope-gpt|seed0` |  |  |  | 0.0627 | x | 512 | 780.8 |
| `mqar64_T512|rope-gpt|seed1` |  |  |  | 0.0621 | x | 512 | 782.0 |
| `mqar64_T512|rope-gpt|seed2` |  |  |  | 0.0616 | x | 512 | 782.7 |
| `sweep|mqar64_T512|sheafdelta-pure|0.0003` |  |  |  | 0.9987 | 10,000 | 512 | 591.5 |
| `sweep|mqar64_T512|sheafdelta-pure|0.001` |  |  |  | 0.9933 | 11,500 | 512 | 676.0 |
| `sweep|mqar64_T512|sheafdelta-pure|0.003` |  |  |  | 0.9051 | x | 512 | 1,124.6 |
| `mqar64_T512|sheafdelta-pure|seed0` |  |  |  | 0.9987 | 10,000 | 512 | 591.5 |
| `mqar64_T512|sheafdelta-pure|seed1` |  |  |  | 0.0621 | x | 512 | 1,126.2 |
| `mqar64_T512|sheafdelta-pure|seed2` |  |  |  | 0.9975 | 10,000 | 512 | 591.0 |
| ... 162 more rows in the JSON |  |  |  |  |  |  |  |
