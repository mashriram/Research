# Synthetic suite screening run (3,000 steps)

**Status:** superseded by synthetic_converged  ·  **Paper:** superseded  ·  **Script(s):** [`gpu_synthetic_suite.py`](../../../benchmarks/gpu_synthetic_suite.py)

## Summary

Under-trained: MQAR / S5 / copy numbers here are NOT valid conclusions; kept for transparency.

## What it tests and how (from the script)

```text
B2 -- synthetic capability suite: tests WHAT each architecture can do, independent of data/tokenisation.
Small models (dim 128, 4 layers, attn heads 4, mixer heads 2 -> mixer head dim 64), vocab 256, compiled-off eager, bf16.
Architectures: alibi-hybrid-h6 ("ddda": 3 SheafDelta + 1 ALiBi), alibi-pure, rope-gpt, sheafdelta-pure (no attention).
ALiBi layers use the exact fused-kernel path (deepa_common.ALIBI_IMPL="flash").
Tasks (loss/accuracy only on target positions):
  mqar{n}   multi-query associative recall (Zoology, Arora et al. 2023): n key-value pairs, then queries scattered among noise;
            predict each queried key's value. T=512. n in {32, 64}. Tests in-context recall / memory capacity.
  s5        S5 permutation word problem: input = sequence of permutations of 5 items, target = running composition.
            NC1-complete; provably beyond fixed-depth attention and diagonal SSMs, solvable by delta rules whose
            transition can have negative eigenvalues (Grazzi et al. 2024) -- our beta in [0,2]. Train T=64, test 64 & 256.
  parity    running parity of a bit string. Train T=64, test 64 & 256 (length generalisation).
  copy      induction/copy: random string of L tokens, separator, repeat it; predict the repeat. Train L=128, test L=128 & 512.
  selcopy   selective copying (Mamba, Gu & Dao 2023): 16 data tokens at random positions in T=256 of noise; after a marker,
            output them in order. Tests content-aware filtering.
Protocol: per (task, arch) LR picked from {1e-3, 3e-3} on seed 0, then 3 seeds (0,1,2) at that LR, 3,000 steps, B=64,
AdamW(0.9,0.95) wd 0.05 (wd 0.0 on nothing -- same uniform recipe as the paper), warmup 100 + cosine. Fresh random batches every
step (infinite data), fixed held-out eval batches (seeded) at each test length: 2,048 sequences.
```

## Reproduce

From the `hsnlm/` root (GPU runs need CUDA; see the top-level README for the environment):

```bash
python benchmarks/gpu_synthetic_suite.py
```

## Files

| file | kind | size |
|---|---|---|
| [`gpu_synthetic_suite.json`](gpu_synthetic_suite.json) | results (machine-readable) | 28.8 KB |
| [`gpu_synthetic_suite.log`](gpu_synthetic_suite.log) | stdout of the run | 10.7 KB |

## Results: `gpu_synthetic_suite.json`

Auto-extracted per-run metrics (bits/byte lower is better; accuracies in [0,1]; `x` = not solved). Paper tables aggregate these over seeds.

| run | acc_T64 | acc_T256 | acc_T512 | train_s | lr | params |
|---|---|---|---|---|---|---|
| `sweep|mqar32|alibi-hybrid-h6|0.001` |  |  | 0.9625 | 360.8 | 0.0010 | 832,490 |
| `sweep|mqar32|alibi-hybrid-h6|0.003` |  |  | 0.0770 | 331.9 | 0.0030 | 832,490 |
| `mqar32|alibi-hybrid-h6|seed0` |  |  | 0.9625 | 360.8 | 0.0010 | 832,490 |
| `mqar32|alibi-hybrid-h6|seed1` |  |  | 0.1339 | 332.3 | 0.0010 | 832,490 |
| `mqar32|alibi-hybrid-h6|seed2` |  |  | 0.0866 | 332.6 | 0.0010 | 832,490 |
| `sweep|mqar32|alibi-pure|0.001` |  |  | 0.0840 | 170.6 | 0.0010 | 820,352 |
| `sweep|mqar32|alibi-pure|0.003` |  |  | 0.0607 | 171.1 | 0.0030 | 820,352 |
| `mqar32|alibi-pure|seed0` |  |  | 0.0840 | 170.6 | 0.0010 | 820,352 |
| `mqar32|alibi-pure|seed1` |  |  | 0.0815 | 171.0 | 0.0010 | 820,352 |
| `mqar32|alibi-pure|seed2` |  |  | 0.0829 | 170.8 | 0.0010 | 820,352 |
| `sweep|mqar32|rope-gpt|0.001` |  |  | 0.0821 | 180.9 | 0.0010 | 820,352 |
| `sweep|mqar32|rope-gpt|0.003` |  |  | 0.0708 | 181.1 | 0.0030 | 820,352 |
| `mqar32|rope-gpt|seed0` |  |  | 0.0821 | 180.9 | 0.0010 | 820,352 |
| `mqar32|rope-gpt|seed1` |  |  | 0.0824 | 180.7 | 0.0010 | 820,352 |
| `mqar32|rope-gpt|seed2` |  |  | 0.0836 | 180.9 | 0.0010 | 820,352 |
| `sweep|mqar32|sheafdelta-pure|0.001` |  |  | 0.9517 | 382.1 | 0.0010 | 851,896 |
| `sweep|mqar32|sheafdelta-pure|0.003` |  |  | 0.1095 | 382.8 | 0.0030 | 851,896 |
| `mqar32|sheafdelta-pure|seed0` |  |  | 0.9517 | 382.1 | 0.0010 | 851,896 |
| `mqar32|sheafdelta-pure|seed1` |  |  | 0.0905 | 382.7 | 0.0010 | 851,896 |
| `mqar32|sheafdelta-pure|seed2` |  |  | 0.8722 | 382.7 | 0.0010 | 851,896 |
| `sweep|mqar64|alibi-hybrid-h6|0.001` |  |  | 0.0612 | 332.9 | 0.0010 | 832,490 |
| `sweep|mqar64|alibi-hybrid-h6|0.003` |  |  | 0.0440 | 332.6 | 0.0030 | 832,490 |
| `mqar64|alibi-hybrid-h6|seed0` |  |  | 0.0612 | 332.9 | 0.0010 | 832,490 |
| `mqar64|alibi-hybrid-h6|seed1` |  |  | 0.0601 | 332.5 | 0.0010 | 832,490 |
| `mqar64|alibi-hybrid-h6|seed2` |  |  | 0.0619 | 332.4 | 0.0010 | 832,490 |
| `sweep|mqar64|alibi-pure|0.001` |  |  | 0.0592 | 170.8 | 0.0010 | 820,352 |
| `sweep|mqar64|alibi-pure|0.003` |  |  | 0.0448 | 170.8 | 0.0030 | 820,352 |
| `mqar64|alibi-pure|seed0` |  |  | 0.0592 | 170.8 | 0.0010 | 820,352 |
| `mqar64|alibi-pure|seed1` |  |  | 0.0522 | 171.0 | 0.0010 | 820,352 |
| `mqar64|alibi-pure|seed2` |  |  | 0.0596 | 170.9 | 0.0010 | 820,352 |
| `sweep|mqar64|rope-gpt|0.001` |  |  | 0.0615 | 180.8 | 0.0010 | 820,352 |
| `sweep|mqar64|rope-gpt|0.003` |  |  | 0.0456 | 181.0 | 0.0030 | 820,352 |
| `mqar64|rope-gpt|seed0` |  |  | 0.0615 | 180.8 | 0.0010 | 820,352 |
| `mqar64|rope-gpt|seed1` |  |  | 0.0616 | 180.9 | 0.0010 | 820,352 |
| `mqar64|rope-gpt|seed2` |  |  | 0.0606 | 180.8 | 0.0010 | 820,352 |
| `sweep|mqar64|sheafdelta-pure|0.001` |  |  | 0.3747 | 381.8 | 0.0010 | 851,896 |
| `sweep|mqar64|sheafdelta-pure|0.003` |  |  | 0.0392 | 382.3 | 0.0030 | 851,896 |
| `mqar64|sheafdelta-pure|seed0` |  |  | 0.3747 | 381.8 | 0.0010 | 851,896 |
| `mqar64|sheafdelta-pure|seed1` |  |  | 0.9164 | 382.1 | 0.0010 | 851,896 |
| `mqar64|sheafdelta-pure|seed2` |  |  | 0.4832 | 381.8 | 0.0010 | 851,896 |
| `sweep|s5|alibi-hybrid-h6|0.001` | 0.0236 | 0.0124 |  | 60.5688 | 0.0010 | 832,490 |
| `sweep|s5|alibi-hybrid-h6|0.003` | 0.0236 | 0.0124 |  | 60.4872 | 0.0030 | 832,490 |
| `s5|alibi-hybrid-h6|seed0` | 0.0236 | 0.0124 |  | 60.5688 | 0.0010 | 832,490 |
| `s5|alibi-hybrid-h6|seed1` | 0.0241 | 0.0122 |  | 60.6875 | 0.0010 | 832,490 |
| `s5|alibi-hybrid-h6|seed2` | 0.0236 | 0.0124 |  | 60.7502 | 0.0010 | 832,490 |
| `sweep|s5|alibi-pure|0.001` | 0.0237 | 0.0124 |  | 20.3816 | 0.0010 | 820,352 |
| `sweep|s5|alibi-pure|0.003` | 0.0236 | 0.0123 |  | 19.5814 | 0.0030 | 820,352 |
| `s5|alibi-pure|seed0` | 0.0237 | 0.0124 |  | 20.3816 | 0.0010 | 820,352 |
| `s5|alibi-pure|seed1` | 0.0238 | 0.0122 |  | 21.1333 | 0.0010 | 820,352 |
| `s5|alibi-pure|seed2` | 0.0241 | 0.0124 |  | 20.9292 | 0.0010 | 820,352 |
| `sweep|s5|rope-gpt|0.001` | 0.0237 | 0.0125 |  | 22.1040 | 0.0010 | 820,352 |
| `sweep|s5|rope-gpt|0.003` | 0.0237 | 0.0123 |  | 21.1784 | 0.0030 | 820,352 |
| `s5|rope-gpt|seed0` | 0.0237 | 0.0123 |  | 21.1784 | 0.0030 | 820,352 |
| `s5|rope-gpt|seed1` | 0.0236 | 0.0124 |  | 21.2614 | 0.0030 | 820,352 |
| `s5|rope-gpt|seed2` | 0.0239 | 0.0122 |  | 25.0565 | 0.0030 | 820,352 |
| `sweep|s5|sheafdelta-pure|0.001` | 0.0235 | 0.0123 |  | 74.2456 | 0.0010 | 851,896 |
| `sweep|s5|sheafdelta-pure|0.003` | 0.0235 | 0.0123 |  | 73.8518 | 0.0030 | 851,896 |
| `s5|sheafdelta-pure|seed0` | 0.0235 | 0.0123 |  | 74.2456 | 0.0010 | 851,896 |
| `s5|sheafdelta-pure|seed1` | 0.0235 | 0.0123 |  | 73.1243 | 0.0010 | 851,896 |
| `s5|sheafdelta-pure|seed2` | 0.0238 | 0.0125 |  | 74.2192 | 0.0010 | 851,896 |
| `sweep|parity|alibi-hybrid-h6|0.001` | 1.0000 | 0.9970 |  | 57.3884 | 0.0010 | 832,490 |
| `sweep|parity|alibi-hybrid-h6|0.003` | 0.9999 | 0.9404 |  | 58.9425 | 0.0030 | 832,490 |
| `parity|alibi-hybrid-h6|seed0` | 1.0000 | 0.9970 |  | 57.3884 | 0.0010 | 832,490 |
| `parity|alibi-hybrid-h6|seed1` | 1.0000 | 0.9523 |  | 57.0160 | 0.0010 | 832,490 |
| `parity|alibi-hybrid-h6|seed2` | 1.0000 | 0.9889 |  | 56.7491 | 0.0010 | 832,490 |
| `sweep|parity|alibi-pure|0.001` | 0.5825 | 0.5191 |  | 16.5485 | 0.0010 | 820,352 |
| `sweep|parity|alibi-pure|0.003` | 0.5888 | 0.5228 |  | 16.7558 | 0.0030 | 820,352 |
| `parity|alibi-pure|seed0` | 0.5888 | 0.5228 |  | 16.7558 | 0.0030 | 820,352 |
| `parity|alibi-pure|seed1` | 0.5404 | 0.5091 |  | 18.6044 | 0.0030 | 820,352 |
| `parity|alibi-pure|seed2` | 0.5526 | 0.5135 |  | 18.0826 | 0.0030 | 820,352 |
| `sweep|parity|rope-gpt|0.001` | 0.7898 | 0.5721 |  | 21.2409 | 0.0010 | 820,352 |
| `sweep|parity|rope-gpt|0.003` | 0.5291 | 0.5079 |  | 21.4918 | 0.0030 | 820,352 |
| `parity|rope-gpt|seed0` | 0.7898 | 0.5721 |  | 21.2409 | 0.0010 | 820,352 |
| `parity|rope-gpt|seed1` | 0.7001 | 0.5490 |  | 21.8900 | 0.0010 | 820,352 |
| `parity|rope-gpt|seed2` | 0.5879 | 0.5230 |  | 19.4608 | 0.0010 | 820,352 |
| `sweep|parity|sheafdelta-pure|0.001` | 1.0000 | 1.0000 |  | 73.9213 | 0.0010 | 851,896 |
| `sweep|parity|sheafdelta-pure|0.003` | 1.0000 | 1.0000 |  | 86.2559 | 0.0030 | 851,896 |
| `parity|sheafdelta-pure|seed0` | 1.0000 | 1.0000 |  | 73.9213 | 0.0010 | 851,896 |
| `parity|sheafdelta-pure|seed1` | 1.0000 | 1.0000 |  | 90.1522 | 0.0010 | 851,896 |
| `parity|sheafdelta-pure|seed2` | 1.0000 | 0.9927 |  | 78.2568 | 0.0010 | 851,896 |
| `sweep|copy|alibi-hybrid-h6|0.001` |  |  |  | 176.3 | 0.0010 | 832,490 |
| `sweep|copy|alibi-hybrid-h6|0.003` |  |  |  | 169.0 | 0.0030 | 832,490 |
| `copy|alibi-hybrid-h6|seed0` |  |  |  | 169.0 | 0.0030 | 832,490 |
| `copy|alibi-hybrid-h6|seed1` |  |  |  | 138.0 | 0.0030 | 832,490 |
| `copy|alibi-hybrid-h6|seed2` |  |  |  | 138.1 | 0.0030 | 832,490 |
| `sweep|copy|alibi-pure|0.001` |  |  |  | 70.0575 | 0.0010 | 820,352 |
| `sweep|copy|alibi-pure|0.003` |  |  |  | 70.1866 | 0.0030 | 820,352 |
| `copy|alibi-pure|seed0` |  |  |  | 70.0575 | 0.0010 | 820,352 |
| `copy|alibi-pure|seed1` |  |  |  | 70.1241 | 0.0010 | 820,352 |
| `copy|alibi-pure|seed2` |  |  |  | 70.1443 | 0.0010 | 820,352 |
| `sweep|copy|rope-gpt|0.001` |  |  |  | 74.1657 | 0.0010 | 820,352 |
| `sweep|copy|rope-gpt|0.003` |  |  |  | 74.0638 | 0.0030 | 820,352 |
| `copy|rope-gpt|seed0` |  |  |  | 74.1657 | 0.0010 | 820,352 |
| `copy|rope-gpt|seed1` |  |  |  | 74.0514 | 0.0010 | 820,352 |
| `copy|rope-gpt|seed2` |  |  |  | 74.1211 | 0.0010 | 820,352 |
| `sweep|copy|sheafdelta-pure|0.001` |  |  |  | 158.3 | 0.0010 | 851,896 |
| `sweep|copy|sheafdelta-pure|0.003` |  |  |  | 158.5 | 0.0030 | 851,896 |
| `copy|sheafdelta-pure|seed0` |  |  |  | 158.3 | 0.0010 | 851,896 |
| `copy|sheafdelta-pure|seed1` |  |  |  | 158.4 | 0.0010 | 851,896 |
| `copy|sheafdelta-pure|seed2` |  |  |  | 158.4 | 0.0010 | 851,896 |
| `sweep|selcopy|alibi-hybrid-h6|0.001` |  | 0.9941 |  | 149.0 | 0.0010 | 832,490 |
| `sweep|selcopy|alibi-hybrid-h6|0.003` |  | 0.9902 |  | 148.9 | 0.0030 | 832,490 |
| `selcopy|alibi-hybrid-h6|seed0` |  | 0.9941 |  | 149.0 | 0.0010 | 832,490 |
| `selcopy|alibi-hybrid-h6|seed1` |  | 0.9915 |  | 148.8 | 0.0010 | 832,490 |
| `selcopy|alibi-hybrid-h6|seed2` |  | 0.9898 |  | 148.9 | 0.0010 | 832,490 |
| `sweep|selcopy|alibi-pure|0.001` |  | 0.9356 |  | 78.5568 | 0.0010 | 820,352 |
| `sweep|selcopy|alibi-pure|0.003` |  | 0.9075 |  | 78.7315 | 0.0030 | 820,352 |
| `selcopy|alibi-pure|seed0` |  | 0.9356 |  | 78.5568 | 0.0010 | 820,352 |
| `selcopy|alibi-pure|seed1` |  | 0.9631 |  | 95.6407 | 0.0010 | 820,352 |
| `selcopy|alibi-pure|seed2` |  | 0.9335 |  | 103.8 | 0.0010 | 820,352 |
| `sweep|selcopy|rope-gpt|0.001` |  | 0.9672 |  | 86.4341 | 0.0010 | 820,352 |
| `sweep|selcopy|rope-gpt|0.003` |  | 0.9968 |  | 83.2304 | 0.0030 | 820,352 |
| `selcopy|rope-gpt|seed0` |  | 0.9968 |  | 83.2304 | 0.0030 | 820,352 |
| `selcopy|rope-gpt|seed1` |  | 0.9939 |  | 83.1028 | 0.0030 | 820,352 |
| `selcopy|rope-gpt|seed2` |  | 0.9970 |  | 85.9878 | 0.0030 | 820,352 |
| `sweep|selcopy|sheafdelta-pure|0.001` |  | 0.9991 |  | 228.2 | 0.0010 | 851,896 |
| `sweep|selcopy|sheafdelta-pure|0.003` |  | 0.9981 |  | 177.9 | 0.0030 | 851,896 |
| `selcopy|sheafdelta-pure|seed0` |  | 0.9991 |  | 228.2 | 0.0010 | 851,896 |
| `selcopy|sheafdelta-pure|seed1` |  | 0.9954 |  | 170.8 | 0.0010 | 851,896 |
| `selcopy|sheafdelta-pure|seed2` |  | 0.9996 |  | 170.8 | 0.0010 | 851,896 |
