# Causal control: parity with beta capped to (0,1)

**Status:** current  ·  **Paper:** Table 9b (parity control)  ·  **Script(s):** [`gpu_synthetic_proper.py`](../../../benchmarks/gpu_synthetic_proper.py)

## Summary

Same protocol as synthetic_converged; only the delta-rule step-size cap changes (no negative eigenvalues).

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
PROPER_ARCHS=alibi-hybrid-h6-beta1,sheafdelta-pure-beta1 PROPER_OUT=benchmark_results/paper/parity_negative_eigenvalue_control/gpu_parity_beta_control.json python benchmarks/gpu_synthetic_proper.py parity
```

## Files

| file | kind | size |
|---|---|---|
| [`gpu_parity_beta_control.json`](gpu_parity_beta_control.json) | results (machine-readable) | 22.6 KB |
| [`gpu_parity_beta_control.log`](gpu_parity_beta_control.log) | stdout of the run | 27.9 KB |

## Results: `gpu_parity_beta_control.json`

Auto-extracted per-run metrics (bits/byte lower is better; accuracies in [0,1]; `x` = not solved). Paper tables aggregate these over seeds.

| run | acc_T64 | acc_T256 | steps_to_solve | reached_len | train_s | lr | params |
|---|---|---|---|---|---|---|---|
| `sweep|parity|alibi-hybrid-h6-beta1|0.0003` | 0.9990 | 0.6454 | 10,000 | 64 | 177.1 | 0.0003 | 832,490 |
| `sweep|parity|alibi-hybrid-h6-beta1|0.001` | 0.9992 | 0.6644 | 11,500 | 64 | 145.7 | 0.0010 | 832,490 |
| `sweep|parity|alibi-hybrid-h6-beta1|0.003` | 0.9991 | 0.6537 | 13,000 | 64 | 153.7 | 0.0030 | 832,490 |
| `parity|alibi-hybrid-h6-beta1|seed0` | 0.9992 | 0.6644 | 11,500 | 64 | 145.7 | 0.0010 | 832,490 |
| `parity|alibi-hybrid-h6-beta1|seed1` | 0.9934 | 0.6454 | 12,500 | 64 | 133.5 | 0.0010 | 832,490 |
| `parity|alibi-hybrid-h6-beta1|seed2` | 0.9980 | 0.6372 | 12,500 | 64 | 134.2 | 0.0010 | 832,490 |
| `sweep|parity|sheafdelta-pure-beta1|0.0003` | 0.9956 | 0.6236 | 9,500 | 64 | 120.0 | 0.0003 | 851,896 |
| `sweep|parity|sheafdelta-pure-beta1|0.001` | 0.9982 | 0.6423 | 11,500 | 64 | 158.3 | 0.0010 | 851,896 |
| `sweep|parity|sheafdelta-pure-beta1|0.003` | 0.9936 | 0.6364 | 18,000 | 64 | 213.5 | 0.0030 | 851,896 |
| `parity|sheafdelta-pure-beta1|seed0` | 0.9982 | 0.6423 | 11,500 | 64 | 158.3 | 0.0010 | 851,896 |
| `parity|sheafdelta-pure-beta1|seed1` | 0.9991 | 0.6438 | 11,000 | 64 | 145.6 | 0.0010 | 851,896 |
| `parity|sheafdelta-pure-beta1|seed2` | 0.9964 | 0.6419 | 17,500 | 64 | 206.1 | 0.0010 | 851,896 |
