# Zero-shot length extrapolation 1k -> 16k (3 seeds)

**Status:** current  ·  **Paper:** Table 5 (tab:extrap), Fig. 2a  ·  **Script(s):** [`gpu_extrap_16k.py`](../../../benchmarks/gpu_extrap_16k.py)

## Summary

Every length is scored on the same 512 KB of validation text. Checkpoints saved in checkpoints/extrap16k/.

## What it tests and how (from the script)

```text
Reviewer item 7 (length extrapolation): trains alibi-hybrid-h6, alibi-pure and rope-gpt at ctx 1024 on enwik8
(seed 0, 1,500 steps, the standard compiled recipe), saves the weights, then evaluates each model zero-shot at
L = 1024, 2048, 4096, 8192, 16384 (up to 16x the training length).
Fair-budget eval: every length is scored on the SAME number of validation bytes (524,288 = 512 x 1024 = 32 x 16384),
all taken from the start of the standard enwik8 validation slice (bytes 40M..41M). Batch 8 @<=2048, 4 @4096,
1 @8192/16384 (fits 8 GB). Also records per-position bits/byte at L=16384 in 1024-token buckets, which shows WHERE in a
long window each model degrades (positions > 1024 were never seen in training).
ALiBi at L > 4096 uses the exact query-chunked path in deepa_common.py (dense bias would need 8.6 GB at 16k).
```

## Reproduce

From the `hsnlm/` root (GPU runs need CUDA; see the top-level README for the environment):

```bash
python benchmarks/gpu_extrap_16k.py 0 1 2
```

## Files

| file | kind | size |
|---|---|---|
| [`gpu_extrap_16k.json`](gpu_extrap_16k.json) | results (machine-readable) | 10.0 KB |
| [`gpu_extrap_16k.log`](gpu_extrap_16k.log) | stdout of the run | 4.1 KB |
| [`gpu_extrap_16k_seeds12.log`](gpu_extrap_16k_seeds12.log) | stdout of the run | 8.0 KB |

## Results: `gpu_extrap_16k.json`

Auto-extracted per-run metrics (bits/byte lower is better; accuracies in [0,1]; `x` = not solved). Paper tables aggregate these over seeds.

| run | val_bpb_len2048 | val_bpb_len4096 | val_bpb_len8192 | val_bpb_len16384 | tok_s | peak_mem_mb | train_s |
|---|---|---|---|---|---|---|---|
| `alibi-hybrid-h6|seed0` | 1.4794 | 1.4687 | 1.4636 | 1.4605 | 81,091.5 | 2,717.1 | 303.1 |
| `alibi-pure|seed0` | 1.5060 | 1.4917 | 1.4850 | 1.4814 | 84,211.9 | 2,241.5 | 291.8 |
| `rope-gpt|seed0` | 2.4197 | 3.5866 | 4.6504 | 5.4797 | 98,118.2 | 2,409.3 | 250.5 |
| `alibi-hybrid-h6|seed1` | 1.4819 | 1.4713 | 1.4663 | 1.4636 | 80,808.0 | 2,717.1 | 304.1 |
| `alibi-pure|seed1` | 1.4975 | 1.4845 | 1.4778 | 1.4744 | 85,866.5 | 2,242.7 | 286.2 |
| `rope-gpt|seed1` | 2.3808 | 3.5853 | 4.6412 | 5.3191 | 98,350.1 | 2,408.7 | 249.9 |
| `alibi-hybrid-h6|seed2` | 1.4998 | 1.4907 | 1.4862 | 1.4838 | 80,872.1 | 2,718.3 | 303.9 |
| `alibi-pure|seed2` | 1.4963 | 1.4812 | 1.4742 | 1.4706 | 85,849.7 | 2,241.3 | 286.3 |
| `rope-gpt|seed2` | 2.2557 | 3.1891 | 4.2658 | 5.0983 | 98,394.8 | 2,408.7 | 249.8 |
