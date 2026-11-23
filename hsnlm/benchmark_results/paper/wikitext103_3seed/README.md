# WikiText-103 (byte-level), 1,500 steps, 3 seeds

**Status:** current  ·  **Paper:** Table 6 (WikiText-103 1.5k row)  ·  **Script(s):** [`gpu_wikitext103_3seed.py`](../../../benchmarks/gpu_wikitext103_3seed.py), [`data_prep/build_wikitext103_bytes.py`](../../../benchmarks/data_prep/build_wikitext103_bytes.py)

## Summary

At 1,500 steps Pure ALiBi leads by 0.012 bpb; see wikitext103_lr_and_horizon for the tuned / longer run.

## What it tests and how (from the script)

```text
Reviewer item 4 (additional dataset): repeats the 3-seed, 1,500-step, compiled comparison on WikiText-103 (raw,
byte-level) instead of enwik8. Same three models and exact same recipe as gpu_splitwd_3seed.py /
gpu_multiseed_compiled.py (uniform-WD AdamW, lr 2e-3, B=16, ctx 1024, bf16, torch.compile).
Data: data/wikitext103_raw.bytes = first 40M bytes of the WikiText-103-raw train split + first 1M bytes of the
official validation split (benchmarks/data_prep/build_wikitext103_bytes.py), i.e. the same 40M/1M sizes as enwik8.
Eval: repo convention (val_bpb = 64 windows @1024, 32 windows @2048/@4096) plus val_bpb_full = all 976 windows
@1024 of the 1M-byte validation slice.
```

## Reproduce

From the `hsnlm/` root (GPU runs need CUDA; see the top-level README for the environment):

```bash
python benchmarks/data_prep/build_wikitext103_bytes.py
python benchmarks/gpu_wikitext103_3seed.py
```

## Files

| file | kind | size |
|---|---|---|
| [`gpu_wikitext103_3seed.json`](gpu_wikitext103_3seed.json) | results (machine-readable) | 6.1 KB |
| [`gpu_wikitext103_3seed.log`](gpu_wikitext103_3seed.log) | stdout of the run | 5.9 KB |

## Results: `gpu_wikitext103_3seed.json`

Auto-extracted per-run metrics (bits/byte lower is better; accuracies in [0,1]; `x` = not solved). Paper tables aggregate these over seeds.

| run | val_bpb_full | val_bpb | val_bpb_len2048 | val_bpb_len4096 | tok_s | peak_mem_mb | train_s |
|---|---|---|---|---|---|---|---|
| `alibi-hybrid-h6|seed0` | 1.6283 | 1.6029 | 1.5929 | 1.6381 | 80,828.8 | 2,718.8 | 304.1 |
| `alibi-pure|seed0` | 1.6154 | 1.5788 | 1.5496 | 1.5949 | 85,656.5 | 2,240.7 | 286.9 |
| `rope-gpt|seed0` | 1.7260 | 1.7184 | 2.3006 | 3.2038 | 98,064.1 | 2,409.1 | 250.6 |
| `alibi-hybrid-h6|seed1` | 1.6290 | 1.6070 | 1.5966 | 1.6421 | 80,681.2 | 2,718.8 | 304.6 |
| `alibi-pure|seed1` | 1.6129 | 1.5714 | 1.5414 | 1.5871 | 85,620.4 | 2,240.7 | 287.0 |
| `rope-gpt|seed1` | 1.7475 | 1.7320 | 2.3896 | 3.2409 | 97,996.4 | 2,409.1 | 250.8 |
| `alibi-hybrid-h6|seed2` | 1.6253 | 1.5988 | 1.5879 | 1.6296 | 80,723.0 | 2,718.8 | 304.4 |
| `alibi-pure|seed2` | 1.6179 | 1.5761 | 1.5472 | 1.5967 | 85,622.1 | 2,240.7 | 287.0 |
| `rope-gpt|seed2` | 1.7410 | 1.7261 | 2.4716 | 3.5477 | 98,013.5 | 2,409.1 | 250.7 |
