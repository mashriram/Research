# Long Range Arena subset (ListOps, IMDb bytes, sCIFAR), 2 seeds

**Status:** current  ·  **Paper:** Table 9 (tab:lra)  ·  **Script(s):** [`gpu_lra_subset.py`](../../../benchmarks/gpu_lra_subset.py)

## Summary

The first pass hit OOM at the 4k-byte IMDb task; the resume pass (gradient accumulation 4x8, same effective batch 32) completed IMDb and sCIFAR. ListOps is at chance for every model.

## What it tests and how (from the script)

```text
B5 -- Long Range Arena subset (Tay et al. 2021), causal formulation: each example is left-padded, followed by a QUERY
token, and the model must emit the class token at the query position (loss + accuracy only there). Vocab 268 =
256 data bytes + QUERY(256) + 10 class tokens (257..266) + PAD(267). Same architectures as the paper, at LRA scale:
dim 256, 4 layers ("ddda" for the hybrid), attn heads 4, mixer heads 4 (head dim 64). ALiBi via the exact fused path.
  listops  ListOps (Nangia & Bowman 2018; LRA generator settings: depth <= 10, <= 10 args, lengths 500-2000), one byte per
           token ([MAX=A [MIN=I [MED=D [SM=S, ']'=], digits '0'-'9'), 10 classes. 40k train / 2k test, generated (seed 0).
  text     IMDb sentiment (stanfordnlp/imdb), raw UTF-8 bytes truncated to the LAST 4,000 bytes, 2 classes. 25k / 25k (eval
           on a fixed 5k subset of test).
  image    sequential CIFAR-10 (grayscale, 1,024 pixels, raster order), 10 classes. 50k / 10k (eval on all 10k).
5,000 steps, B=32, AdamW lr 1e-3 (the common LRA setting; identical for every model), wd 0.05, warmup 200 + cosine, bf16,
seeds 0 and 1. Reports test accuracy and training throughput.
```

## Reproduce

From the `hsnlm/` root (GPU runs need CUDA; see the top-level README for the environment):

```bash
python benchmarks/gpu_lra_subset.py
```

## Files

| file | kind | size |
|---|---|---|
| [`gpu_lra_subset.json`](gpu_lra_subset.json) | results (machine-readable) | 4.0 KB |
| [`gpu_lra_subset.log`](gpu_lra_subset.log) | stdout of the run | 5.7 KB |
| [`gpu_lra_subset_resume.log`](gpu_lra_subset_resume.log) | stdout of the run | 3.1 KB |

## Results: `gpu_lra_subset.json`

Auto-extracted per-run metrics (bits/byte lower is better; accuracies in [0,1]; `x` = not solved). Paper tables aggregate these over seeds.

| run | test_acc | tok_s | peak_mem_mb | train_s | params |
|---|---|---|---|---|---|
| `listops|alibi-hybrid-h6|seed0` | 0.2020 | 119,128.6 | 5,190.1 | 2,583.4 | 3,256,852 |
| `listops|alibi-pure|seed0` | 0.1880 | 216,671.9 | 3,570.1 | 1,420.4 | 3,216,640 |
| `listops|rope-gpt|seed0` | 0.1795 | 196,271.4 | 3,771.2 | 1,568.0 | 3,216,640 |
| `listops|alibi-hybrid-h6|seed1` | 0.1965 | 119,984.5 | 5,194.1 | 2,565.9 | 3,256,852 |
| `listops|alibi-pure|seed1` | 0.1990 | 217,683.0 | 3,572.0 | 1,414.3 | 3,216,640 |
| `listops|rope-gpt|seed1` | 0.1845 | 196,725.4 | 3,772.0 | 1,565.0 | 3,216,640 |
| `text|alibi-hybrid-h6|seed0` | 0.7612 | 130,616.8 | 2,646.0 | 3,498.8 | 3,256,852 |
| `text|alibi-pure|seed0` | 0.6356 | 207,307.4 | 1,817.5 | 2,204.5 | 3,216,640 |
| `text|rope-gpt|seed0` | 0.5774 | 203,052.3 | 1,923.6 | 2,250.7 | 3,216,640 |
| `text|alibi-hybrid-h6|seed1` | 0.7550 | 129,641.1 | 2,664.6 | 3,530.4 | 3,256,852 |
| `text|alibi-pure|seed1` | 0.6382 | 207,059.2 | 1,828.8 | 2,210.4 | 3,216,640 |
| `text|rope-gpt|seed1` | 0.5156 | 203,278.5 | 1,939.2 | 2,251.5 | 3,216,640 |
| `image|alibi-hybrid-h6|seed0` | 0.3953 | 127,053.3 | 2,682.0 | 1,290.8 | 3,256,852 |
| `image|alibi-pure|seed0` | 0.3736 | 235,322.9 | 1,845.7 | 696.9 | 3,216,640 |
| `image|rope-gpt|seed0` | 0.3395 | 213,921.0 | 1,948.6 | 766.6 | 3,216,640 |
| `image|alibi-hybrid-h6|seed1` | 0.3963 | 126,962.8 | 2,682.0 | 1,291.7 | 3,256,852 |
| `image|alibi-pure|seed1` | 0.3632 | 235,302.4 | 1,845.7 | 697.0 | 3,216,640 |
| `image|rope-gpt|seed1` | 0.3582 | 213,747.4 | 1,948.6 | 767.3 | 3,216,640 |
