# Fused Triton delta-rule kernel speed crossover

**Status:** superseded by training_overhead_profile  ·  **Paper:** not in paper (development history)  ·  **Script(s):** [`gpu_crossover.py`](../../../benchmarks/gpu_crossover.py)

## What it tests and how (from the script)

```text
Focused test of G1 (findings_report.md): does a real fused Triton kernel (fla chunk_gated_delta_rule)
make sheaf-delta train FASTER than a matched Transformer at long context? Fixed token budget/step
(B*N=8192, so B shrinks as N grows -- same convention as experiments/rbf.py), width 128/4 heads (narrow,
to afford the longest context this 8GB card can hold), fp32 and bf16, 3 warmup + median-of-5 timed iters.
```

## Reproduce

From the `hsnlm/` root (GPU runs need CUDA; see the top-level README for the environment):

```bash
python benchmarks/gpu_crossover.py
```

## Files

| file | kind | size |
|---|---|---|
| [`gpu_crossover.json`](gpu_crossover.json) | results (machine-readable) | 5.4 KB |
| [`gpu_crossover.log`](gpu_crossover.log) | stdout of the run | 5.2 KB |

## Results: `gpu_crossover.json`

Auto-extracted per-run metrics (bits/byte lower is better; accuracies in [0,1]; `x` = not solved). Paper tables aggregate these over seeds.

| run | ms | tok_s | peak_mem_mb | params |
|---|---|---|---|---|
| `gpt|fp32|256` | 48.2500 | 169,798 | 394.8 | 820,352 |
| `hsn-eager|fp32|256` | 178.3 | 45,938 | 841.9 | 826,736 |
| `hsn-fused|fp32|256` | 73.9300 | 110,815 | 650.3 | 826,736 |
| `gpt|fp32|512` | 48.2000 | 169,973 | 395.1 | 820,352 |
| `hsn-eager|fp32|512` | 207.9 | 39,412 | 844.6 | 826,736 |
| `hsn-fused|fp32|512` | 76.3100 | 107,356 | 650.6 | 826,736 |
| `gpt|fp32|1024` | 63.1000 | 129,824 | 395.6 | 820,352 |
| `hsn-eager|fp32|1024` | 304.1 | 26,943 | 846.0 | 826,736 |
| `hsn-fused|fp32|1024` | 74.8100 | 109,500 | 650.7 | 826,736 |
| `gpt|fp32|2048` | 89.8900 | 91,136 | 396.7 | 820,352 |
| `hsn-eager|fp32|2048` | 487.4 | 16,807 | 846.9 | 826,736 |
| `hsn-fused|fp32|2048` | 77.9900 | 105,043 | 651.0 | 826,736 |
| `gpt|fp32|4096` | 147.8 | 55,434 | 398.8 | 820,352 |
| `hsn-eager|fp32|4096` | 945.1 | 8,668 | 847.7 | 826,736 |
| `hsn-fused|fp32|4096` | 80.2800 | 102,048 | 651.5 | 826,736 |
| `gpt|fp32|8192` | 255.1 | 32,108 | 403.0 | 820,352 |
| `hsn-eager|fp32|8192` | 1,794.8 | 4,564 | 832.2 | 826,736 |
| `hsn-fused|fp32|8192` | 82.7800 | 98,967 | 652.4 | 826,736 |
| `gpt|fp32|16384` | 1,082.0 | 15,143 | 785.6 | 820,352 |
| `hsn-eager|fp32|16384` | 7,339.8 | 2,232 | 1,636.9 | 826,736 |
| `hsn-fused|fp32|16384` | 185.5 | 88,344 | 1,283.8 | 826,736 |
| `gpt|fp32|32768` | 3,931.7 | 8,334 | 1,550.8 | 820,352 |
| `hsn-eager|fp32|32768` | 28,709.5 | 1,141 | 3,256.5 | 826,736 |
| `hsn-fused|fp32|32768` | 391.1 | 83,791 | 2,547.6 | 826,736 |
| `gpt|bf16|256` | 19.7900 | 413,848 | 264.3 | 820,352 |
| `hsn-eager|bf16|256` | 138.4 | 59,190 | 573.5 | 826,736 |
| `hsn-fused|bf16|256` | 71.6900 | 114,267 | 443.8 | 826,736 |
| `gpt|bf16|512` | 23.8200 | 343,900 | 264.6 | 820,352 |
| `hsn-eager|bf16|512` | 198.8 | 41,212 | 575.4 | 826,736 |
| `hsn-fused|bf16|512` | 71.4000 | 114,742 | 443.8 | 826,736 |
| `gpt|bf16|1024` | 23.7400 | 345,043 | 265.1 | 820,352 |
| `hsn-eager|bf16|1024` | 316.1 | 25,919 | 576.5 | 826,736 |
| `hsn-fused|bf16|1024` | 71.0700 | 115,273 | 443.9 | 826,736 |
| `gpt|bf16|2048` | 24.4500 | 335,060 | 266.2 | 820,352 |
| `hsn-eager|bf16|2048` | 552.8 | 14,819 | 577.1 | 826,736 |
| `hsn-fused|bf16|2048` | 71.0000 | 115,387 | 444.1 | 826,736 |
| `gpt|bf16|4096` | 33.8700 | 241,873 | 268.3 | 820,352 |
| `hsn-eager|bf16|4096` | 1,038.6 | 7,887 | 577.8 | 826,736 |
| `hsn-fused|bf16|4096` | 74.9400 | 109,310 | 444.5 | 826,736 |
| `gpt|bf16|8192` | 43.7800 | 187,133 | 272.5 | 820,352 |
| `hsn-eager|bf16|8192` | 1,940.5 | 4,222 | 578.7 | 826,736 |
| `hsn-fused|bf16|8192` | 73.1000 | 112,066 | 445.3 | 826,736 |
| `gpt|bf16|16384` | 123.3 | 132,890 | 523.0 | 820,352 |
| `hsn-eager|bf16|16384` | 7,377.8 | 2,221 | 1,136.5 | 826,736 |
| `hsn-fused|bf16|16384` | 154.7 | 105,892 | 869.2 | 826,736 |
| `gpt|bf16|32768` | 427.6 | 76,634 | 1,024.0 | 820,352 |
| `hsn-eager|bf16|32768` | 14,854.5 | 2,206 | 2,242.2 | 826,736 |
| `hsn-fused|bf16|32768` | 179.8 | 182,270 | 1,718.6 | 826,736 |
