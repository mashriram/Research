# Training speed/memory vs context: Transformer vs HSN (eager)

**Status:** superseded by training_overhead_profile  ·  **Paper:** not in paper (development history)  ·  **Script(s):** [`gpu_speed.py`](../../../benchmarks/gpu_speed.py)

## What it tests and how (from the script)

```text
GPU training-speed/memory scaling vs context length: Transformer (fused SDPA) vs HSN eager (python chunk loop)
vs HSN fused (fla Triton chunk_gated_delta_rule). Answers G1 from findings_report.md: does a fused kernel
close/flip the CPU-measured 2-3x training-speed gap, and does the O(N) vs O(N^2) crossover appear at longer context?
Fixed batch B=16; context N swept; forward+backward timed after warmup, 3 best-of-5 iters. fp32 and bf16-autocast both measured.
usage: gpu_speed.py [smoke]
```

## Reproduce

From the `hsnlm/` root (GPU runs need CUDA; see the top-level README for the environment):

```bash
python benchmarks/gpu_speed.py
```

## Files

| file | kind | size |
|---|---|---|
| [`gpu_speed.json`](gpu_speed.json) | results (machine-readable) | 3.7 KB |
| [`gpu_speed.log`](gpu_speed.log) | stdout of the run | 3.6 KB |

## Results: `gpu_speed.json`

Auto-extracted per-run metrics (bits/byte lower is better; accuracies in [0,1]; `x` = not solved). Paper tables aggregate these over seeds.

| run | ms | tok_s | peak_mem_mb | params |
|---|---|---|---|---|
| `gpt|fp32|256` | 59.4800 | 68,865 | 396.1 | 3,213,568 |
| `hsn-eager|fp32|256` | 188.9 | 21,684 | 843.1 | 3,228,256 |
| `hsn-fused|fp32|256` | 91.4000 | 44,815 | 649.8 | 3,228,256 |
| `gpt|fp32|512` | 135.8 | 60,305 | 762.0 | 3,213,568 |
| `hsn-eager|fp32|512` | 519.7 | 15,763 | 1,652.1 | 3,228,256 |
| `hsn-fused|fp32|512` | 200.0 | 40,964 | 1,270.1 | 3,228,256 |
| `gpt|fp32|1024` | 356.8 | 45,922 | 1,479.1 | 3,213,568 |
| `hsn-eager|fp32|1024` | 1,453.4 | 11,273 | 3,259.8 | 3,228,256 |
| `hsn-fused|fp32|1024` | 428.5 | 38,233 | 2,492.2 | 3,228,256 |
| `gpt|fp32|2048` | 982.2 | 33,360 | 2,923.7 | 3,213,568 |
| `hsn-eager|fp32|2048` | 7,142.3 | 4,588 | 6,489.8 | 3,228,256 |
| `hsn-fused|fp32|2048` | 925.0 | 35,424 | 4,950.5 | 3,228,256 |
| `gpt|fp32|4096` | 2,898.4 | 22,611 | 5,817.2 | 3,213,568 |
| `hsn-eager|fp32|4096` |  |  |  |  |
| `hsn-fused|fp32|4096` |  |  |  |  |
| `gpt|fp32|8192` |  |  |  |  |
| `hsn-eager|fp32|8192` |  |  |  |  |
| `hsn-fused|fp32|8192` |  |  |  |  |
| `gpt|bf16|256` | 23.0300 | 177,825 | 276.5 | 3,213,568 |
| `hsn-eager|bf16|256` | 137.4 | 29,805 | 584.9 | 3,228,256 |
| `hsn-fused|bf16|256` | 68.4300 | 59,855 | 456.2 | 3,228,256 |
| `gpt|bf16|512` | 49.0600 | 166,993 | 516.6 | 3,213,568 |
| `hsn-eager|bf16|512` | 377.5 | 21,702 | 1,139.5 | 3,228,256 |
| `hsn-fused|bf16|512` | 142.6 | 57,431 | 875.8 | 3,228,256 |
| `gpt|bf16|1024` | 116.8 | 140,241 | 982.1 | 3,213,568 |
| `hsn-eager|bf16|1024` | 1,216.5 | 13,468 | 2,218.4 | 3,228,256 |
| `hsn-fused|bf16|1024` | 327.1 | 50,095 | 1,697.7 | 3,228,256 |
| `gpt|bf16|2048` | 294.6 | 111,213 | 1,923.4 | 3,213,568 |
| `hsn-eager|bf16|2048` | 6,521.5 | 5,025 | 4,396.3 | 3,228,256 |
| `hsn-fused|bf16|2048` | 724.1 | 45,251 | 3,348.8 | 3,228,256 |
| `gpt|bf16|4096` | 750.6 | 87,306 | 3,810.3 | 3,213,568 |
| `hsn-eager|bf16|4096` |  |  |  |  |
| `hsn-fused|bf16|4096` |  |  |  |  |
| `gpt|bf16|8192` |  |  |  |  |
| `hsn-eager|bf16|8192` |  |  |  |  |
| `hsn-fused|bf16|8192` |  |  |  |  |
