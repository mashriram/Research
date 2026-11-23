# GPU replica of the CPU suite

**Status:** superseded  ·  **Paper:** not in paper (development history)  ·  **Script(s):** [`gpu_suite.py`](../../../benchmarks/gpu_suite.py)

## What it tests and how (from the script)

```text
GPU replica of cpu_suite.py: same architectures/sizes/steps/data, device='cuda'.
Purpose: direct CPU-vs-GPU comparison at matched scale (same 0.82M-param setup as benchmark_results/development/cpu_suite/cpu_suite.json).
Resumable: benchmark_results/development/gpu_suite/gpu_suite.json holds finished runs; rerun skips them.  usage: gpu_suite.py [smoke]
```

## Reproduce

From the `hsnlm/` root (GPU runs need CUDA; see the top-level README for the environment):

```bash
python benchmarks/gpu_suite.py
```

## Files

| file | kind | size |
|---|---|---|
| [`gpu_suite.json`](gpu_suite.json) | results (machine-readable) | 7.3 KB |
| [`gpu_suite.log`](gpu_suite.log) | stdout of the run | 6.2 KB |

## Results: `gpu_suite.json`

Auto-extracted per-run metrics (bits/byte lower is better; accuracies in [0,1]; `x` = not solved). Paper tables aggregate these over seeds.

| run | val_bpb | val_bpb_len2048 | tok_s | peak_mem_mb | train_s | params |
|---|---|---|---|---|---|---|
| `sweep|gpt|0.001` | 3.5945 |  | 294,086.7 | 214.3 | 2.7856 | 820,352 |
| `sweep|gpt|0.002` | 3.3387 |  | 325,513.0 | 214.3 | 2.5166 | 820,352 |
| `sweep|gpt|0.004` | 3.1821 |  | 324,893.2 | 214.3 | 2.5214 | 820,352 |
| `sweep|sheaf-delta|0.001` | 3.1726 |  | 65,000.5 | 437.9 | 12.6030 | 826,736 |
| `sweep|sheaf-delta|0.002` | 2.8519 |  | 64,086.7 | 437.9 | 12.7827 | 826,736 |
| `sweep|sheaf-delta|0.004` | 2.6860 |  | 52,571.9 | 437.9 | 15.5825 | 826,736 |
| `sweep|hybrid|0.001` | 3.1793 |  | 70,286.2 | 384.5 | 11.6552 | 837,812 |
| `sweep|hybrid|0.002` | 2.8873 |  | 75,483.2 | 384.5 | 10.8527 | 837,812 |
| `sweep|hybrid|0.004` | 2.7633 |  | 85,842.4 | 384.5 | 9.5431 | 837,812 |
| `best_lr` |  |  |  |  |  |  |
| `lm|gpt|seed0` | 1.9534 | 4.0846 | 298,719.2 | 214.3 | 34.2797 | 820,352 |
| `lm|sheaf-delta|seed0` | 1.8933 | 1.8825 | 45,498.4 | 437.9 | 225.1 | 826,736 |
| `lm|hybrid|seed0` | 1.8928 | 1.9370 | 47,882.0 | 384.5 | 213.9 | 837,812 |
| `mqar|gpt|pairs4` |  |  |  |  | 11.6833 |  |
| `mqar|sheaf-delta|pairs4` |  |  |  |  | 25.8188 |  |
| `mqar|hybrid|pairs4` |  |  |  |  | 21.7967 |  |
| `mqar|gpt|pairs8` |  |  |  |  | 12.1309 |  |
| `mqar|sheaf-delta|pairs8` |  |  |  |  | 39.3971 |  |
| `mqar|hybrid|pairs8` |  |  |  |  | 23.7099 |  |
| `mqar|gpt|pairs16` |  |  |  |  | 8.5165 |  |
| `mqar|sheaf-delta|pairs16` |  |  |  |  | 34.6641 |  |
| `mqar|hybrid|pairs16` |  |  |  |  | 22.6617 |  |
| `parity|gpt` |  |  |  |  | 10.5218 |  |
| `parity|sheaf-delta` |  |  |  |  | 36.5309 |  |
| `parity|hybrid` |  |  |  |  | 23.5731 |  |
| `lm|gpt|seed1` | 1.9623 | 4.2255 | 160,015.6 | 214.3 | 63.9938 | 820,352 |
| `lm|sheaf-delta|seed1` | 1.8776 | 1.8711 | 37,757.9 | 437.9 | 271.2 | 826,736 |
| `lm|hybrid|seed1` | 1.8909 | 1.9447 | 53,395.9 | 384.5 | 191.8 | 837,812 |
