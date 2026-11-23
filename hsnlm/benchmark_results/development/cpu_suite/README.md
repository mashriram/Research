# Original CPU benchmark suite (HSN vs GPT)

**Status:** superseded  ·  **Paper:** not in paper (development history)  ·  **Script(s):** [`cpu_suite.py`](../../../benchmarks/cpu_suite.py)

## What it tests and how (from the script)

```text
Full CPU benchmark suite. Resumable: benchmark_results/development/cpu_suite/cpu_suite.json holds finished runs; rerun skips them.
Stages: sweep (lr per arch) -> lm seed0 -> synthetic (MQAR, parity) -> lm seed1.   usage: cpu_suite.py [smoke]
```

## Reproduce

From the `hsnlm/` root (GPU runs need CUDA; see the top-level README for the environment):

```bash
python benchmarks/cpu_suite.py
```

## Files

| file | kind | size |
|---|---|---|
| [`cpu_suite.json`](cpu_suite.json) | results (machine-readable) | 6.9 KB |
| [`cpu_suite.log`](cpu_suite.log) | stdout of the run | 6.2 KB |
| [`BENCH_CPU.md`](BENCH_CPU.md) | write-up | 1.6 KB |

## Results: `cpu_suite.json`

Auto-extracted per-run metrics (bits/byte lower is better; accuracies in [0,1]; `x` = not solved). Paper tables aggregate these over seeds.

| run | val_bpb | val_bpb_len2048 | tok_s | train_s | params |
|---|---|---|---|---|---|
| `sweep|gpt|0.001` | 3.5945 |  | 12,959.5 | 63.2123 | 820,352 |
| `sweep|gpt|0.002` | 3.3387 |  | 12,803.0 | 63.9850 | 820,352 |
| `sweep|gpt|0.004` | 3.1821 |  | 12,932.1 | 63.3461 | 820,352 |
| `sweep|sheaf-delta|0.001` | 3.1841 |  | 5,956.9 | 137.5 | 826,736 |
| `sweep|sheaf-delta|0.002` | 2.8509 |  | 6,023.2 | 136.0 | 826,736 |
| `sweep|sheaf-delta|0.004` | 2.7102 |  | 5,959.8 | 137.5 | 826,736 |
| `sweep|hybrid|0.001` | 3.1709 |  | 6,872.3 | 119.2 | 837,812 |
| `sweep|hybrid|0.002` | 2.8914 |  | 6,765.4 | 121.1 | 837,812 |
| `sweep|hybrid|0.004` | 2.7500 |  | 6,720.3 | 121.9 | 837,812 |
| `best_lr` |  |  |  |  |  |
| `lm|gpt|seed0` | 1.9567 | 4.1575 | 12,483.3 | 820.3 | 820,352 |
| `lm|sheaf-delta|seed0` | 1.8888 | 1.8820 | 6,255.9 | 1,636.8 | 826,736 |
| `lm|hybrid|seed0` | 1.8866 | 1.9358 | 6,873.5 | 1,489.8 | 837,812 |
| `mqar|gpt|pairs4` |  |  |  | 43.2989 |  |
| `mqar|sheaf-delta|pairs4` |  |  |  | 106.8 |  |
| `mqar|hybrid|pairs4` |  |  |  | 72.8749 |  |
| `mqar|gpt|pairs8` |  |  |  | 60.4959 |  |
| `mqar|sheaf-delta|pairs8` |  |  |  | 147.6 |  |
| `mqar|hybrid|pairs8` |  |  |  | 102.6 |  |
| `mqar|gpt|pairs16` |  |  |  | 88.6823 |  |
| `mqar|sheaf-delta|pairs16` |  |  |  | 226.8 |  |
| `mqar|hybrid|pairs16` |  |  |  | 156.3 |  |
| `parity|gpt` |  |  |  | 96.6609 |  |
| `parity|sheaf-delta` |  |  |  | 221.4 |  |
| `parity|hybrid` |  |  |  | 146.7 |  |
| `lm|gpt|seed1` | 1.9665 | 4.2656 | 14,097.3 | 726.4 | 820,352 |
| `lm|sheaf-delta|seed1` | 1.8702 | 1.8752 | 6,785.5 | 1,509.1 | 826,736 |
| `lm|hybrid|seed1` | 1.8924 | 1.9493 | 7,826.3 | 1,308.4 | 837,812 |
