# Speed and quality measured together (isolated processes)

**Status:** superseded  ·  **Paper:** not in paper (development history)  ·  **Script(s):** [`gpu_reconcile.py`](../../../benchmarks/gpu_reconcile.py), [`gpu_reconcile_clean.py`](../../../benchmarks/gpu_reconcile_clean.py)

## What it tests and how (from the script)

```text
Reconciles the two halves of this report that were never measured together: §3's speed crossover
(narrow width, long context, bf16 -- sheaf-delta fused beats GPT there) and §5a's quality ranking
(3:1 hybrid ratio beats GPT -- but measured at width 384/ctx 1024, where GPT is still faster). This
trains REAL models (not just speed-timing with random data) at width=128/8 layers/ctx=16384/batch=1 --
the exact width and a context length inside the confirmed bf16 speed-crossover zone (§3.1: ctx 16384,
bf16, fused ~190-210k tok/s vs GPT's 132.9k tok/s) -- and asks: does the 3:1 hybrid's quality edge
survive here, in a regime where it is ALSO the faster architecture? Real enwik8 data, 1 seed.
Cheap: batch=1 x ctx=16384 is only 16,384 tokens/step, so even thousands of steps finish in minutes.
```

## Reproduce

From the `hsnlm/` root (GPU runs need CUDA; see the top-level README for the environment):

```bash
python benchmarks/gpu_reconcile.py
python benchmarks/gpu_reconcile_clean.py <arch>
```

## Files

| file | kind | size |
|---|---|---|
| [`gpu_reconcile.json`](gpu_reconcile.json) | results (machine-readable) | 1.7 KB |
| [`gpu_reconcile.log`](gpu_reconcile.log) | stdout of the run | 4.4 KB |
| [`gpu_reconcile_clean_gpt.json`](gpu_reconcile_clean_gpt.json) | results (machine-readable) | 0.2 KB |
| [`gpu_reconcile_clean_hsn-fused.json`](gpu_reconcile_clean_hsn-fused.json) | results (machine-readable) | 0.2 KB |
| [`gpu_reconcile_clean_hybrid.json`](gpu_reconcile_clean_hybrid.json) | results (machine-readable) | 0.2 KB |
| [`gpu_reconcile_clean_hsn.log`](gpu_reconcile_clean_hsn.log) | stdout of the run | 0.3 KB |

## Results: `gpu_reconcile.json`

Auto-extracted per-run metrics (bits/byte lower is better; accuracies in [0,1]; `x` = not solved). Paper tables aggregate these over seeds.

| run | val_bpb | tok_s | peak_mem_mb | train_s | params |
|---|---|---|---|---|---|
| `sweep|gpt|0.001` | 4.0507 | 108,919.1 | 978.3 | 45.1271 | 1,607,808 |
| `sweep|gpt|0.002` | 3.9634 | 125,749.1 | 978.3 | 39.0873 | 1,607,808 |
| `sweep|hsn-fused|0.001` | 2.9864 | 90,758.5 | 1,799.2 | 54.1569 | 1,620,576 |
| `sweep|hsn-fused|0.002` | 2.7502 | 104,308.2 | 1,672.1 | 47.1219 | 1,620,576 |
| `sweep|hybrid-fused|0.001` | 3.0223 | 107,502.1 | 1,512.4 | 45.7219 | 1,642,728 |
| `sweep|hybrid-fused|0.002` | 2.7811 | 107,441.9 | 1,512.4 | 45.7475 | 1,642,728 |
| `best_lr` |  |  |  |  |  |
| `lm|gpt` | 2.1763 | 122,805.2 | 978.3 | 400.2 | 1,607,808 |
| `lm|hsn-fused` | 1.9098 | 50,978.8 | 1,672.1 | 964.2 | 1,620,576 |
| `lm|hybrid-fused` | 1.9347 | 94,345.8 | 1,512.4 | 521.0 | 1,642,728 |

## Results: `gpu_reconcile_clean_gpt.json`

Auto-extracted per-run metrics (bits/byte lower is better; accuracies in [0,1]; `x` = not solved). Paper tables aggregate these over seeds.

_No per-run records; open the file directly._

## Results: `gpu_reconcile_clean_hsn-fused.json`

Auto-extracted per-run metrics (bits/byte lower is better; accuracies in [0,1]; `x` = not solved). Paper tables aggregate these over seeds.

_No per-run records; open the file directly._

## Results: `gpu_reconcile_clean_hybrid.json`

Auto-extracted per-run metrics (bits/byte lower is better; accuracies in [0,1]; `x` = not solved). Paper tables aggregate these over seeds.

_No per-run records; open the file directly._
