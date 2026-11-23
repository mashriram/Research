# Backprop-free credit assignment (DAD, DFA, targets, JVP)

**Status:** negative result  ·  **Paper:** not in paper (development history)  ·  **Script(s):** [`gpu_credit.py`](../../../benchmarks/gpu_credit.py)

## What it tests and how (from the script)

```text
GPU port of experiments/credit.py: confirms the CPU finding that no backprop-alternative credit-assignment
rule saves wall-clock (each block still does its own forward+local-backward). Char-level Shakespeare,
same 0.82M-param HSN (default 'sheaf' v1 mixer, matching the original CPU experiment), 800 steps, 1 seed.
```

## Reproduce

From the `hsnlm/` root (GPU runs need CUDA; see the top-level README for the environment):

```bash
python benchmarks/gpu_credit.py
```

## Files

| file | kind | size |
|---|---|---|
| [`gpu_credit.json`](gpu_credit.json) | results (machine-readable) | 0.3 KB |
| [`gpu_credit.log`](gpu_credit.log) | stdout of the run | 0.6 KB |

## Results: `gpu_credit.json`

Auto-extracted per-run metrics (bits/byte lower is better; accuracies in [0,1]; `x` = not solved). Paper tables aggregate these over seeds.

| run | train_s |
|---|---|
| `backprop(teacher)` | 53.2830 |
| `DFA(random feedback)` | 36.3479 |
| `layerwise teacher-target (no x-block grad)` | 71.7322 |
