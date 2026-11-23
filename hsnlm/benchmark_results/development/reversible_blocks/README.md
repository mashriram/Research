# Reversible-block memory/speed trade-off

**Status:** exploratory  ·  **Paper:** not in paper (development history)  ·  **Script(s):** [`gpu_reversible.py`](../../../benchmarks/gpu_reversible.py)

## What it tests and how (from the script)

```text
GPU port of experiments/rbf.py: reversible-block memory/speed tradeoff (RBlockFast policy) vs
plain autograd, at context lengths the CPU run couldn't reach cheaply. CUDA-synced timing + memory.
```

## Reproduce

From the `hsnlm/` root (GPU runs need CUDA; see the top-level README for the environment):

```bash
python benchmarks/gpu_reversible.py
```
