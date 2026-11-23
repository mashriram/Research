# ALiBi hybrids: 3:1 vs 4:1 dilution

**Status:** current (negative result for 4:1)  ·  **Paper:** not in paper (development history)  ·  **Script(s):** [`gpu_alibi_hybrid.py`](../../../benchmarks/gpu_alibi_hybrid.py), [`gpu_alibi_hybrid_ratio.py`](../../../benchmarks/gpu_alibi_hybrid_ratio.py)

## Summary

gpu_alibi_hybrid.json was produced by a run with a residual bug and is archived (see _archive/).

## What it tests and how (from the script)

```text
Direct answer to 'how do we beat ALiBi': ALiBi's quality edge (0.956 bpb, gpu_posenc.py) over the
RoPE-based 3:1 hybrid (1.014 bpb, gpu_ratio.py) might come from its attention mechanism specifically.
This builds the SAME 3:1 sheaf-delta:attention hybrid, but with the attention layers using ALiBi
instead of RoPE -- if ALiBi's quality boost transfers, this should beat both pure ALiBi (extrapolation
is ALiBi's weak point, 1.192 at 4x) and the RoPE-hybrid (quality is RoPE-hybrid's weak point, 1.014 at
train length), potentially dominating both. Same 14M-param/width-384/8-layer/ctx-1024 recipe, same
enwik8 data. Also includes a second-seed check on the three key configs (ALiBi, RoPE-hybrid, ALiBi-
hybrid) to make sure the ranking isn't within noise before investing further.
```

## Reproduce

From the `hsnlm/` root (GPU runs need CUDA; see the top-level README for the environment):

```bash
python benchmarks/gpu_alibi_hybrid.py
python benchmarks/gpu_alibi_hybrid_ratio.py
```

## Files

| file | kind | size |
|---|---|---|
| [`gpu_alibi_hybrid.log`](gpu_alibi_hybrid.log) | stdout of the run | 2.0 KB |
| [`gpu_alibi_hybrid_4to1.json`](gpu_alibi_hybrid_4to1.json) | results (machine-readable) | 0.6 KB |
| [`gpu_alibi_hybrid_4to1.log`](gpu_alibi_hybrid_4to1.log) | stdout of the run | 2.0 KB |

Archived (invalid, do not use): [`gpu_alibi_hybrid.json.BROKEN_residualbug`](../../_archive/gpu_alibi_hybrid.json.BROKEN_residualbug), [`gpu_alibi_hybrid_4to1.json.BROKEN_residualbug`](../../_archive/gpu_alibi_hybrid_4to1.json.BROKEN_residualbug)

## Results: `gpu_alibi_hybrid_4to1.json`

Auto-extracted per-run metrics (bits/byte lower is better; accuracies in [0,1]; `x` = not solved). Paper tables aggregate these over seeds.

| run | val_bpb | val_bpb_len2048 | val_bpb_len4096 | tok_s | peak_mem_mb | train_s | params |
|---|---|---|---|---|---|---|---|
| `sweep|0.001` | 3.4675 |  |  | 26,819.6 | 5,123.3 | 61.0896 | 18,147,520 |
| `sweep|0.002` | 3.0959 |  |  | 30,665.7 | 5,126.4 | 53.4277 | 18,147,520 |
| `lm` | 1.0417 | 1.0230 | 1.1006 | 30,615.4 | 5,126.4 | 802.7 | 18,147,520 |
