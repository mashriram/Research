# Sliding-window ALiBi hybrid (W=512)

**Status:** negative result  ·  **Paper:** not in paper (development history)  ·  **Script(s):** [`gpu_swa_alibi_hybrid.py`](../../../benchmarks/gpu_swa_alibi_hybrid.py)

## Summary

Source of the SWA 1.061 bpb negative result.

## What it tests and how (from the script)

```text
Sliding-Window ALiBi + Sheaf-Delta 3:1 Hybrid (`dddaddda`) + compiled zero-gather dyadic mixer.

Why the un-windowed ALiBi-Hybrid (`gpu_alibi_hybrid.py`) beat pure ALiBi at training length
(0.951 vs 0.956 bpb @ 1024) but lost extrapolation at 4x (1.222 vs 1.084 bpb @ 4096):
1) Unbounded ALiBi penalties (-slope * 4096) at 4x context push attention logits far outside the
   training distribution, while the 6 recurrent sheaf-delta layers were never trained to bridge
   attention window boundaries because full-context attention covered all 1024 tokens during training.
2) Training with Sliding-Window ALiBi (W=512 on ctx=1024, or W=1024 clamped at eval) forces the
   model DURING training to use ALiBi attention for sharp local lookups (<= W) and the 6 sheaf-delta
   layers for long-range context (> W), while capping the attention KV cache to O(W) = O(1) at inference.
3) Combined with the zero-gather `dyadic` fix in `fastmixer.py` and `torch.compile` on the sheaf-delta
   mixer, this also closes the training speed and VRAM gap.
```

## Reproduce

From the `hsnlm/` root (GPU runs need CUDA; see the top-level README for the environment):

```bash
python benchmarks/gpu_swa_alibi_hybrid.py
```

## Files

| file | kind | size |
|---|---|---|
| [`gpu_swa_alibi_hybrid.json`](gpu_swa_alibi_hybrid.json) | results (machine-readable) | 1.2 KB |
| [`gpu_swa_alibi_hybrid.log`](gpu_swa_alibi_hybrid.log) | stdout of the run | 2.6 KB |

Archived (invalid, do not use): [`gpu_swa_alibi_hybrid.json.BROKEN_residualbug`](../../_archive/gpu_swa_alibi_hybrid.json.BROKEN_residualbug)

## Results: `gpu_swa_alibi_hybrid.json`

Auto-extracted per-run metrics (bits/byte lower is better; accuracies in [0,1]; `x` = not solved). Paper tables aggregate these over seeds.

| run | val_bpb | val_bpb_len2048 | val_bpb_len4096 | tok_s | peak_mem_mb | train_s | params |
|---|---|---|---|---|---|---|---|
| `decode_o1` |  |  |  |  |  |  |  |
| `lm_swa512` | 1.0611 | 1.0418 | 1.1200 | 44,685.5 | 4,115.7 | 550.0 | 14,468,592 |
| `lm_swa1024` | 1.0298 | 1.0118 | 1.0896 | 44,657.2 | 4,157.3 | 550.3 | 14,468,592 |
