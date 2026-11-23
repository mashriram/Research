# Multi-seed compiled comparison (hybrids vs GPT)

**Status:** current  ·  **Paper:** Sec. 5.1 (iso-quality context)  ·  **Script(s):** [`gpu_multiseed_compiled.py`](../../../benchmarks/gpu_multiseed_compiled.py)

## What it tests and how (from the script)

```text
Multi-seed check on the closest race in the whole report: RoPE-hybrid 3:1 (gpu_ratio.py, seed 0:
1.0136/0.9950/1.0835), pure ALiBi (gpu_posenc_recheck.py, seed 0: 1.0300/1.0062/1.0872), and ALiBi-hybrid
3:1 (gpu_posenc_recheck.py, seed 0: 1.0237/1.0048/1.0875) are separated by only 0.01-0.016 bpb -- on the
order of single-seed noise. This reruns all three at seeds 0, 1, 2 (seed 0 redone fresh here too, under
torch.compile, rather than reusing the uncompiled seed-0 numbers, so the whole multi-seed set is
internally consistent) with full eval (nwin=64 @1024, nwin=32 @2048/@4096) and no missing cells.
```

## Reproduce

From the `hsnlm/` root (GPU runs need CUDA; see the top-level README for the environment):

```bash
python benchmarks/gpu_multiseed_compiled.py
```

## Files

| file | kind | size |
|---|---|---|
| [`gpu_multiseed_compiled.json`](gpu_multiseed_compiled.json) | results (machine-readable) | 2.7 KB |

## Results: `gpu_multiseed_compiled.json`

Auto-extracted per-run metrics (bits/byte lower is better; accuracies in [0,1]; `x` = not solved). Paper tables aggregate these over seeds.

| run | val_bpb | val_bpb_len2048 | val_bpb_len4096 | tok_s | peak_mem_mb | train_s | params |
|---|---|---|---|---|---|---|---|
| `rope-hybrid|seed0` | 1.0289 | 1.0122 | 1.1005 | 81,666.6 | 2,874.0 | 300.9 | 14,468,592 |
| `rope-hybrid|seed1` | 1.0079 | 0.9897 | 1.0741 | 81,398.8 | 2,872.6 | 301.9 | 14,468,592 |
| `rope-hybrid|seed2` | 1.0724 | 1.0590 | 1.1462 | 81,278.1 | 2,872.6 | 302.4 | 14,468,592 |
| `alibi-pure|seed0` | 1.0359 | 1.0117 | 1.0877 | 85,703.5 | 2,152.6 | 286.8 | 14,260,608 |
| `alibi-pure|seed1` | 1.0364 | 1.0088 | 1.0855 | 85,249.9 | 2,152.6 | 288.3 | 14,260,608 |
| `alibi-pure|seed2` | 1.0385 | 1.0082 | 1.0832 | 85,481.3 | 2,152.6 | 287.5 | 14,260,608 |
| `alibi-hybrid|seed0` | 1.0201 | 1.0003 | 1.0816 | 78,483.0 | 2,828.7 | 313.1 | 14,468,592 |
| `alibi-hybrid|seed1` | 1.0241 | 1.0041 | 1.0778 | 78,540.0 | 2,828.7 | 312.9 | 14,468,592 |
| `alibi-hybrid|seed2` | 1.0378 | 1.0158 | 1.0898 | 78,531.9 | 2,828.7 | 312.9 | 14,468,592 |
