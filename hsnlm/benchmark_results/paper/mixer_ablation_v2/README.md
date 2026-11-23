# Sheaf-Delta mixer ablations (h6, decay init, post-norm, token shift)

**Status:** current  ·  **Paper:** Table 8 (tab:ablation)  ·  **Script(s):** [`gpu_v2_ablation.py`](../../../benchmarks/gpu_v2_ablation.py)

## Summary

3 seeds x 8 variants, 1,500 steps, enwik8 64 KB slice. h6 (6 mixer heads, d_h 64) adopted.

## What it tests and how (from the script)

```text
DEEPA v2 bonus ablation (NOT part of the locked v1 result set -- v1 is untouched by this,
every new knob is an opt-in Config field defaulting to exact v1 behavior, verified bit-exact
step()==forward() for both the eager and fused mixer before this script was trusted, see the
conversation for the sanity-check transcript). Tests the three genuinely zero-/near-zero-risk
items from improvements-scope.md, one at a time plus combined, against the v1 winner (ALiBi-
hybrid 3:1, gpu_multiseed_compiled.py's "baseline" entry is reused rather than rerun):
  h6        -- SheafDeltaMixer uses mixer_heads=6 (dh=64) instead of attention's heads=8 (dh=48);
               decoupled via Config.mixer_heads so the 2 ALiBi attention layers are NOT changed
               (a confound in the original proposal, which assumed a single shared `heads` field).
  decay     -- multi-scale decay-bias init (linspace(-1.0, 4.5, H)) instead of flat 3.0.
  postnorm  -- SheafDeltaMixer's onorm applied after summing delta+dyadic instead of before (uncompensated,
               later found to be a regression caused by chunk_gated_delta_rule's implicit dh**-0.5 readout
               scale -- see DEEPA_V2_ABLATION.md).
  combined  -- all three together.
Round 2 (layered on top of h6, the only round-1 win, per the corrected proposals):
  h6_postnorm_scaled -- postnorm with the dh**-0.5 scale compensated first (Config.postnorm_scaled).
  h6_token_shift     -- causal token-shift on q,k before F.normalize, learnable per-head alpha, with a
                        1-token raw-q/k state added to .step() (Config.token_shift).
  h6_both            -- both together.
Same 14.5M-param/width-384/8-layer/ctx-1024/dddaddda recipe as gpu_multiseed_compiled.py, 3
seeds, compiled, full eval (nwin=64 @1024, nwin=32 @2048/@4096), so directly comparable to the
"alibi-hybrid" row there.
```

## Reproduce

From the `hsnlm/` root (GPU runs need CUDA; see the top-level README for the environment):

```bash
python benchmarks/gpu_v2_ablation.py
```

## Files

| file | kind | size |
|---|---|---|
| [`gpu_v2_ablation.json`](gpu_v2_ablation.json) | results (machine-readable) | 6.3 KB |

## Results: `gpu_v2_ablation.json`

Auto-extracted per-run metrics (bits/byte lower is better; accuracies in [0,1]; `x` = not solved). Paper tables aggregate these over seeds.

| run | val_bpb | val_bpb_len2048 | val_bpb_len4096 | tok_s | peak_mem_mb | train_s | params |
|---|---|---|---|---|---|---|---|
| `h6|seed0` | 1.0292 | 1.0111 | 1.0943 | 80,832.1 | 2,719.9 | 304.0 | 14,436,348 |
| `h6|seed1` | 1.0311 | 1.0105 | 1.0859 | 80,744.1 | 2,720.3 | 304.4 | 14,436,348 |
| `h6|seed2` | 1.0272 | 1.0061 | 1.0804 | 80,631.9 | 2,720.3 | 304.8 | 14,436,348 |
| `decay|seed0` | 1.0566 | 1.0390 | 1.1195 | 75,429.8 | 2,827.0 | 325.8 | 14,468,592 |
| `decay|seed1` | 1.0345 | 1.0140 | 1.0933 | 75,366.8 | 2,827.0 | 326.1 | 14,468,592 |
| `decay|seed2` | 1.0444 | 1.0242 | 1.0983 | 75,339.6 | 2,827.0 | 326.2 | 14,468,592 |
| `postnorm|seed0` | 1.0845 | 1.0690 | 1.1510 | 74,544.4 | 2,753.4 | 329.7 | 14,468,592 |
| `postnorm|seed1` | 1.0793 | 1.0613 | 1.1403 | 74,515.6 | 2,753.4 | 329.8 | 14,468,592 |
| `postnorm|seed2` | 1.0508 | 1.0313 | 1.1006 | 74,543.7 | 2,753.4 | 329.7 | 14,468,592 |
| `combined|seed0` | 1.0440 | 1.0244 | 1.1087 | 75,737.3 | 2,718.2 | 324.5 | 14,436,348 |
| `combined|seed1` | 1.0885 | 1.0722 | 1.1434 | 75,745.2 | 2,718.2 | 324.5 | 14,436,348 |
| `combined|seed2` | 1.0823 | 1.0669 | 1.1465 | 75,790.4 | 2,718.2 | 324.3 | 14,436,348 |
| `h6_postnorm_scaled|seed0` | 1.0597 | 1.0422 | 1.1257 | 82,050.7 | 2,645.2 | 299.5 | 14,436,348 |
| `h6_postnorm_scaled|seed1` | 1.0666 | 1.0474 | 1.1202 | 81,963.6 | 2,644.7 | 299.8 | 14,436,348 |
| `h6_postnorm_scaled|seed2` | 1.0497 | 1.0308 | 1.1060 | 81,968.3 | 2,644.7 | 299.8 | 14,436,348 |
| `h6_token_shift|seed0` | 1.0609 | 1.0434 | 1.1236 | 77,552.2 | 3,169.6 | 316.9 | 14,436,420 |
| `h6_token_shift|seed1` | 1.0852 | 1.0670 | 1.1375 | 77,598.0 | 3,169.6 | 316.7 | 14,436,420 |
| `h6_token_shift|seed2` | 1.0792 | 1.0595 | 1.1341 | 77,545.7 | 3,169.6 | 316.9 | 14,436,420 |
| `h6_both|seed0` | 1.0292 | 1.0103 | 1.0951 | 78,648.7 | 3,097.3 | 312.5 | 14,436,420 |
| `h6_both|seed1` | 1.0593 | 1.0408 | 1.1160 | 78,724.4 | 3,097.3 | 312.2 | 14,436,420 |
| `h6_both|seed2` | 1.0731 | 1.0546 | 1.1297 | 78,709.4 | 3,097.3 | 312.2 | 14,436,420 |
