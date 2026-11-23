# CPU benchmark suite (2026-11-09) — ~0.82-0.84M params, enwik8 bytes, 10M training tokens/run, 8 CPU cores, eager PyTorch
All three archs: 4 layers, width 128, matched params (+-2%), same optimizer/schedule/data order; lr picked per arch from {1e-3,2e-3,4e-3} (all chose 4e-3 = grid edge, so absolute numbers may improve for everyone).
Raw: results/cpu_suite.json, results/cpu_suite.log. Reproduce: python benchmarks/cpu_suite.py (resumable).

## Language modelling, validation bits/byte (lower better), mean of 2 seeds [seed0, seed1]
| arch | @256 (train len) | @512 | @1024 | @2048 | train tok/s (eager CPU) |
|---|---|---|---|---|---|
| Transformer (RoPE) | 1.962 [1.957, 1.966] | 2.501 | 3.482 | 4.212 | 13.3k |
| sheaf-delta (4 layers) | **1.880** [1.889, 1.870] | **1.845** | **1.886** | **1.879** | 6.5k |
| hybrid (3 sheaf-delta + 1 attn) | 1.890 [1.887, 1.892] | 1.863 | 1.930 | 1.943 | 7.4k |

## Associative recall (2 layers, width 64, 2500 steps, 1 seed, chance ~0.016)
| pairs | Transformer | sheaf-delta | hybrid |
|---|---|---|---|
| 4 | 0.462 | 0.532 | 0.502 |
| 8 | 0.222 | 0.330 | 0.344 |
| 16 | 0.109 | 0.170 | 0.238 |
All undertrained; shows faster learning of recall, not an attention limit.

## Parity / state tracking (train length 32; accuracy at 32/64/128; chance 0.5; 1 seed)
Transformer 0.566/0.531/0.514 | sheaf-delta 1.000/0.999/0.965 | hybrid 1.000/1.000/0.997

## Not shown / caveats
Tiny scale; speed is eager-mode CPU (sheaf models ~2x slower to train, no fused kernels); Transformer has no extrapolation tricks (e.g. NoPE/YaRN);
lr sweep hit grid edge; synthetic tasks single-seed; no downstream benchmarks; nothing here predicts 100M+ behaviour.
