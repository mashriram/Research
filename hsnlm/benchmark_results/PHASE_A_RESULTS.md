# Phase A results: reviewer-response experiments

All runs: byte-level LM, vocab 256, 14.4M-param width-384 recipe unless stated (AdamW β=(0.9,0.95), uniform wd 0.05, lr 2e-3, 100-step warmup, cosine to 0.1×, clip 1.0, B=16, ctx 1024, bf16, torch.compile), single RTX 2000 Ada Laptop GPU (8 GB). bpb = bits per byte (lower is better). ± = population std (ddof=0) over seeds, the same convention as Table 9.

**Evaluation-slice caveat (applies to every earlier table in the paper too).** The repo convention `val_bpb` scores only the first 64 windows × 1024 = 64 KB of the 1 MB validation slice. On enwik8 that region is much easier than the rest: the seed-0 h6 model scores 1.02 bpb on it but 1.21–1.90 on each later 64 KB block. Relative comparisons on the same windows remain valid, but absolute bpb on the 64 KB slice is optimistic. The new tables therefore also report `full` = all 976 windows of the validation slice, or a fixed 512 KB budget.

## 1. Additional dataset: WikiText-103 (raw, byte-level), 3 seeds, 1,500 steps (Reviewer items 4, 10)

Train = first 40 MB of the WikiText-103-raw train split; val = first 1 MB of the official validation split (sizes match enwik8).

| Model | Params | bpb @1024 (64 KB) | @2048 | @4096 | bpb full val @1024 (1 MB) | train tok/s | peak train MB |
|---|---|---|---|---|---|---|---|
| ALiBi-DEEPA 3:1 (h6) | 14.44M | 1.6029 ± 0.0033 | 1.5925 ± 0.0036 | 1.6366 ± 0.0052 | 1.6275 ± 0.0016 | 80,744 | 2,719 |
| Pure ALiBi | 14.26M | 1.5754 ± 0.0031 | 1.5461 ± 0.0034 | 1.5929 ± 0.0042 | 1.6154 ± 0.0020 | 85,633 | 2,241 |
| RoPE GPT | 14.26M | 1.7255 ± 0.0056 | 2.3873 ± 0.0698 | 3.3308 ± 0.1541 | 1.7382 ± 0.0090 | 98,025 | 2,409 |

**Finding:** on WikiText-103, pure ALiBi beats the hybrid at every length and every seed (gap ≈ 0.03 bpb on the 64 KB slice, ≈ 0.012 bpb on the full validation set, many times the seed std). The hybrid's enwik8 advantage does **not** transfer to this dataset at this budget. Both ALiBi models extrapolate stably to 2×/4× while RoPE GPT degrades sharply, the same pattern as on enwik8.

## 2. Length extrapolation to 16× the training length (Reviewer item 7)

Seed 0, trained at ctx 1024 on enwik8 (1,500 steps), evaluated zero-shot. Each length is scored on the **same 512 KB** of validation text (512×1k, 256×2k, 128×4k, 64×8k, 32×16k windows), so the numbers are directly comparable across lengths.

| Model | L=1024 | L=2048 | L=4096 | L=8192 | L=16384 | Δ(16k − 1k) |
|---|---|---|---|---|---|---|
| ALiBi-DEEPA 3:1 (h6) | 1.5000 | 1.4794 | 1.4687 | 1.4636 | 1.4605 | -0.0395 |
| Pure ALiBi | 1.5329 | 1.5060 | 1.4917 | 1.4850 | 1.4814 | -0.0515 |
| RoPE GPT | 1.5763 | 2.4197 | 3.5866 | 4.6504 | 5.4797 | +3.9034 |

Per-position bpb inside a 16,384-byte window (mean over 32 windows, 1,024-position buckets; positions > 1,024 were never seen in training):

| Model | 0k–1k | 1k–2k | 2k–3k | 3k–4k | 4k–5k | 5k–6k | 6k–7k | 7k–8k | 8k–9k | 9k–10k | 10k–11k | 11k–12k | 12k–13k | 13k–14k | 14k–15k | 15k–16k |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ALiBi-DEEPA 3:1 (h6) | 1.46 | 1.39 | 1.48 | 1.50 | 1.47 | 1.40 | 1.40 | 1.45 | 1.47 | 1.55 | 1.51 | 1.44 | 1.44 | 1.43 | 1.46 | 1.52 |
| Pure ALiBi | 1.48 | 1.42 | 1.50 | 1.53 | 1.50 | 1.42 | 1.41 | 1.47 | 1.49 | 1.58 | 1.54 | 1.46 | 1.46 | 1.43 | 1.47 | 1.53 |
| RoPE GPT | 1.52 | 3.16 | 4.55 | 5.05 | 5.46 | 5.63 | 5.87 | 5.85 | 5.98 | 6.09 | 6.22 | 6.38 | 6.52 | 6.45 | 6.38 | 6.56 |

**Finding:** both ALiBi models keep improving as context grows (no degradation out to 16k), and the hybrid leads pure ALiBi by 0.020–0.033 bpb at every length (single seed). RoPE GPT collapses immediately past its training length (bpb 1.58 → 5.48), and the per-position profile shows the collapse begins exactly at position 1,024. Eval peak VRAM at 16k (B=1): ALiBi-DEEPA 3:1 (h6) 1,423 MB, Pure ALiBi 1,420 MB, RoPE GPT 353 MB.

## 3. Parameter scaling at a matched 5 tokens/parameter (Reviewer item 6)

Seed 0. Both models at a scale get the identical step count (5 × hybrid params / 16,384 tokens per step). Train = enwik8 minus the validation slice (99 MB, ≤ 2 epochs); val = the paper's standard 1 MB slice. LR 2e-3 for ≤ 14M; at ~38M each model uses its best LR from {1e-3, 2e-3} (1,000-step sweep).

| Scale | dim / layers / attn heads / mixer heads | Model | Params | Steps | Tokens | LR | bpb @1024 (64 KB) | @2048 | @4096 | bpb full val | tok/s | peak MB | grad ckpt |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2M | 192/4/4/3 | ALiBi-DEEPA 3:1 (h6) | 1.85M | 563 | 9.2M | 0.002 | 1.7638 | 1.7554 | 1.7979 | 2.1117 | 392,707 | 703 | False |
| 2M | 192/4/4/– | Pure ALiBi | 1.82M | 563 | 9.2M | 0.002 | 2.0407 | 2.0278 | 2.0708 | 2.4094 | 402,245 | 586 | False |
| 6M | 256/8/4/4 | ALiBi-DEEPA 3:1 (h6) | 6.44M | 1,966 | 32.2M | 0.002 | 1.0523 | 1.0282 | 1.1010 | 1.5983 | 148,486 | 1,773 | False |
| 6M | 256/8/4/– | Pure ALiBi | 6.36M | 1,966 | 32.2M | 0.002 | 1.0749 | 1.0436 | 1.1138 | 1.6196 | 164,210 | 1,444 | False |
| 14M | 384/8/8/6 | ALiBi-DEEPA 3:1 (h6) | 14.44M | 4,406 | 72.2M | 0.002 | 0.8765 | 0.8534 | 0.9130 | 1.3938 | 80,516 | 2,719 | False |
| 14M | 384/8/8/– | Pure ALiBi | 14.26M | 4,406 | 72.2M | 0.002 | 0.9058 | 0.8757 | 0.9344 | 1.4301 | 85,494 | 2,241 | False |
| 38M | 512/12/8/8 | ALiBi-DEEPA 3:1 (h6) | 38.34M | 11,701 | 191.7M | 0.002 | 0.7304 | 0.7085 | 0.7567 | 1.2349 | 14,948 | 1,634 | True |
| 38M | 512/12/8/– | Pure ALiBi | 37.89M | 11,701 | 191.7M | 0.002 | 0.7523 | 0.7313 | 0.7773 | 1.2624 | 43,091 | 4,480 | False |

Hybrid advantage (pure − hybrid, positive = hybrid better):

| Scale | Δ bpb @1024 (64 KB) | Δ bpb full val |
|---|---|---|
| 2M | +0.2769 | +0.2977 |
| 6M | +0.0226 | +0.0213 |
| 14M | +0.0293 | +0.0362 |
| 38M | +0.0218 | +0.0275 |

**Finding:** the hybrid beats pure ALiBi at all four scales (1.8M → 38M) at matched tokens, by 0.021–0.036 bpb on the full validation set from 6M upward; the margin does not shrink with scale. The 2M gap (~0.28–0.30) is an outlier: at only 563 steps pure ALiBi has barely left the early plateau, so it mostly reflects the hybrid's faster early learning, not a large-model trend. Caveats: one seed per cell; at 38M the hybrid's B=16 step exceeded the 7 GB budget and ran with activation checkpointing while pure ALiBi did not, so the 38M tok/s column is **not** a like-for-like speed comparison (quality is unaffected by checkpointing).


38M LR sweep (1,000 steps, seed 100, bpb @1024): alibi-hybrid-h6 lr=0.001: 1.1230; alibi-hybrid-h6 lr=0.002: 1.1143; alibi-pure lr=0.001: 1.1649; alibi-pure lr=0.002: 1.1382

## 4. Computational overhead and memory vs. sequence length (Reviewer items 5, 8)

14.4M models, B=1, bf16, torch.compile, AdamW. Median of 10 timed steps after 3 warm-up steps. `train` = full step (fwd+bwd+clip+AdamW); `prefill` = no-grad forward over the sequence.

**Training step, no activation checkpointing** (ms / peak MB):

| Model | T=1024 | T=2048 | T=4096 | T=8192 | T=16384 |
|---|---|---|---|---|---|
| ALiBi-DEEPA 3:1 (h6) | 22.8 ms / 415 MB | 36.1 ms / 669 MB | 74.7 ms / 1,376 MB | 181.2 ms / 1,529 MB | 530.6 ms / 2,933 MB |
| Pure ALiBi | 21.1 ms / 472 MB | 48.5 ms / 995 MB | 135.0 ms / 2,847 MB | 404.4 ms / 2,068 MB | 1475.9 ms / 4,595 MB |
| RoPE GPT | 18.3 ms / 359 MB | 31.1 ms / 495 MB | 62.5 ms / 775 MB | 150.9 ms / 1,324 MB | 440.4 ms / 2,409 MB |

**Training step, per-block activation checkpointing** (ms / peak MB):

| Model | T=1024 | T=2048 | T=4096 | T=8192 | T=16384 |
|---|---|---|---|---|---|
| ALiBi-DEEPA 3:1 (h6) | 59.8 ms / 316 MB | 78.9 ms / 544 MB | 205.2 ms / 1,445 MB | 568.3 ms / 951 MB | 1936.5 ms / 1,702 MB |
| Pure ALiBi | 25.6 ms / 309 MB | 64.1 ms / 495 MB | 179.1 ms / 1,331 MB | 416.5 ms / 1,580 MB | 1506.2 ms / 3,321 MB |
| RoPE GPT | 18.6 ms / 310 MB | 32.3 ms / 311 MB | 64.4 ms / 385 MB | 171.5 ms / 541 MB | 525.7 ms / 873 MB |

**Inference prefill (forward only)** (ms / peak MB):

| Model | T=1024 | T=2048 | T=4096 | T=8192 | T=16384 |
|---|---|---|---|---|---|
| ALiBi-DEEPA 3:1 (h6) | 5.0 ms / 97 MB | 6.4 ms / 153 MB | 22.3 ms / 366 MB | 52.6 ms / 254 MB | 136.8 ms / 432 MB |
| Pure ALiBi | 4.5 ms / 105 MB | 15.2 ms / 167 MB | 44.6 ms / 390 MB | 102.5 ms / 776 MB | 366.2 ms / 2,498 MB |
| RoPE GPT | 3.6 ms / 93 MB | 8.6 ms / 108 MB | 18.7 ms / 133 MB | 46.4 ms / 186 MB | 135.0 ms / 294 MB |

**Speed ratio vs. hybrid** (other model's ms ÷ hybrid ms; > 1 = hybrid faster), training without checkpointing:

| Baseline | T=1024 | T=2048 | T=4096 | T=8192 | T=16384 |
|---|---|---|---|---|---|
| Pure ALiBi | 0.93× | 1.34× | 1.81× | 2.23× | 2.78× |
| RoPE GPT | 0.80× | 0.86× | 0.84× | 0.83× | 0.83× |

**Analytical cost** (per model; inference cache = per-sequence state that must be kept during autoregressive decoding, bf16):

| Model | Params | Weights bf16 | Train state fp32+Adam | Attn layers | SheafDelta layers | Cache @1024 | Cache @4096 | Cache @16384 | Cache @65536 |
|---|---|---|---|---|---|---|---|---|---|
| ALiBi-DEEPA 3:1 (h6) | 14.44M | 28.9 MB | 231.0 MB | 2 | 6 | 3.51 MB | 12.95 MB | 50.70 MB | 201.70 MB |
| Pure ALiBi | 14.26M | 28.5 MB | 228.2 MB | 8 | 0 | 12.58 MB | 50.33 MB | 201.33 MB | 805.31 MB |
| RoPE GPT | 14.26M | 28.5 MB | 228.2 MB | 8 | 0 | 12.58 MB | 50.33 MB | 201.33 MB | 805.31 MB |

Cache formula: attention layer = 2·T·D values (K and V); SheafDelta layer = H·d_h² (matrix memory, 6·64·64) + 16·D (dyadic ring buffer) = 30,720 values, independent of T. The hybrid's cache is 6 × 30,720 + 2 × 2·T·384, i.e. ≈ ¼ of a full-attention model's at every length (the remaining growth comes from its 2 attention layers).

**Findings:** (i) Against pure ALiBi (the architecture it is meant to improve on), the hybrid's training step is 8% slower at 1k but 1.3×–2.8× faster from 2k to 16k (the gap widens with length), with 12–52% lower peak memory at every length; prefill is likewise 10% slower at 1k and 2.0–2.7× faster from 2k to 16k. (ii) RoPE GPT on FlashAttention is the fastest model at every length. The hybrid's speed advantage is over ALiBi's explicit-bias attention, which cannot use the FlashAttention kernel, not over attention in general; RoPE GPT, however, cannot extrapolate (Section 2). (iii) With activation checkpointing, the hybrid's step is 2.6–3.7× slower than without, a much larger recompute penalty than the attention models pay (1.0–1.3×); at B=1 the fused delta-rule kernel's recompute cost dominates. Checkpointing is therefore only worth it for the hybrid when memory is the binding constraint.

## Implementation notes

- Scripts: `benchmarks/gpu_wikitext103_3seed.py`, `gpu_extrap_16k.py`, `gpu_scaling_law.py`, `gpu_overhead_profile.py`; shared code in `benchmarks/deepa_common.py`; data prep `benchmarks/data_prep/build_wikitext103_bytes.py`; this file from `benchmarks/summarize_phaseA.py`.
- Model classes and the training recipe are copied from `gpu_splitwd_3seed.py` (param counts reproduce exactly: 14,436,348 / 14,260,608).
- ALiBi beyond 4,096 tokens uses query-chunked attention (1,024-row blocks, bias built per block, K/V sliced to the causal prefix). It is mathematically identical to the dense-bias path (max abs diff 0.0 at T = 2,048 and 4,096) and needed because the dense fp32 [H,T,T] bias is 8.6 GB at T = 16,384. FlexAttention was not usable (torch 2.6 rejects head dim 48).
- Single-seed rows (extrapolation, scaling) carry seed noise of about ±0.002–0.005 bpb (see Section 1 stds); differences below ~0.01 should not be over-interpreted.
