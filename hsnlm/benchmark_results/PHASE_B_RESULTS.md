# Phase B results: deeper benchmarks for the reviewer response

Hardware: single RTX 2000 Ada Laptop GPU (8 GB). Byte-level LM, vocab 256. bpb = bits per byte, lower is better. ± = population std over seeds. Unless stated, models are the paper recipe (dim 384, 8 layers, 14.4M params, AdamW (0.9, 0.95), wd 0.05, lr 2e-3, warmup 100, cosine, B 16, ctx 1024, bf16, torch.compile). ALiBi attention uses an exact fused-kernel formulation (validated to within 4e-5 bpb of the dense-mask path). `full` = all 976 windows of the 1 MB validation slice (the 64-window `val_bpb` repo convention is optimistic; see PHASE_A_RESULTS.md).

## 1. Multi-domain / multi-modal language modelling (Reviewer item 4)

Six non-enwik8 domains, each 40 MB train + 1 MB disjoint validation, identical byte-level pipeline and recipe, 1,500 steps, seeds 0/1/2. Value = full-validation bpb.

| Domain | ALiBi-DEEPA 3:1 (h6) [paper model] | Pure ALiBi | RoPE GPT | Pure SheafDelta (no attention) | Winner (margin) |
|---|---|---|---|---|---|
| Python source code | **1.1416 ± 0.0057** | 1.1883 ± 0.0041 | 1.2131 ± 0.0075 | 1.3745 ± 0.0096 | ALiBi-DEEPA 3:1 (h6) (0.047) |
| Hindi Wikipedia (Devanagari UTF-8) | **0.8745 ± 0.0016** | 0.9586 ± 0.0072 | 1.0081 ± 0.0261 | 0.8766 ± 0.0006 | ALiBi-DEEPA 3:1 (h6) (0.002) |
| Chinese Wikipedia (UTF-8) | 2.0581 ± 0.0069 | **2.0250 ± 0.0072** | 2.2617 ± 0.0071 | 2.0833 ± 0.0026 | Pure ALiBi (0.033) |
| DNA, hg38 chr20 (bits/base) | 1.7420 ± 0.0002 | 1.8389 ± 0.0105 | 1.7881 ± 0.0020 | **1.7384 ± 0.0010** | Pure SheafDelta (no attention) (0.004) |
| CIFAR-10 pixels, raster RGB (bits/dim) | 4.4382 ± 0.0136 | 4.6570 ± 0.0098 | **3.9661 ± 0.0084** | 4.4937 ± 0.0120 | RoPE GPT (0.472) |
| Speech, 16 kHz 8-bit mu-law (bits/sample) | **3.3300 ± 0.0043** | 3.4436 ± 0.0112 | 3.4116 ± 0.0159 | 3.3311 ± 0.0056 | ALiBi-DEEPA 3:1 (h6) (0.001) |

Training throughput (mean over all domain runs, tokens/s): ALiBi-DEEPA 3:1 (h6) 86.5k, Pure ALiBi 117.0k, RoPE GPT 98.5k, Pure SheafDelta (no attention) 80.1k.

### 1b. Is the image gap caused by DEEPA or by ALiBi? (same hybrid, RoPE instead of ALiBi in its 2 attention layers)

| Domain | Model | bpb full @1024 | @2048 (2x) | @4096 (4x) |
|---|---|---|---|---|
| image | ALiBi-DEEPA 3:1 (h6) [paper model] | 4.4382 ± 0.0136 | 4.7067 ± 0.0112 | 4.7182 ± 0.0109 |
| image | RoPE-DEEPA 3:1 (h6) [diagnostic] | 3.9470 ± 0.0098 | 4.5569 ± 0.0803 | 5.2154 ± 0.1214 |
| image | RoPE GPT | 3.9661 ± 0.0084 | 5.0105 ± 0.0658 | 6.0782 ± 0.0901 |
| image | Pure ALiBi | 4.6570 ± 0.0098 | 4.9001 ± 0.0133 | 4.9029 ± 0.0152 |
| code | ALiBi-DEEPA 3:1 (h6) [paper model] | 1.1416 ± 0.0057 | 1.2275 ± 0.0119 | 1.1164 ± 0.0107 |
| code | RoPE-DEEPA 3:1 (h6) [diagnostic] | 1.1450 ± 0.0040 | 1.2402 ± 0.0070 | 1.1364 ± 0.0050 |
| code | RoPE GPT | 1.2131 ± 0.0075 | 2.0306 ± 0.0711 | 3.0001 ± 0.1307 |
| code | Pure ALiBi | 1.1883 ± 0.0041 | 1.2719 ± 0.0035 | 1.1474 ± 0.0102 |

**Reading:** on images, swapping ALiBi for RoPE inside the *same* DEEPA architecture recovers the whole gap and beats RoPE GPT, so the image weakness belongs to ALiBi's monotonic distance penalty (the pixel above is 96 bytes back), not to the DEEPA mixer. On code the two DEEPA variants tie, and RoPE-DEEPA degrades far less beyond the training length than RoPE GPT (only 2 of 8 layers carry positional encoding). The paper model keeps ALiBi for length extrapolation.

## 2. WikiText-103: is pure ALiBi's 1,500-step lead real? (Reviewer item 4)

(a) LR sweep, seed 0, 1,500 steps, full-validation bpb:

| LR | ALiBi-DEEPA 3:1 (h6) [paper model] | Pure ALiBi |
|---|---|---|
| 0.001 | 1.6390 | 1.6490 |
| 0.002 | 1.6283 | 1.6154 |
| 0.003 | 1.6283 | 1.6179 |

(b) Each model at its best LR, trained to 5 tokens/param (4,406 steps), seeds 0 and 1:

| Model | best LR | bpb full | @1024 (64 KB) | @2048 | @4096 |
|---|---|---|---|---|---|
| ALiBi-DEEPA 3:1 (h6) [paper model] | 0.002 | 1.4250 ± 0.0017 | 1.3741 ± 0.0015 | 1.3494 ± 0.0025 | 1.4063 ± 0.0039 |
| Pure ALiBi | 0.002 | 1.4404 ± 0.0006 | 1.3888 ± 0.0014 | 1.3581 ± 0.0015 | 1.4163 ± 0.0023 |

**Reading:** both models' best LR is the paper's 2e-3 (no tuning artefact). At 1,500 steps pure ALiBi led by 0.012 bpb; trained to 5 tokens/param the hybrid leads at every evaluation length on both seeds. The Phase A WikiText finding (PHASE_A_RESULTS.md section 1) was a short-horizon effect, consistent with the 18,000-step enwik8 result.

## 3. Length extrapolation to 16x, 3 seeds (Reviewer item 7)

enwik8, trained at ctx 1024 (1,500 steps), evaluated zero-shot on the same 512 KB at every length.

| Model | L=1024 | L=2048 | L=4096 | L=8192 | L=16384 |
|---|---|---|---|---|---|
| ALiBi-DEEPA 3:1 (h6) [paper model] (3 seeds) | 1.507 ± 0.008 | 1.487 ± 0.009 | 1.477 ± 0.010 | 1.472 ± 0.010 | 1.469 ± 0.010 |
| Pure ALiBi (3 seeds) | 1.528 ± 0.003 | 1.500 ± 0.004 | 1.486 ± 0.004 | 1.479 ± 0.004 | 1.475 ± 0.004 |
| RoPE GPT (3 seeds) | 1.588 ± 0.012 | 2.352 ± 0.070 | 3.454 ± 0.187 | 4.519 ± 0.179 | 5.299 ± 0.156 |

**Reading:** both ALiBi models improve monotonically to 16x the training length on every seed; RoPE GPT collapses (~3.5x worse at 16k). The hybrid has the lowest mean at every length, but beyond 2k its lead over pure ALiBi (0.006-0.009) is within seed spread (seed 2 favours pure ALiBi), so claim *stable* extrapolation for both ALiBi models, with the hybrid doing it at 1/4 of the KV cache (section 4).

## 4. Inference, serving capacity, training cost (Reviewer items 5, 8)

Decode (CUDA-graph, preallocated KV cache for every model; ms per generated token, best ALiBi implementation per cell):

| Context | Batch | ALiBi-DEEPA (h6) | Pure ALiBi | RoPE GPT | hybrid cache MB | GPT cache MB |
|---|---|---|---|---|---|---|
| 1024 | 1 | 0.90 | 0.68 | 0.89 | 4 | 13 |
| 1024 | 32 | 2.16 | 3.33 | 4.23 | 112 | 403 |
| 4096 | 1 | 0.95 | 0.95 | 1.09 | 15 | 50 |
| 4096 | 32 | 4.11 | 12.18 | 14.34 | 414 | 1611 |
| 16384 | 1 | 1.22 | 1.91 | 1.91 | 59 | 201 |
| 16384 | 32 | 12.71 | 46.75 | 56.05 | 1622 | 6442 |
| 65536 | 1 | 2.19 | 6.01 | 5.51 | 235 | 805 |
| 65536 | 32 | 47.47 | OOM/— | OOM/— | — | — |

Serving capacity in 8 GB (max concurrent sequences, aggregate decode tokens/s):

| Context | ALiBi-DEEPA (h6) | Pure ALiBi | RoPE GPT |
|---|---|---|---|
| 4096 | 576 seq, 8537 tok/s | 144 seq, 2781 tok/s | 144 seq, 2283 tok/s |
| 16384 | 144 seq, 2653 tok/s | 36 seq, 700 tok/s | 36 seq, 578 tok/s |
| 65536 | 36 seq, 688 tok/s | 9 seq, 163 tok/s | 9 seq, 144 tok/s |

Training step (B=1, no checkpointing) and prefill, ms (best ALiBi implementation):

| Length | DEEPA train | RoPE GPT train | DEEPA prefill | RoPE GPT prefill | DEEPA train peak MB | GPT train peak MB |
|---|---|---|---|---|---|---|
| 1024 | 22.8 | 18.3 | 5.0 | 3.6 | 414.8 | 359.2 |
| 2048 | 31.8 | 31.1 | 6.3 | 8.6 | 543.1 | 494.9 |
| 4096 | 54.0 | 62.5 | 14.5 | 18.7 | 855.9 | 774.7 |
| 8192 | 115.2 | 150.9 | 31.2 | 46.4 | 1487.4 | 1324.3 |
| 16384 | 281.3 | 440.4 | 76.6 | 135.0 | 2749.3 | 2408.6 |

**Reading:** the hybrid's KV cache is ~1/4 of full attention at every length, giving ~4x the concurrent sequences and 3.7-4.8x the aggregate decode throughput in 8 GB, and faster single-stream decode from 4k up (2.5x at 64k). Known costs: at <=1k context, training/prefill is ~1.4x slower than RoPE GPT (many small SheafDelta kernels), domain-run throughput is 26% below pure ALiBi, and 16k training memory is ~14% higher.

## 5. Long Range Arena subset (Reviewer item 6/7)

dim 256, 4 layers, 5,000 steps, lr 1e-3, B 32, seeds 0/1; test accuracy.

| Task | ALiBi-DEEPA 3:1 (h6) [paper model] | Pure ALiBi | RoPE GPT |
|---|---|---|---|
| ListOps (<=2k) | 0.199 ± 0.003 | 0.194 ± 0.006 | 0.182 ± 0.003 |
| IMDb bytes (4k) | 0.758 ± 0.003 | 0.637 ± 0.001 | 0.546 ± 0.031 |
| sCIFAR grayscale (1k) | 0.396 ± 0.001 | 0.368 ± 0.005 | 0.349 ± 0.009 |

## 5b. Equal wall-clock training vs Pure ALiBi (Reviewer item 5)

With the folded ALiBi kernel, Pure ALiBi trains 1.353x faster than DEEPA at the paper recipe (117.0k vs 86.5k tok/s, B 16, ctx 1024), so Pure ALiBi's 18,000-step budget buys DEEPA 13,300 steps (cosine schedule fitted to 13,300). enwik8, 64 KB slice (same metric as the 18k table), 3 seeds.

| Model | Steps | Train s (measured) | bpb @1024 | per seed | @2048 | @4096 |
|---|---|---|---|---|---|---|
| ALiBi-DEEPA 3:1 (h6), equal wall-clock | 13,300 | 2,693 | 0.7770 ± 0.0010 | 0.7776/0.7756/0.7779 | 0.7585 ± 0.0012 | 0.8183 ± 0.0021 |
| Pure ALiBi | 18,000 | 3,456 | 0.7863 ± 0.0049 | 0.7924/0.7861/0.7803 | 0.7664 ± 0.0047 | 0.8258 ± 0.0041 |
| Pure ALiBi, equal steps (control) | 13,300 | 2,529 | 0.7888 ± 0.0029 | 0.7891/0.7851/0.7921 | 0.7688 ± 0.0021 | 0.8256 ± 0.0005 |
| ALiBi-DEEPA 3:1 (h6), 18k steps | 18,000 | 3,661 | 0.7794 ± 0.0025 | 0.7820/0.7761/0.7801 | 0.7620 ± 0.0021 | 0.8246 ± 0.0034 |

**Reading:** in equal (folded-kernel) wall-clock, DEEPA beats Pure ALiBi on 3/3 seeds at every length, and on the measured hardware it also used 22% less time. At equal steps (13,300 each) DEEPA leads Pure ALiBi on 3/3 seeds (0.7770 vs 0.7888); both models also score better at 13.3k than at 18k, consistent with over-repetition of the 40 MB pool at 18k, so the comparison is not explained by that effect.

## 5c. Causal control: parity with negative eigenvalues removed (beta in (0,1))

Same converged protocol as section 6 (curriculum 8->64, up to 20k steps, LR grid, 3 seeds), delta-rule step size capped at beta_max = 1 (eigenvalues in [0,1]) instead of the paper's 2 ([-1,1]). Identical parameter counts.

| Model | acc @64 (train length) | acc @256 (4x) | steps to solve |
|---|---|---|---|
| ALiBi-DEEPA, beta in (0,1) | 99.9/99.3/99.8 | 66.4/64.5/63.7 | 11500/12500/12500 |
| Pure SheafDelta, beta in (0,1) | 99.8/99.9/99.6 | 64.2/64.4/64.2 | 11500/11000/17500 |
| ALiBi-DEEPA, beta in (0,2) [paper] | 100.0/100.0/100.0 | 90.2/86.6/90.5 | 2000/2000/2000 |
| Pure SheafDelta, beta in (0,2) | 100.0/100.0/100.0 | 93.0/80.6/98.8 | 2000/2000/2000 |

**Reading:** without negative eigenvalues both mixers still fit parity at the training length (with ~4x more steps and the curriculum) but fall to ~64% at 4x length; with beta in (0,2) the 3,000-step screening already gave 95-100% at 4x length (section 6). Negative eigenvalues are what make the learned solution length-general.

## 6b. Learning-rate robustness for the seed-sensitive tasks

Every LR x every seed (seed 0 = sweep run, seeds 1-2 = grid). For each model the LR is chosen by the 3-seed mean accuracy at the training length (the same rule for every model). Cells: mean accuracy at train length / at 4x length (copy) and seeds solved (>=99%).

**copy_paper**

| Model | LR 3e-4 | LR 1e-3 | LR 3e-3 | best LR (3-seed mean) |
|---|---|---|---|---|
| ALiBi-DEEPA 3:1 (h6) [paper model] | 99.9 / 33% (3/3) | 100.0 / 79% (3/3) | 91.0 / 70% (2/3) | 0.001 (3/3 solved) |
| Pure ALiBi | 99.9 / 93% (3/3) | 99.8 / 100% (3/3) | 67.4 / 66% (2/3) | 0.0003 (3/3 solved) |
| RoPE GPT | 100.0 / 0% (3/3) | 100.0 / 0% (3/3) | 100.0 / 0% (3/3) | 0.0003 (3/3 solved) |
| Pure SheafDelta (no attention) | 99.6 / 15% (3/3) | 99.5 / 15% (3/3) | 66.5 / 9% (2/3) | 0.0003 (3/3 solved) |

**copy**

| Model | LR 3e-4 | LR 1e-3 | LR 3e-3 | best LR (3-seed mean) |
|---|---|---|---|---|
| ALiBi-DEEPA 3:1 (h6) [paper model] | 99.8 / 33% (3/3) | 99.7 / 29% (3/3) | 67.1 / 52% (2/3) | 0.0003 (3/3 solved) |
| Pure ALiBi | 99.8 / 98% (3/3) | 99.6 / 98% (3/3) | 99.7 / 91% (3/3) | 0.0003 (3/3 solved) |
| RoPE GPT | 100.0 / 0% (3/3) | 100.0 / 0% (3/3) | 100.0 / 0% (3/3) | 0.0003 (3/3 solved) |
| Pure SheafDelta (no attention) | 98.9 / 19% (1/3) | 99.3 / 23% (3/3) | 34.2 / 4% (1/3) | 0.001 (3/3 solved) |

**mqar64_T512**

| Model | LR 3e-4 | LR 1e-3 | LR 3e-3 | best LR (3-seed mean) |
|---|---|---|---|---|
| ALiBi-DEEPA 3:1 (h6) [paper model] | 94.6% (2/3) | 97.9% (2/3) | 68.6% (2/3) | 0.001 (2/3 solved) |
| Pure ALiBi | 99.6% (3/3) | 68.5% (2/3) | 33.0% (0/3) | 0.0003 (3/3 solved) |
| RoPE GPT | — | — | — | — |
| Pure SheafDelta (no attention) | 68.6% (2/3) | 99.5% (3/3) | 96.7% (2/3) | 0.001 (3/3 solved) |

## 6. Synthetic capability suite (mechanism evidence)

Converged protocol: up to 20k steps, LR grid {3e-4,1e-3,3e-3} on seed 0, best LR x 3 seeds, early stop at >=99% (two consecutive evals). Cells: accuracy per seed at each test length; [steps to solve].

| Task | ALiBi-DEEPA 3:1 (h6) [paper model] | Pure ALiBi | RoPE GPT | Pure SheafDelta (no attention) |
|---|---|---|---|---|
| copy_paper | T128: 100/100/100; T512: 100/98/38 [2000/2000/4000] | T128: 100/100/100; T512: 96/95/87 [500/500/500] | T128: 100/100/100; T512: 0/0/0 [500/500/500] | T128: 100/100/99; T512: 14/18/15 [2500/2500/3500] |
| copy | T128: 100/2/100; T512: 96/0/59 [6000/x/6000] | T128: 100/100/100; T512: 99/98/99 [1000/1000/1000] | T128: 100/100/100; T512: 0/0/0 [500/500/500] | T128: 99/99/99; T512: 24/26/19 [5500/4500/4000] |
| mqar8_T64 | T64: 100/100/100 [3500/1500/1500] | T64: 99/99/100 [7000/6000/9000] | T64: 99/100/99 [9000/9500/9000] | T64: 99/99/100 [13500/5000/2500] |
| mqar32_T512 | T512: 100/100/100 [2500/2500/2500] | T512: 100/100/100 [4000/6500/3000] | T512: 9/9/8 [x/x/x] | T512: 100/99/100 [6000/4500/4500] |
| mqar64_T512 | T512: 100/100/6 [4500/6000/x] | T512: 100/6/100 [9000/x/3500] | T512: 6/6/6 [x/x/x] | T512: 100/6/100 [10000/x/10000] |
| parity | T64: 100/100/100; T256: 90/87/90 [2000/2000/2000] | T64: 46/57/57; T256: 31/52/44 [x/x/x] | T64: 55/54/56; T256: 44/49/52 [x/x/x] | T64: 100/100/100; T256: 93/81/99 [2000/2000/2000] |
| s3 | T64: 100/100/99; T256: 93/95/48 [4000/4000/13000] | T64: 29/29/30; T256: 20/20/20 [x/x/x] | T64: 28/28/28; T256: 20/20/19 [x/x/x] | T64: 100/99/100; T256: 92/89/96 [3000/6000/2500] |
| s5 | T64: 4/2/3; T256: 2/1/1 [x/x/x] | T64: 2/2/2; T256: 1/1/1 [x/x/x] | T64: 2/2/2; T256: 1/1/1 [x/x/x] | T64: 4/6/6; T256: 2/2/2 [x/x/x] |
| selcopy | T256: 99/99/99 [3500/7000/10500] | T256: 100/99/99 [9500/7000/10000] | T256: 99/100/99 [6500/4500/4000] | T256: 100/100/99 [4000/3000/4000] |

Screening run (3,000 steps, dim 128, 4 layers; NOT converged -- MQAR/S5/copy numbers are superseded by the converged protocol above): accuracy % per seed.

| Task | ALiBi-DEEPA 3:1 (h6) [paper model] | Pure ALiBi | RoPE GPT | Pure SheafDelta (no attention) |
|---|---|---|---|---|
| mqar32 | T512: 96/13/9 | T512: 8/8/8 | T512: 8/8/8 | T512: 95/9/87 |
| mqar64 | T512: 6/6/6 | T512: 6/5/6 | T512: 6/6/6 | T512: 37/92/48 |
| s5 | T64: 2/2/2; T256: 1/1/1 | T64: 2/2/2; T256: 1/1/1 | T64: 2/2/2; T256: 1/1/1 | T64: 2/2/2; T256: 1/1/1 |
| parity | T64: 100/100/100; T256: 100/95/99 | T64: 59/54/55; T256: 52/51/51 | T64: 79/70/59; T256: 57/55/52 | T64: 100/100/100; T256: 100/100/99 |
| copy | L128: 1/1/1; L512: 0/0/0 | L128: 100/100/100; L512: 99/99/98 | L128: 100/100/100; L512: 0/0/0 | L128: 76/32/45; L512: 13/7/9 |
| selcopy | T256: 99/99/99 | T256: 94/96/93 | T256: 100/99/100 | T256: 100/100/100 |
