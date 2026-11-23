# Benchmarks

All scripts run from the `hsnlm/` root (`python benchmarks/<script>.py`). Results go to `benchmark_results/<group>/<benchmark>/`; each folder has a README with the exact command and the results.

## Complete Master Results Table (All Benchmarks)

| # | Category | Benchmark / Task | Setup (Seeds / Steps) | ALiBi-DEEPA 3:1 (`h6`) | Pure ALiBi | RoPE GPT | Pure SheafDelta | Outcome / Takeaway |
|---|---|---|---|---|---|---|---|---|
| **1** | **enwik8 Convergence** | Chinchilla-ratio `enwik8` (64K val) | 3s / 18,000 (20.4 tok/p) | **`0.7794 ± .0025`** (`@2k: 0.7620`, `@4k: 0.8246`) | `0.7863 ± .0049` (`@2k: 0.7664`, `@4k: 0.8258`) | — | — | **DEEPA wins `3/3` seeds** (`-0.0069 bpb`, 2× tighter std) |
| **2** | **enwik8 Convergence** | Equal Wall-Clock & Equal-Steps `enwik8` | 3s / 13,300 (iso-time vs 18k) | **`0.7770 ± .0010`** (`@2k: 0.7585`, `@4k: 0.8183`) | `0.7888 ± .0029` (13.3k) / `0.7863` (18k) | — | — | **DEEPA wins `3/3` seeds** at both equal time (`-0.0093`) & equal steps (`-0.0118`) |
| **3** | **enwik8 Optimizer** | Uniform vs. Split Weight Decay | 3s / 1,500 (`enwik8`) | Uniform: **`1.0291 ± .0016`**; Split: `1.0324 ± .0032` | Uniform: `1.0369 ± .0012`; Split: `1.0407 ± .0091` | — | — | **Uniform WD wins**; DEEPA margin invariant (`-0.0078` vs `-0.0083`) |
| **4** | **enwik8 Efficiency** | Iso-Quality vs. Compiled RoPE GPT | 1s / 1,500 (`enwik8`) | **`1.0305`** (reaches GPT loss in **864 steps / 175.1s**) | `1.0369` (3s mean) | `1.1014` (`251.4s`) | — | **DEEPA uses `42.4%` fewer tokens & `30.3%` less time** than RoPE GPT |
| **5** | **Parameter Scaling** | Scaling Law `1.85M → 38.34M` (Full 1MB val) | 1s / 5 tok/p (`enwik8`) | `2M: 2.112` / `6M: 1.598` / `14M: 1.394` / **`38M: 1.235`** | `2M: 2.409` / `6M: 1.620` / `14M: 1.430` / `38M: 1.262` | — | — | **DEEPA wins `4/4` scales** (`-0.298, -0.021, -0.036, -0.028 bpb`); no margin decay |
| **6** | **Extrapolation** | Zero-Shot `1k → 16k` (identical 512KB) | 3s / 1,500 (`enwik8`) | **`1.507 → 1.487 → 1.477 → 1.472 → 1.469`** | `1.528 → 1.500 → 1.486 → 1.479 → 1.475` | `1.588 → 2.352 → 3.454 → 4.519 → 5.299` | — | **Stable to `16×`** on `3/3` seeds (ties Pure ALiBi past `2k` at `4×` less KV; RoPE collapses) |
| **7** | **Multi-Domain** | Python Source Code (Full 1MB val) | 3s / 1,500 | **`1.1416 ± .0057`** (`RoPE-DEEPA: 1.1450`) | `1.1883 ± .0041` | `1.2131 ± .0075` | `1.3745 ± .0096` | **DEEPA wins** (`-0.047` vs ALiBi, `-0.072` vs RoPE, `-0.233` vs Pure SheafDelta) |
| **8** | **Multi-Domain** | Hindi Wikipedia UTF-8 (Full 1MB val) | 3s / 1,500 | **`0.8745 ± .0016`** | `0.9586 ± .0072` | `1.0081 ± .0261` | `0.8766 ± .0006` | **DEEPA wins** (`-0.084` vs ALiBi, `-0.134` vs RoPE; ties Pure SheafDelta) |
| **9** | **Multi-Domain** | Chinese Wikipedia UTF-8 (Full 1MB val) | 3s / 1,500 | `2.0581 ± .0069` | **`2.0250 ± .0072`** | `2.2617 ± .0071` | `2.0833 ± .0026` | **Pure ALiBi wins** (`-0.033`); DEEPA beats RoPE (`-0.204`) & Pure SheafDelta (`-0.025`) |
| **10** | **Multi-Domain** | DNA `hg38 chr20` (bits/base, Full 1MB) | 3s / 1,500 | `1.7420 ± .0002` | `1.8389 ± .0105` | `1.7881 ± .0020` | **`1.7384 ± .0010`** | **Pure SheafDelta wins** (`0.49 MB` const. cache); DEEPA beats ALiBi (`-0.097`) & RoPE (`-0.046`) |
| **11** | **Multi-Domain** | CIFAR-10 Pixels RGB (bpd, Full 1MB) | 3s / 1,500 | `4.4382 ± .0136` (**`RoPE-DEEPA: 3.9470`**) | `4.6570 ± .0098` | `3.9661 ± .0084` | `4.4937 ± .0120` | **RoPE-DEEPA wins (`3.947 < 3.966`)**; proves stride-96 gap is ALiBi, not DEEPA |
| **12** | **Multi-Domain** | Speech 16 kHz µ-law (bps, Full 1MB) | 3s / 1,500 | **`3.3300 ± .0043`** | `3.4436 ± .0112` | `3.4116 ± .0159` | `3.3311 ± .0056` | **DEEPA wins** (`-0.114` vs ALiBi, `-0.082` vs RoPE; ties Pure SheafDelta) |
| **13** | **Multi-Domain** | WikiText-103 (1.5k vs. 4.4k = 5 tok/p) | 3s@1.5k / 2s@4.4k | `1.5k: 1.6275` / **`4.4k: 1.4250 ± .0017`** | `1.5k: 1.6154` / `4.4k: 1.4404 ± .0006` | `1.5k: 1.7382` | — | **DEEPA wins at 4.4k** (`-0.0154 bpb`, both seeds, all lengths; 1.5k was short-horizon) |
| **14** | **Systems (Train)** | Step Latency & VRAM `1k → 16k` (B=1) | Flash-ALiBi compiled | `1k: 22.8ms` / `4k: 54.0` / **`16k: 281.3ms`** (`2,749 MB`) | `1k: 17.0` / `4k: 57.4` / `16k: 410.5ms` (`2,284 MB`) | `1k: 18.3` / `4k: 62.5` / `16k: 440.4ms` (`2,409 MB`) | — | **DEEPA `1.16–1.57×` faster than RoPE** (`1.06–1.46×` vs ALiBi) at `4k–16k`; slower at `1k` |
| **15** | **Systems (Prefill)** | Forward Prefill Latency `1k → 16k` (B=1) | Flash-ALiBi compiled | `1k: 5.0` / `2k: 6.3` / `4k: 14.5` / **`16k: 76.6ms`** | `1k: 2.8` / `2k: 5.8` / `4k: 15.6` / `16k: 125.1ms` | `1k: 3.6` / `2k: 8.6` / `4k: 18.7` / `16k: 135.0ms` | — | **DEEPA `1.37–1.76×` faster than RoPE** from `2k` up (`1.63×` vs ALiBi at `16k`) |
| **16** | **Systems (Decode)** | CUDA-Graph Decode Latency (`1k → 64k`) | B=1 / B=32 | `1k: 0.90/2.16` · `16k: 1.22/12.71` · **`64k: 2.19/47.47ms`** | `1k: 0.68/3.33` · `16k: 1.91/46.75` · `64k: 6.01/OOM` | `1k: 0.89/4.23` · `16k: 1.91/56.05` · `64k: 5.51/OOM` | — | **DEEPA `2.0–4.4×` faster at B=32** (`2.5×` at B=1, `64k`); baselines OOM at `64k, B=32` |
| **17** | **Systems (Serving)** | Max 8 GB Serving Capacity (`4k / 16k / 64k`) | CUDA-graph, prealloc KV | **`576/144/36 seq`** (**`8,537 / 2,653 / 688 tok/s`**) | `144/36/9 seq` (`2,781 / 700 / 163 tok/s`) | `144/36/9 seq` (`2,283 / 578 / 144 tok/s`) | — | **`4×` concurrent users**; **`3.7–4.8×` throughput vs RoPE** (`3.1–4.2×` vs Pure ALiBi) |
| **18** | **LRA** | Long Range Arena (ListOps / IMDb / sCIFAR) | 2s / 5k (D=256, L=4) | ListOps `19.9%` · **IMDb `75.8%`** · **sCIFAR `39.6%`** | `19.4%` · `63.7%` · `36.8%` | `18.2%` · `54.6%` · `34.9%` | — | **2 clear wins** (IMDb `+12.1%`, sCIFAR `+2.8%`), tie at chance on ListOps |
| **19** | **Synthetic (Recall)** | MQAR-8 (`T64`) / MQAR-32 (`T512`) / MQAR-64 | 3s / ≤20k (D=128, L=4) | **`100% [1.5–3.5k]`** · **`100% [2.5k]`** · `2/3 (miss 94%)` | `99–100% [6–9k]` · `100% [3–6.5k]` · **`3/3`** | `99% [9k]` · `9% [x]` · `6% [x]` | `99% [2.5–13k]` · `100% [4.5–6k]` · **`3/3`** | **DEEPA fastest on MQAR-8 & MQAR-32**; `2/3` on MQAR-64 (RoPE fails 32 & 64) |
| **20** | **Synthetic (State)** | Parity (`64→256`) & S3 / S5 Groups | 3s / ≤20k (D=128, L=4) | **Parity `100→87–90%`** · **S3 `100→93/95/48%`** · S5 `x` | Parity `x` · S3 `x` · S5 `x` | Parity `x` · S3 `x` · S5 `x` | **Parity `100→81–99%`** · **S3 `100→89–96%`** · S5 `x` | **Only Sheaf-Delta models solve Parity & S3**; all 4 fail NC1-complete S5 |
| **21** | **Causal Control** | Parity with β∈(0,1) vs. β∈(0,2) | 3s / ≤20k (same params) | β∈(0,1): `64–66% @4× [12k]` · **β∈(0,2): `87–90% [2k]`** | — | — | β∈(0,1): `64% @4× [11–17k]` · **β∈(0,2): `81–99% [2k]`** | **Causal evidence**: negative eigenvalues make parity `5.5–8.7×` faster & length-general |
| **22** | **Synthetic (Copy)** | `Copy_paper` (`128→512`), `Copy`, `SelCopy` | 3s / ≤20k + LR grid | `Copy_paper`: `3/3 (79% @4×)` · `Copy`: `3/3 (33% @4×)` · `Sel`: `99%` | **`3/3 (93–100% @4×)`** · **`3/3 (98–99% @4×)`** · `99%` | `3/3 (0% @4×)` · `3/3 (0% @4×)` · `99%` | `3/3 (15–23% @4×)` · `3/3 (19–26%)` · **`100%`** | **DEEPA solves `3/3` at trained len**, `79%` at `4×` (Pure ALiBi better & `4×` faster) |
| **23** | **Ablations (Mixer)** | v2 Mixer Ablations (`h6`, `decay`, `postnorm`, `shift`) | 3s / 1,500 (24 runs) | v1: `1.0273±.0076` · **`h6: 1.0291±.0016`** (`4.7×` tighter) | `decay: 1.0452` · `postnorm_sc: 1.0587` | `postnorm: 1.0715` · `shift: 1.0751` | — | **Adopt `h6` alone**; reject decay reinit, postnorm (both), and q,k token shift |
| **24** | **Negative Results** | Ratio `4:1`, SWA-512, NoPE, YaRN, Low-Rank MLP, Non-Backprop | `enwik8` / Shakespeare | `3:1 ALiBi-DEEPA`: **`1.027–1.029`** (`backprop: 2.202 bpc`) | `4:1: 1.042` · `SWA-512: 1.061` · `NoPE: 2.344` | `YaRN: 1.101/1.225` · `LowRank MLP: +0.034–.057` | `DAD: 2.372` · `Targets: 2.315` · `DFA: 3.128` | **All 6 alternatives falsified**; confirms `3:1` full ALiBi + standard backprop |
| **25** | **Scale / Systems** | Max VRAM Scale (`206M`), Long-Ctx `16k` Width-128 Crossover, Reversible Blocks | Exploratory GPU runs | **`206M` fits in `5.3 GB`** (ckpt); `16k` width-128: **`+9.5%` faster & `-0.28 bpb`** | — | `16k` width-128 GPT baseline | `+16.1%` faster & `-0.30 bpb`; `RevLM` O(1)-depth mem | Confirms memory scalability to `206M` on 8 GB and isolated `16k` crossover |

## Benchmark scripts

| script | benchmark(s) | results folder |
|---|---|---|
| [`cpu_suite.py`](cpu_suite.py) | Original CPU benchmark suite (HSN vs GPT) | [cpu_suite](../benchmark_results/development/cpu_suite/README.md) |
| [`data_prep/build_domains.py`](data_prep/build_domains.py) | Six extra domains (code, Hindi, Chinese, DNA, images, audio), 3 seeds | [multidomain_3seed](../benchmark_results/paper/multidomain_3seed/README.md) |
| [`data_prep/build_wikitext103_bytes.py`](data_prep/build_wikitext103_bytes.py) | WikiText-103 (byte-level), 1,500 steps, 3 seeds | [wikitext103_3seed](../benchmark_results/paper/wikitext103_3seed/README.md) |
| [`gpu_alibi_hybrid.py`](gpu_alibi_hybrid.py) | ALiBi hybrids: 3:1 vs 4:1 dilution | [alibi_hybrid_ratio](../benchmark_results/development/alibi_hybrid_ratio/README.md) |
| [`gpu_alibi_hybrid_ratio.py`](gpu_alibi_hybrid_ratio.py) | ALiBi hybrids: 3:1 vs 4:1 dilution | [alibi_hybrid_ratio](../benchmark_results/development/alibi_hybrid_ratio/README.md) |
| [`gpu_blockdecode.py`](gpu_blockdecode.py) | Block / self-speculative decoding primitive | [block_decode](../benchmark_results/development/block_decode/README.md) |
| [`gpu_chinchilla_15m.py`](gpu_chinchilla_15m.py) | Chinchilla-horizon convergence on enwik8 (18,000 steps, 3 seeds); Equal wall-clock and equal-steps control (13,300 steps, 3 seeds) | [enwik8_18k_convergence](../benchmark_results/paper/enwik8_18k_convergence/README.md), [enwik8_equal_wallclock](../benchmark_results/paper/enwik8_equal_wallclock/README.md) |
| [`gpu_credit.py`](gpu_credit.py) | Backprop-free credit assignment (DAD, DFA, targets, JVP) | [credit_assignment](../benchmark_results/development/credit_assignment/README.md) |
| [`gpu_crossover.py`](gpu_crossover.py) | Fused Triton delta-rule kernel speed crossover | [fused_kernel_crossover](../benchmark_results/development/fused_kernel_crossover/README.md) |
| [`gpu_decode.py`](gpu_decode.py) | Early decode benchmarks (torch.cat KV cache; per-layer cost) | [decode_early](../benchmark_results/development/decode_early/README.md) |
| [`gpu_decode_fair.py`](gpu_decode_fair.py) | CUDA-graph decoding latency and 8 GB serving capacity | [decode_and_serving](../benchmark_results/paper/decode_and_serving/README.md) |
| [`gpu_decode_hybrid_recheck.py`](gpu_decode_hybrid_recheck.py) | Early decode benchmarks (torch.cat KV cache; per-layer cost) | [decode_early](../benchmark_results/development/decode_early/README.md) |
| [`gpu_domains_3seed.py`](gpu_domains_3seed.py) | Six extra domains (code, Hindi, Chinese, DNA, images, audio), 3 seeds | [multidomain_3seed](../benchmark_results/paper/multidomain_3seed/README.md) |
| [`gpu_extrap_16k.py`](gpu_extrap_16k.py) | Zero-shot length extrapolation 1k -> 16k (3 seeds) | [length_extrapolation_16k](../benchmark_results/paper/length_extrapolation_16k/README.md) |
| [`gpu_isoquality_compiled.py`](gpu_isoquality_compiled.py) | Iso-quality: tokens / wall-clock to reach RoPE GPT's final loss | [isoquality_vs_rope](../benchmark_results/paper/isoquality_vs_rope/README.md) |
| [`gpu_lowrank_mlp.py`](gpu_lowrank_mlp.py) | Low-rank MLP factorisation | [lowrank_mlp](../benchmark_results/development/lowrank_mlp/README.md) |
| [`gpu_lra_subset.py`](gpu_lra_subset.py) | Long Range Arena subset (ListOps, IMDb bytes, sCIFAR), 2 seeds | [long_range_arena](../benchmark_results/paper/long_range_arena/README.md) |
| [`gpu_multiseed_compiled.py`](gpu_multiseed_compiled.py) | Multi-seed compiled comparison (hybrids vs GPT) | [multiseed_compiled](../benchmark_results/paper/multiseed_compiled/README.md) |
| [`gpu_overhead_profile.py`](gpu_overhead_profile.py) | Training step / prefill latency and memory vs length (1k-16k) | [training_overhead_profile](../benchmark_results/paper/training_overhead_profile/README.md) |
| [`gpu_posenc.py`](gpu_posenc.py) | NoPE / ALiBi / RoPE positional-encoding comparison | [positional_encoding](../benchmark_results/development/positional_encoding/README.md) |
| [`gpu_posenc_recheck.py`](gpu_posenc_recheck.py) | NoPE / ALiBi / RoPE positional-encoding comparison | [positional_encoding](../benchmark_results/development/positional_encoding/README.md) |
| [`gpu_ratio.py`](gpu_ratio.py) | Hybrid attention-ratio ablation (RoPE hybrids) | [attention_ratio](../benchmark_results/development/attention_ratio/README.md) |
| [`gpu_reconcile.py`](gpu_reconcile.py) | Speed and quality measured together (isolated processes) | [reconcile_speed_quality](../benchmark_results/development/reconcile_speed_quality/README.md) |
| [`gpu_reconcile_clean.py`](gpu_reconcile_clean.py) | Speed and quality measured together (isolated processes) | [reconcile_speed_quality](../benchmark_results/development/reconcile_speed_quality/README.md) |
| [`gpu_reversible.py`](gpu_reversible.py) | Reversible-block memory/speed trade-off | [reversible_blocks](../benchmark_results/development/reversible_blocks/README.md) |
| [`gpu_scale.py`](gpu_scale.py) | Largest single-GPU LM run (pre-checkpointing) | [gpu_scale](../benchmark_results/development/gpu_scale/README.md) |
| [`gpu_scale_ckpt.py`](gpu_scale_ckpt.py) | Scaling past 14M with gradient checkpointing (7:1 and 3:1) | [gpu_scale_ckpt](../benchmark_results/development/gpu_scale_ckpt/README.md) |
| [`gpu_scale_ckpt_31.py`](gpu_scale_ckpt_31.py) | Scaling past 14M with gradient checkpointing (7:1 and 3:1) | [gpu_scale_ckpt](../benchmark_results/development/gpu_scale_ckpt/README.md) |
| [`gpu_scaling_law.py`](gpu_scaling_law.py) | Parameter scaling 2M -> 38M at 5 tokens/param | [scaling_2m_to_38m](../benchmark_results/paper/scaling_2m_to_38m/README.md) |
| [`gpu_speed.py`](gpu_speed.py) | Training speed/memory vs context: Transformer vs HSN (eager) | [speed_vs_length](../benchmark_results/development/speed_vs_length/README.md) |
| [`gpu_splitwd_3seed.py`](gpu_splitwd_3seed.py) | Uniform vs split weight decay (1,500 steps, 3 seeds) | [enwik8_split_weight_decay](../benchmark_results/paper/enwik8_split_weight_decay/README.md) |
| [`gpu_suite.py`](gpu_suite.py) | GPU replica of the CPU suite | [gpu_suite](../benchmark_results/development/gpu_suite/README.md) |
| [`gpu_swa_alibi_hybrid.py`](gpu_swa_alibi_hybrid.py) | Sliding-window ALiBi hybrid (W=512) | [sliding_window_alibi](../benchmark_results/development/sliding_window_alibi/README.md) |
| [`gpu_synthetic_proper.py`](gpu_synthetic_proper.py) | Converged synthetic suite (MQAR, parity, S3, S5, copy, selective copy); Causal control: parity with beta capped to (0,1) | [synthetic_converged](../benchmark_results/paper/synthetic_converged/README.md), [parity_negative_eigenvalue_control](../benchmark_results/paper/parity_negative_eigenvalue_control/README.md) |
| [`gpu_synthetic_suite.py`](gpu_synthetic_suite.py) | Synthetic suite screening run (3,000 steps) | [synthetic_screening_3k](../benchmark_results/paper/synthetic_screening_3k/README.md) |
| [`gpu_v2_ablation.py`](gpu_v2_ablation.py) | Sheaf-Delta mixer ablations (h6, decay init, post-norm, token shift) | [mixer_ablation_v2](../benchmark_results/paper/mixer_ablation_v2/README.md) |
| [`gpu_wikitext103_3seed.py`](gpu_wikitext103_3seed.py) | WikiText-103 (byte-level), 1,500 steps, 3 seeds | [wikitext103_3seed](../benchmark_results/paper/wikitext103_3seed/README.md) |
| [`gpu_wikitext103_fair.py`](gpu_wikitext103_fair.py) | WikiText-103: LR sweep + 5 tokens/param (4,406 steps) | [wikitext103_lr_and_horizon](../benchmark_results/paper/wikitext103_lr_and_horizon/README.md) |
| [`gpu_yarn.py`](gpu_yarn.py) | YaRN RoPE rescaling for GPT and the RoPE hybrid | [yarn_rope_scaling](../benchmark_results/development/yarn_rope_scaling/README.md) |
| [`gpu_yarn_hybrid.py`](gpu_yarn_hybrid.py) | YaRN RoPE rescaling for GPT and the RoPE hybrid | [yarn_rope_scaling](../benchmark_results/development/yarn_rope_scaling/README.md) |
| [`gpu_yarn_recheck.py`](gpu_yarn_recheck.py) | YaRN RoPE rescaling for GPT and the RoPE hybrid | [yarn_rope_scaling](../benchmark_results/development/yarn_rope_scaling/README.md) |

## Support scripts

| script | purpose |
|---|---|
| [`deepa_common.py`](deepa_common.py) | Shared model builders (paper model, Pure ALiBi, RoPE GPT, Pure SheafDelta, RoPE-DEEPA), training loop and eval for all paper benchmarks. |
| [`summarize_phaseA.py`](summarize_phaseA.py) | Rebuilds benchmark_results/PHASE_A_RESULTS.md from the Phase A JSONs. |
| [`summarize_phaseB.py`](summarize_phaseB.py) | Rebuilds benchmark_results/PHASE_B_RESULTS.md (all reviewer-response tables) from the JSONs. |
| [`build_results_index.py`](build_results_index.py) | Regenerates every benchmark_results/**/README.md and both index files from catalog.py. |
| [`catalog.py`](catalog.py) | This catalog. |
| [`run_bench.py`](run_bench.py) | Early generic benchmark runner (development). |
| [`rerun_failed_seeds.py`](rerun_failed_seeds.py) | NOT RUN. Would rerun only DEEPA's failed synthetic seeds at another LR (one-sided); superseded by the fair PROPER_FULLGRID pass in synthetic_converged. |
| [`data_prep/build_domains.py`](data_prep/build_domains.py) | Downloads and builds the six 41 MB domain files (code, Hindi, Chinese, DNA, CIFAR-10 pixels, mu-law audio). |
| [`data_prep/build_wikitext103_bytes.py`](data_prep/build_wikitext103_bytes.py) | Builds data/wikitext103_raw.bytes (40 MB train + 1 MB validation). |

## GPU queues

`queues/` holds the shell queues that ran these benchmarks back to back on the single 8 GB GPU (kept as an exact record of run order). Their logs are in `benchmark_results/_queue_logs/`.

## Reproducing the paper end to end

```bash
uv venv && uv pip install -e .        # torch 2.6 + cu124, flash-linear-attention 0.6, triton 3.2 (see pyproject)
python benchmarks/data_prep/build_wikitext103_bytes.py && python -I benchmarks/data_prep/build_domains.py .
bash benchmarks/queues/queue10_phaseA.sh   # then queue12, 13, 14, 15 (each waits for the previous one)
python benchmarks/summarize_phaseB.py      # rebuild the aggregated tables
python benchmarks/build_results_index.py   # rebuild these READMEs
```