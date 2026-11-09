# Findings Report — Householder Sheaf-Delta Language Model (`hsnlm`)

**Date:** 2026-11-09  **Scope:** everything done on CPU before the first GPU run
**One-line verdict:** At ~0.8M parameters, a Householder/delta-rule "sheaf" sequence mixer beats a matched Transformer on quality (~4%), length extrapolation, associative recall and state tracking, and decodes with constant cost and memory. It currently trains about 2x *slower* on CPU (eager PyTorch, no fused kernels), no backprop replacement we tried saves compute, and nothing here yet predicts behaviour at 100M+ parameters.

---

## 1. Executive summary

| Question | Answer | Evidence strength |
|---|---|---|
| Can a sheaf-style (Householder + delta-rule) mixer replace attention? | Plausibly yes as the main token mixer; keep a few attention layers as insurance | Moderate at tiny scale (2 seeds LM, 1 seed synthetic) |
| Does it beat a matched Transformer? | Yes at this scale: 1.880 vs 1.962 bits/byte (-4.2%), seed spread 0.01 | Moderate; tiny scale, lr sweep hit grid edge |
| Does it extrapolate to longer context? | Yes: flat to 8x training length; the RoPE Transformer collapses (4.21 at 2048) | Moderate; Transformer baseline had no extrapolation tricks |
| Can it track state (parity)? | Yes (96-100% up to 4x length); Transformer at chance | Mechanism-level, 1 seed |
| Is decoding O(1)? | Yes, measured: flat 4-7 ms/layer and 0.33 MB state vs attention 3.6 -> 56.6 ms and 2 -> 134 MB (context 256 -> 16k) | Strong (direct measurement) |
| Is training memory O(1) in depth with exact gradients? | Yes with reversible blocks: flat 1.29 MB for 4-32 layers, gradients match autograd to ~1e-7 | Strong |
| Is training *faster* than a Transformer? | **No, not yet.** ~2x slower (LM) / ~2.9x slower (step) on CPU eager. Needs fused GPU kernels to settle | Measured negative |
| Can backprop be replaced with something cheaper? | **Nothing we tested saved compute**; DFA failed; local methods lose accuracy | Strong negative |
| Can training time be O(1)? | No. Per-token training cost has a floor of ~1 forward + 1 backward | Argument + measurement |

---

## 2. Background and what we set out to test

The starting point was `research.md`, which proposed two ideas:

1. **DAD** (Decoupled Analytical Dense): replace backprop with local target propagation, closed-form readout, async layer updates.
2. **AC-SFN** (Asymptotic Conformal Sheaf-Flow Network): replace the Transformer with sheaf diffusion (Householder restriction maps, dyadic shortcuts), an implicit/DEQ depth solver, an O(sqrt V) vMF vocabulary head, and non-autoregressive flow-matching generation.

User goals as the project evolved: cheap to train, cheap/fast/smart to infer, runnable by anyone with a reasonable GPU (CPU for inference and small demos), eventually competitive with 1-4B open models; a hybrid between causal and diffusion decoding was desired; backprop's costs and the Transformer's O(N^2) were the named problems.

**Method:** we built a library, tested each claim against a matched baseline, and kept only what survived. Literature checks (web search, Nov 2026) supplemented our own measurements; items we could not verify in full are marked (?).

---

## 3. What was built (`hsnlm/`)

| Component | File | Notes |
|---|---|---|
| Transformer baseline (RoPE, KV-cache decode) | `hsnlm/models.py` | pre-RMSNorm, GELU MLP |
| `SheafMixer` (v1: diagonal gated recurrence + dyadic Householder edges) | `models.py` | first design; weak at recall |
| `DeltaMixer` (Householder-type matrix-memory, gated delta rule) | `models.py` | S_t = a_t (S_{t-1} + b_t (v_t - S_{t-1}k_t) k_t^T), b in [0,2] |
| `SheafDeltaMixer` (delta memory + 5 dyadic Householder edges, thin transport) | `hsnlm/fastmixer.py` | the final mixer |
| Chunkwise delta rule (exact, matmul-only inner loop) | `fastmixer.py` | verified == sequential loop |
| Thin Householder transport (O(M·D) not O(D^2) per edge) | `fastmixer.py` | verified == explicit reflections, isometric |
| `HybridLM` (pattern string, e.g. `ddad`) | `models.py` | d = sheaf-delta block, a = attention block |
| Reversible stack + memory-budget policy (`fit_memory`) | `hsnlm/reversible.py` | exact grads, O(1) depth memory |
| Local-learning trainer (DAD-style) | `hsnlm/train.py` | greedy per-block local loss |
| Benchmarks / experiments | `benchmarks/`, `experiments/` | resumable suite; see section 15 |
| Tests | `tests/` | step==parallel, causality, isometry, chunk==seq, thin==explicit |

Test status: all correctness tests pass (recurrent step decoding equals the parallel forward for Transformer, sheaf, and hybrid; Householder transport is isometric; models are causal; chunked delta == sequential delta to 1e-4; thin transport == explicit reflections).

---

## 4. Key conceptual finding: the sheaf idea *is* the delta rule

The Gated DeltaNet state update is
`S_t = a_t S_{t-1} (I - b_t k_t k_t^T) + b_t v_t k_t^T`.
`(I - b k k^T)` with b = 2 is exactly a Householder reflection (eigenvalue -1 along k); b in [0,2] interpolates identity <-> reflection. So the research's "Householder restriction maps give isometric transport" is the same mathematics as the delta rule used by the current best linear-attention models (Gated DeltaNet, DeltaProduct), and it comes with fast training kernels. Consequences:

* Diagonal positive-gated recurrences (our v1 mixer, Mamba-1/2-style) provably cannot solve parity-type state tracking; transitions with negative eigenvalues can. Our parity experiment confirms this empirically (section 5.3).
* The global-recall behaviour comes from the matrix-valued delta-rule state (DeltaNet lineage); the dyadic sheaf edges contribute only cheap local mixing. "Sheaf" is a loose label for the combination; credit for the memory mechanism belongs to the delta-rule literature.

---

## 5. Main benchmark: CPU suite (the headline result)

**Setup.** Byte-level enwik8 (train: first 8M bytes, val: next 1M), 4 layers, width 128, 4 heads, context 256, batch 16, 2,500 steps = 10.2M tokens/run, AdamW (betas 0.9/0.95, wd 0.05), 50-step warmup + cosine, grad-clip 1. Learning rate chosen per architecture from {1e-3, 2e-3, 4e-3} by a 200-step sweep (all three chose 4e-3). Parameters matched within 2%: Transformer 0.820M, sheaf-delta 0.827M, hybrid 0.838M. 2 seeds. Machine: 8 CPU cores, shared (timings noisy).

### 5.1 Language modelling — validation bits per byte (lower is better)

| Architecture | seed 0 | seed 1 | mean @256 | @512 | @1024 | @2048 | train tok/s |
|---|---|---|---|---|---|---|---|
| Transformer (RoPE) | 1.957 | 1.966 | 1.962 | 2.501 | 3.482 | 4.212 | 13.3k |
| **Sheaf-delta (4 layers)** | 1.889 | 1.870 | **1.880** | **1.845** | **1.886** | **1.879** | 6.5k |
| Hybrid (3 sheaf-delta + 1 attention) | 1.887 | 1.892 | 1.890 | 1.863 | 1.930 | 1.943 | 7.4k |

Reading:
* Sheaf-delta beats the Transformer by 0.082 bits/byte (4.2%); the Transformer's own seed-to-seed spread is 0.009, so the gap is ~9x seed noise. Hybrid and sheaf-delta are statistically tied at the training length (difference 0.010, comparable to seed spread).
* Length extrapolation (trained at 256, evaluated at 512/1024/2048): sheaf-delta is flat; hybrid degrades mildly (its single attention layer uses RoPE past its training length); the Transformer collapses. A fixed-size recurrent state has no position-dependent failure mode — a structural advantage — but the baseline here has no extrapolation tricks (NoPE, YaRN, ALiBi), so the true gap against a tuned baseline is smaller.
* Training throughput is ~2x *lower* for the sheaf models in this eager-mode CPU setup.

### 5.2 Associative recall (multi-query key->value lookup; 2 layers, width 64, 2,500 steps, 1 seed; chance ~0.016)

| Key-value pairs | Transformer | Sheaf-delta | Hybrid |
|---|---|---|---|
| 4 | 0.462 | 0.532 | 0.502 |
| 8 | 0.222 | 0.330 | 0.344 |
| 16 | 0.109 | 0.170 | **0.238** |

All three are far from solving the task, i.e. undertrained. The honest reading is "the delta memory learns recall faster here", not "attention cannot recall". An earlier run (older delta variant, 8 pairs, 1,500 steps) gave attention 0.370 / diagonal-sheaf 0.265 / Householder-delta 0.466, consistent in direction.

### 5.3 State tracking — parity (train length 32; test 32 / 64 / 128; chance 0.5; 1 seed)

| Transformer | Sheaf-delta | Hybrid |
|---|---|---|
| 0.566 / 0.531 / 0.514 | 1.000 / 0.999 / 0.965 | 1.000 / 1.000 / 0.997 |

The Transformer did not learn the task in 2,500 steps; the Householder-based models solve it and generalise to 4x length. This is the cleanest empirical support for the reflection-transition claim.

---

## 6. Inference: O(1) decoding is real (measured)

Per-layer single-token decode, batch 1, width 1024, 16 heads (CPU), cache pre-filled to the given length:

| Context | Attention step | KV cache | Sheaf-delta step | State |
|---|---|---|---|---|
| 256 | 3.62 ms | 2 MB | 7.10 ms | 0.33 MB |
| 1,024 | 6.16 ms | 8 MB | 5.09 ms | 0.33 MB |
| 4,096 | 18.01 ms | 34 MB | 4.60 ms | 0.33 MB |
| 16,384 | 56.64 ms | 134 MB | 4.03 ms | 0.33 MB |

Attention cost and cache grow with context; the sheaf-delta layer is flat (the 256-token row is dominated by one-off overhead). Per layer, per sequence — multiply by layers and batch for a full model. Decode is memory-bandwidth-bound on CPU, so block decoding / speculative verification (section 13, G7) is expected to help most on CPU/edge, but we have **not** measured that.

---

## 7. Backprop: what we tested and what it showed

### 7.1 Credit-assignment alternatives (same model, 0.82M params, 800 steps, char-level Shakespeare, 1 seed)

| Rule | Cross-block gradient? | Val bits/char | Wall time |
|---|---|---|---|
| Backprop | yes | **2.202** | 536 s |
| Greedy local next-token loss per block (DAD-style) | no | 2.372 | ~458 s |
| Layer-wise regression to a trained teacher's hidden states (oracle) | no | 2.316 | 589 s |
| DFA (random fixed feedback matrices) | no | 3.164 | 463 s |

* **No alternative reduced wall-clock.** Each block still performs its own forward and local backward; only the inter-layer chain is removed. The gain is memory (no tape across depth) and layer-parallelism, not FLOPs.
* DFA failed (about a full bit worse). Greedy local lost 0.17 bit. Rich per-layer targets from a teacher recovered part of the gap (0.11 bit behind) but require an existing teacher, which the project excluded (train from scratch).
* In the very first benchmark (older, weaker Transformer baseline — learned positions, no tuning) local learning looked "free" for the Transformer (2.592 -> 2.618) only because both were badly undertrained; do not rely on that comparison. Literature agrees local methods lag at scale (one LM result: ~6.6% BPB gap at 897M params (?), [FF critique](https://arxiv.org/pdf/2603.24790)).

### 7.2 Forward-mode gradients cannot substitute

Forward-gradient (JVP) estimate vs true gradient on a 59,412-parameter block: cosine alignment 0.0016 (1 tangent), 0.013 (8), 0.032 (64), 0.095 (512) — matching sqrt(k/d). Learning from forward passes alone needs ~d passes' worth of information; hopeless at scale.

### 7.3 Why a drop-in replacement should not be expected to cut cost

Backprop computes the full gradient for ~2x the forward cost (the "cheap gradient principle"); methods that cost less extract less information per step. A step costs ~3 units (1 forward + 2 backward); reversible blocks add one forward recompute (4 units). Our results are consistent with this accounting.

### 7.4 A gradient-free data point worth noting

A 5-gram count model ("trained" by counting in 3.9 s, no gradients) reaches 2.41 bits/char on the Shakespeare split, vs 2.20 (sheaf, backprop, ~540 s) and 2.59 (our early Transformer). Tiny-data regime, and counts do not generalise as data grows, but it motivates *slow weights by gradient, fast memory by closed-form writes*: the delta rule is itself an online least-squares update executed during the forward pass, so the final architecture already embodies this split.

---

## 8. Memory: reversible blocks (solved, exact)

Householder maps are exactly invertible, so blocks are built reversible (RevNet coupling `y1 = x1 + F(x2); y2 = x2 + G(y1)`); only the final activations are stored and earlier ones are reconstructed in backward.

| Layers | Autograd saved activations | Reversible | Step time (std / rev) |
|---|---|---|---|
| 4 | 19.4 MB | 1.29 MB | 0.10 / 0.08 s |
| 8 | 37.8 MB | 1.29 MB | 0.11 / 0.13 s |
| 16 | 74.5 MB | 1.29 MB | 0.22 / 0.28 s |
| 32 | 147.8 MB | 1.29 MB | 0.51 / 0.54 s |

Gradients match ordinary autograd: max abs difference 3.7e-9 (v1 mixer) and 3.7e-7 (delta mixer) at gradient scale ~0.03-0.07. Overhead is the theoretical +1 forward pass (up to ~+33%; 0-30% observed at tiny size). A policy (`RevLM.fit_memory`) reverses only as many layers as a memory budget requires, so on a large GPU the overhead can be zero. Caveat: the reversible two-stream model in the timing experiment had fewer parameters (0.63M) than the Transformer (0.86M), so those rows are indicative, not matched.

---

## 9. The negative result: training speed is not solved

Profile and measurement (CPU, eager, 8,192 tokens/step, width 128, 4 layers):

| N | Transformer | Sheaf-delta (plain autograd) | Saved activations (Transformer / sheaf) |
|---|---|---|---|
| 512 | 765 ms | 2,190 ms (2.9x slower) | 378 MB / 1,059 MB |
| 2,048 | 1,462 ms | 5,771 ms (3.9x slower) | 378 MB / 1,060 MB |

In the LM suite the end-to-end gap is ~2x (6.5k vs 13.3k tok/s). Cause (profiler): ~50% of mixer time is memory-bound elementwise ops (`add_`, `fill_`, `mul`) plus ~20% in many small `bmm` calls from the per-chunk Python loop; the Transformer benefits from a fused attention kernel. Replacing the D x D Householder GEMM with thin matrix products (30x fewer FLOPs) and the chunked delta rule were verified correct but did not remove the eager-mode overhead. This is a kernel-fusion problem (e.g. `fla`'s chunked gated-delta Triton kernels), **not demonstrated solved**. FLOP accounting suggests parity at short context and an advantage at long context (attention FLOPs/token ~ N·D vs the fixed-size state), but at N <= 2048 and width 128 attention is not the bottleneck, so the O(N) advantage is not visible in these timings.

Training time cannot be O(1) in any method; only memory (reversible blocks) and decode cost (fixed state) are constant.

---

## 10. Assessment of the original `research.md` claims

| Claim | Status | Basis |
|---|---|---|
| DAD local learning replaces backprop | **Not supported** | Our tests (section 7); literature |
| Closed-form ridge readout | Dropped for LMs | Reasoning only (softmax over large vocab needs calibrated CE) — not tested by us |
| Householder isometric transport | **Supported** — and equivalent to the delta rule | Tests + parity |
| Dyadic shortcuts / O(N) sheaf mixer | Supported only as cheap local mixing; global recall comes from delta memory | Recall/parity experiments |
| Implicit/DEQ O(1) memory | Dropped in favour of reversible blocks (exact, no fixed-point iterations) | Reversible results; DEQ not tested by us |
| vMF / O(sqrt V) head | Deprioritised (inference-only, quality risk) | Literature/reasoning only — not tested |
| Continuous flow-matching generation | Replaced by discrete masked/block diffusion in the plan | Literature (below); not tested by us |

---

## 11. Literature context (searched Nov 2026; verify before citing)

* **Block diffusion vs autoregressive:** a compute-matched study (~1,700 models) reports the pure-diffusion compute gap shrinking from ~16x to ~7.8x (block 32), 4.6x (block 8), 3.0x (block 4) — single unreplicated preprint ([2610.17590](https://arxiv.org/abs/2610.17590)). Diffusion models are better data learners in data-limited/multi-epoch regimes ([2511.03276](https://arxiv.org/pdf/2511.03276)). LLaDA-8B is competitive with similar-size autoregressive models; the strongest general-purpose models remain autoregressive.
* **Linear/delta-rule layers:** Gated DeltaNet-2 beats Mamba-2/GDN/KDA/Mamba-3 at 1.3B/100B tokens ([2605.22791](https://arxiv.org/pdf/2605.22791)); production stacks interleave a minority of attention layers ([hybrid analysis](https://arxiv.org/pdf/2507.06457)); short sliding-window attention + linear memory is a strong pairing ([2509.24552](https://arxiv.org/pdf/2509.24552)).
* **Cheap training:** FP8 gives ~1.3x throughput, consumer-GPU pretraining recipes exist ([Puro-2B](https://aiweekly.co/alerts/tsinghuas-puro-2b-pretrains-on-rtx-5090s-for-under-69k), [LLMQ](https://www.themoonlight.io/review/llmq-efficient-lower-precision-pretraining-for-consumer-gpus)); Muon is widely used in speedruns (gain size not verified (?)). Distilling Transformers into hybrids is an active area ([2512.20569](https://arxiv.org/pdf/2512.20569), [2606.16429](https://arxiv.org/pdf/2606.16429)) — relevant as a fallback, though the project's stated constraint is training from scratch.

---

## 12. Threats to validity

1. **Scale:** ~0.8M parameters, 10M tokens, one dataset (enwik8 bytes). Early-training advantages from locality/recurrence biases often vanish or reverse at scale. Nothing here predicts 100M+ behaviour.
2. **Learning-rate grid edge:** all three architectures picked the largest value (4e-3); true optima may be higher and could change the ranking or the gap.
3. **Seeds:** 2 for language modelling, 1 for synthetic tasks; no confidence intervals.
4. **Baseline strength:** the Transformer is a plain RoPE model (no NoPE/YaRN/ALiBi, no tuned width/depth); extrapolation comparisons favour the recurrent models.
5. **Speed numbers:** eager CPU on a shared machine (timing noise); the Transformer used PyTorch's fused SDPA while our layers use unfused ops, so absolute speed comparisons say little about GPU kernels.
6. **Variant drift:** the delta-rule variant changed between the early recall run (decay-before-write) and the suite (decay-after-write, chunk-friendly); both directions agree but numbers are not comparable.
7. **Synthetic tasks undertrained:** recall accuracies are far from saturated; they measure learning speed, not capacity limits.
8. **No downstream/real-task evaluation** (no HellaSwag/ARC/MMLU, no long-context retrieval benchmarks, no text samples reviewed).
9. **Timing experiment parameter mismatch:** reversible model had 0.63M parameters vs 0.86M Transformer.
10. **Diffusion/block-decoding ideas are untested here:** everything in this report is autoregressive; the causal/diffusion hybrid is design only.

---

## 13. Plan for the GPU phase

Principle: the same suite, scaled, with kill criteria. Ordered so cheap decisive tests come first.

| # | Experiment | Pass criterion | If it fails |
|---|---|---|---|
| G1 | Fused kernels (`fla` chunked gated-delta; Triton thin Householder) — step time vs Transformer at 125M, ctx 2k and 8k | within ~1.2x at 2k, faster at 8k | Reconsider: speed claim not supportable |
| G2 | Wider lr sweep (incl. > 4e-3), 3+ seeds, 100-300M params, equal tokens | sheaf-delta/hybrid >= Transformer val loss | Treat tiny-scale win as an artefact |
| G3 | Baseline with extrapolation tricks (NoPE / YaRN) | gap to sheaf narrows but persists | Drop extrapolation claim |
| G4 | Long-context retrieval (needle-in-haystack, RULER-like) pure vs hybrid vs Transformer | hybrid ~ Transformer | Raise attention ratio |
| G5 | Hybrid ratio ablation (1:3, 1:4, 1:7) | find best quality/speed point | — |
| G6 | Reversible memory policy at scale; Muon vs AdamW; FP8 | combined >= 1.3x token efficiency | Revert pieces |
| G7 | Block-diffusion mode (block 4-16) on same backbone; AR->block adaptation cost | adaptation <= ~5% of AR tokens; self-speculative acceptance high | Stay autoregressive |
| G8 | Downstream benchmarks (HellaSwag/ARC/MMLU-style) at the largest size reached | competitive with same-size Transformer | Narrow claims to efficiency |

The earlier proposal to fund a ~1B-class run was deferred; G1-G2 are the cheap gate before any larger spend.

---

## 14. Conclusions

1. **Architecture:** Householder-transition delta-rule memory plus cheap local sheaf mixing, optionally with a few attention layers, is a credible attention replacement: better quality, extrapolation, recall and state tracking at tiny scale, with constant-cost decoding.
2. **Backprop:** keep it. Nothing tested cuts compute; reversible blocks fix depth memory exactly, and the delta rule already provides gradient-free fast memory in the forward pass. Stop pursuing DFA/forward-gradient/greedy-local as cost reducers.
3. **Speed:** the unresolved risk. The O(N) / O(1)-decode advantages are real in the maths and in decode measurements, but a faster-than-Transformer *training* claim requires fused kernels and has not been shown.
4. **Honest framing for any write-up:** "a Transformer-attention replacement with constant-cost decoding and strong state tracking, showing promising tiny-scale results; scale and speed evidence pending."

---

## 15. Reproducibility and file map

```
hsnlm/                          package: models.py, fastmixer.py, reversible.py, train.py, data.py
benchmarks/cpu_suite.py         full resumable suite (sweep -> LM seed0 -> synthetic -> LM seed1)
benchmarks/run_bench.py         first Shakespeare benchmark
experiments/credit.py           backprop vs DFA vs layer-wise teacher targets vs greedy local
experiments/fixes.py            reversible-block correctness + memory
experiments/rbf.py              speed/memory table (Transformer vs sheaf-delta vs reversible)
experiments/recall.py           early associative-recall comparison
experiments/decode_o1.py        per-layer decode cost vs context
experiments/fwdgrad.py          forward-gradient alignment
experiments/ngram.py            count-based (gradient-free) baseline
tests/                          correctness tests
results/cpu_suite.json|.log     raw suite results; results/BENCH_CPU.md summary
RESULTS.md, RESEARCH_V2.md      running notes and design document
```
Run: `python benchmarks/cpu_suite.py` (resumes automatically). Data: `data/enwik8` (100 MB, mattmahoney.net), `data/shakespeare.txt`.
