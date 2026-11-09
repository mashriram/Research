# RESEARCH V2 — a cheap-to-train, cheap-to-run hybrid (block-diffusion × Householder-delta memory) 1B model

Status: design + falsifiable experiments. Nothing here beyond the 0.8M-param CPU result in RESULTS.md has been measured by us.
Literature claims below are from web search (Nov 2026); items marked (?) I could not verify in full.

## 0. Goal as constraints
Anyone with one consumer GPU (24 GB) can *train/adapt* a ~1B model; CPU can *run* it; quality competitive with 1–3B open models.
Three costs to attack: **training FLOPs**, **training memory**, **decode cost (bandwidth-bound)**.

## 1. What the evidence says about each idea in research.md

| Idea | Verdict | Why |
|---|---|---|
| Backprop-free local learning (DAD) | **Demote to optional** | Local methods lag at scale; one LM result: 6.6% BPB gap at 897M, 1.33x *more* compute (?) ([PoE local learning](https://pulseaugur.com/cluster/77133), [FF critique](https://arxiv.org/pdf/2603.24790)). Our own tiny run: +0.17 bit. Backward is only ~2/3 of FLOPs and every layer still needs its own local backward, so there is **no FLOP saving**; the only real win is activation memory / layer-parallel pipelining. Exact alternatives get that memory win without quality loss (§4). |
| Closed-form ridge readout | **Drop for LM** | Great for a 10-class head; a 32k-vocab softmax needs calibrated CE, ridge logits are not. |
| vMF / O(√V) head | **Inference-only, low priority** | Hierarchical heads lose quality; real saving is only at V≥128k. Use small vocab + fused chunked CE for training. |
| Continuous flow-matching text | **Replace by discrete masked diffusion** | Discrete masked diffusion is what scales (LLaDA 8B ≈ LLaMA3-8B class). |
| Householder orthogonal transport | **Keep — and it is the same math as the best linear RNNs** | See §2. |
| Dyadic shortcuts | **Keep as cheap local mixing** (≈ dilated short-conv), not as the global mechanism | Global recall needs exact attention somewhere (§2). |
| Implicit/DEQ O(1)-memory | **Drop** | Fixed-point iterations add FLOPs; reversible blocks give O(1) memory exactly (§4). |

## 2. Math: the sheaf/Householder idea *is* the delta rule
Gated DeltaNet state update (matrix state S ∈ R^{d_v×d_k}):
S_t = α_t S_{t-1}(I − β_t k_t k_tᵀ) + β_t v_t k_tᵀ , ‖k_t‖=1.
- (I − β kkᵀ) with β=2 is exactly a **Householder reflection** (orthogonal, eigenvalue −1 along k); β∈[0,2] interpolates identity↔reflection; α<1 is the learned forgetting.
- With α=1, β=2: ‖S_t‖ is preserved by the transition → no vanishing/exploding memory (the research's "isometry" claim, now in a form with fast training kernels).
- **Expressivity:** diagonal positive-gated recurrences (my current mixer, Mamba-1/2-style) cannot do parity/state-tracking; transition matrices with negative eigenvalues can (Grazzi et al.). Products of n_h Householders per token (DeltaProduct) extend this to rotations (det +1) — the "stalk transport" of the sheaf, with n_h as a compute/expressivity dial.
- Current frontier: Gated DeltaNet-2 beats Mamba-2/GDN/KDA/Mamba-3 at 1.3B/100B tokens ([arXiv 2605.22791](https://arxiv.org/pdf/2605.22791)); production models stack linear layers with a minority of attention layers (Qwen3-Next/Kimi-Linear style; [hybrid analysis](https://arxiv.org/pdf/2507.06457)); short sliding-window attention + linear memory is a strong pair ([2509.24552](https://arxiv.org/pdf/2509.24552)).
- **Design decision:** backbone = Householder-delta-rule memory layers (n_h=1–2, β∈[0,2], gated α) + ~1:4–1:7 exact-attention layers + dyadic/short-conv local mixing. This honestly absorbs "sheaf" and drops what is unproven.
- Caveat on our result: HSN beating GPT at 0.8M params/800 steps/char-level is most likely a locality inductive-bias effect in early training, not evidence of scale superiority.

## 3. Hybrid causal ⟷ diffusion (block diffusion) — why it is the right middle
Factorize p(x) = ∏_blocks p(block_i | blocks_<i), blocks of size b; inside a block, masked discrete diffusion (bidirectional); across blocks, causal. b=1 → AR, b=N → full diffusion.
- Cross-block context is a **compressed recurrent state** (§2) or exact KV of earlier blocks → decode memory O(1)/O(N) as chosen; inside-block attention costs O(b²), tiny.
- **Training-efficiency problem (the honest cost):** AR supervises every token; masked diffusion supervises ~(mask fraction) of tokens with 1/t weighting → higher gradient variance. Compute-matched 2026 study: pure diffusion ≈16× the compute gap, reduced to ~7.8× (b=32), **4.6× (b=8), 3.0× (b=4)** with block diffusion ([2610.17590](https://arxiv.org/abs/2610.17590), single preprint, unreplicated).
- **But** diffusion models are far better in *data-limited / multi-epoch* regimes (any-order = free augmentation) ([2511.03276](https://arxiv.org/pdf/2511.03276)); and a diffusion LM can be created by **adapting a pretrained AR model** with a short run (done by Dream/LLaDA-2-style recipes (?)).
- **Decision:** don't pretrain diffusion from scratch. Train the cheap/efficient AR (b=1) objective first, then *convert* to block diffusion b=4–16 with a short continued run, keeping **both modes in one set of weights**. Pay the 3–4.6× gap on ~1–5% of tokens, not 100%.
- **Multimodality:** images/audio tokenized to discrete codes drop straight into masked diffusion with the same backbone; no extra machinery on day one. (Quality at 1B unproven.)

## 4. Cheap training — ranked by expected leverage (all must be measured)
1. **Distill instead of pretraining from scratch.** From-scratch 1B @ 20 tokens/param ≈ 6·10⁹·2·10¹⁰ = 1.2·10²⁰ FLOP: ~weeks on one 4090, and comparators (Gemma/Llama 1–3B) saw 2–9T tokens (100–450× more). Matching them on tokens is impossible on a small budget; **logit-distillation from an open 1–3B teacher into our hybrid architecture** (layer-selection distillation e.g. [2512.20569](https://arxiv.org/pdf/2512.20569), [Taylor-Calibrate](https://arxiv.org/pdf/2606.16429)) is the only credible route to competing on benchmarks. Check teacher licence/ToS.
2. **Muon optimizer** (common in speedruns, token-efficiency gain over AdamW reported, size unverified here (?)) + **FP8** (~1.3× throughput; [Puro-2B](https://aiweekly.co/alerts/tsinghuas-puro-2b-pretrains-on-rtx-5090s-for-under-69k) trained 2B on RTX 5090s; [LLMQ](https://www.themoonlight.io/review/llmq-efficient-lower-precision-pretraining-for-consumer-gpus) on 4090s).
3. **Memory:** Householder maps are exactly invertible ⇒ build blocks **reversible** (RevNet coupling): activations recomputed backward, exact gradients, O(1) depth memory, ≈+33% FLOPs — what local learning was supposed to buy, without the accuracy loss. Add fused chunked cross-entropy, 8-bit optimizer states, grad checkpointing.
4. **Data:** quality filtering + curriculum is usually the largest lever at ≤3B; budget most effort here.
5. Local learning kept only as optional layer-parallel mode for multi-GPU/pipeline or tiny-VRAM devices.

## 5. Cheap inference — the math that favors diffusion/speculation on CPU
Decode is memory-bandwidth-bound: tokens/s ≈ BW / (bytes of weights read per forward pass). 1B int4 ≈ 0.5 GB; 60 GB/s DDR5 ⇒ ≲120 tok/s *upper bound* for one token per pass.
A pass that processes a block of b positions reads the same weights once ⇒ throughput ≈ b·accept-rate × that bound until compute-bound (CPU ≈ 1 TFLOP/s → 2 GFLOP/token for 1B ⇒ ~500 tok/s ceiling). So **block decoding is worth most exactly on CPU/edge**.
- **Self-speculative decoding with the same weights:** draft a block in diffusion mode, verify in causal mode in one pass ⇒ output distribution exactly the AR one (lossless). Expected tokens per pass with per-token acceptance a and block γ: (1 − a^{γ+1})/(1 − a). a=0.7, γ=8 ⇒ ≈3.3. (TiDAR-style idea (?)).
- State is O(1) per layer for the memory layers; KV only for the few attention layers (window-limited) ⇒ small long-context memory.
- Weight quantization int4/ternary for CPU; optional approximate top-k vocab routing (the R-SCHA idea) only at inference.

## 6. CPU honesty
Training 1B on CPU is infeasible: 6·10⁹ FLOP/token × 10⁹ tokens = 6·10¹⁸ FLOP ≈ 70 days at an optimistic 1 TFLOP/s, per *billion* tokens (20B tokens ≈ 4 years). CPU roles: inference, ≤100M-param demos, distilling small students. Say so in the README.

## 7. Experiment plan with kill criteria (each ≤ a few GPU-hours at 30–120M params)
| # | Question | Pass criterion | If fail |
|---|---|---|---|
| E0 | Does Householder-delta recurrence solve parity/state-tracking where our diagonal one fails? (CPU, minutes) | diag fails, delta ≥99% | drop the "sheaf=delta" claim |
| E1 | Mixer ablation equal-FLOP: attention vs GDN vs ours vs hybrid 1:4 | hybrid ≥ attention loss at lower train time | use plain GDN hybrid |
| E2 | AR vs block-diffusion b∈{4,8,16} equal-FLOP; and AR→BD adaptation tokens-to-parity | adaptation ≤5% of AR tokens | stay AR + speculative only |
| E3 | Self-speculative acceptance & real tok/s on CPU int4 | ≥2× over AR on CPU | drop diffusion mode |
| E4 | Muon vs AdamW; FP8; reversible memory vs speed | ≥1.3× token-efficiency combined | revert |
| E5 | Distill 1B teacher → hybrid: tokens needed to reach 95% of teacher on HellaSwag/ARC/MMLU | ≤10B tokens | reduce claims to "efficiency" rather than "beats" |

## 8. What would count as "beating" 1–3B models
Not raw benchmark wins from architecture alone (data dominates). Defensible claims: (i) same quality at a fraction of training cost via distillation+Muon+FP8; (ii) lower decode memory/latency on CPU/edge; (iii) long-context memory; (iv) day-one discrete multimodal path. Report all with seeds and equal-compute controls.

## 9. Measured update — backprop alternatives (see RESULTS.md)
DFA 3.16 bits (fails), greedy-local 2.37, teacher-target layer-wise 2.32, backprop 2.20. No method saved wall-clock.
Principled reading: gradient-free *credit assignment* is only competitive when each layer gets an informative target (teacher hidden states), i.e. in the distillation setting.
