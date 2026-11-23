# DEEPA: Dyadic-Delta Enhanced-Sheaf, Efficient Pairing with ALiBi for Long-Context Language Modeling and High-Throughput Serving

**Paper Title:** *DEEPA: Dyadic-Delta Enhanced-Sheaf, Efficient Pairing with ALiBi for Long-Context Language Modeling and High-Throughput Serving*  
**Authors:** Dr. Deepa, Mr. J. V. K. Hem, and Mr. M. A. Shriram (*Independent Researchers, Chennai, India*; `{deepa.rME, hemjvk2004, mashriram}@gmail.com`)  
**Acronym Definition:** **DEEPA** = **D**yadic-Delta **E**nhanced-sheaf, **E**fficient **P**airing with **A**LiBi  
**Springer LNCS PDF:** [DEEPA_Springer.pdf](file:///home/mukundan/learning/llm/Research/Research/hsnlm/paper/DEEPA_Springer.pdf) ([DEEPA_Springer.tex](file:///home/mukundan/learning/llm/Research/Research/hsnlm/paper/DEEPA_Springer.tex))  
**Codebase & Implementation:** `hsnlm` ([models.py](file:///home/mukundan/learning/llm/Research/Research/hsnlm/hsnlm/models.py), [fastmixer.py](file:///home/mukundan/learning/llm/Research/Research/hsnlm/hsnlm/fastmixer.py), [fastmixer_fused.py](file:///home/mukundan/learning/llm/Research/Research/hsnlm/hsnlm/fastmixer_fused.py), [reversible.py](file:///home/mukundan/learning/llm/Research/Research/hsnlm/hsnlm/reversible.py))

---

## Abstract

**Objectives:** Autoregressive Transformers with Rotary Position Embeddings (RoPE) incur $O(N)$ inference KV caches, quadratic $O(N^2)$ long-context training cost, and perplexity collapse beyond the training horizon. Purely recurrent models offer $O(1)$ state but degrade in exact retrieval at scale. We introduce **DEEPA** (**D**yadic-Delta **E**nhanced-sheaf, **E**fficient **P**airing with **A**LiBi), a $3:1$ hybrid sequence backbone pairing negative-eigenvalue gated delta-rule matrix memory and orthogonal dyadic sheaf diffusion with ALiBi attention.

**Methods:** DEEPA interleaves three *Sheaf-Delta* blocks ($H_{\text{mix}}=6, d_h=64$) with one *ALiBi Attention* block ($H_{\text{attn}}=8, d_a=48$) in a `dddaddda` motif. Each Sheaf-Delta layer pairs a Triton-fused chunkwise WY gated delta-rule memory ($b_t \in (0,2)$) with 5-lag isometric dyadic sheaf diffusion ($d \in \{1,2,4,8,16\}$), independent delta normalization, and SiLU output gating. Folding ALiBi biases into query/key vectors enables native FlashAttention execution (matching dense-mask loss within $4\times 10^{-5}\text{ bpb}$).

**Results:** Across 8 byte-level domains, ALiBi-DEEPA beats Pure ALiBi on $7/8$ datasets (e.g., code $-0.047$, Hindi $-0.084$, DNA $-0.097$, Audio $-0.114$, WikiText-103 $-0.015\text{ bpb}$), leads across all four scales ($1.85\text{M}\to 38.34\text{M}$), and wins $2/3$ Long Range Arena tasks ($+12.1\%$ on 4k IMDb; ListOps tie at chance). In equal wall-clock training DEEPA beats Pure ALiBi on $3/3$ seeds ($0.7770$ vs. $0.7863\text{ bpb}$), and it solves parity and $S_3$ state tracking where attention baselines fail (a $\beta\in(0,1)$ control confirms negative eigenvalues are responsible). It extrapolates monotonically to `16k` ($1.507\to 1.469$ vs. RoPE $5.299$), cuts KV cache by $4\times$, increases 8 GB serving throughput by $3.7\text{--}4.8\times$ vs. RoPE ($3.1\text{--}4.2\times$ vs. ALiBi), and trains $1.16\text{--}1.57\times$ faster than RoPE ($1.06\text{--}1.46\times$ faster than folded ALiBi) at `4k–16k`.

**Conclusion:** DEEPA delivers competitive or superior modeling quality across diverse modalities alongside $4\times$ smaller KV caches, stable $16\times$ extrapolation, and higher long-context throughput, while documenting its boundaries at short contexts ($T\le 1\text{k}$) and fixed 2D raster strides.

**Keywords:** Hybrid Sequence Modeling, Gated Delta Rule, Dyadic Sheaf Diffusion, ALiBi, FlashAttention, Length Extrapolation, Constant-State Decoding.

---

## 1. Introduction

Autoregressive Transformers (Vaswani et al., 2017) paired with Rotary Position Embeddings (RoPE; Su et al., 2024) form the backbone of modern language models, yet impose three structural bottlenecks in long-context and resource-constrained settings: (1) an $O(N)$ per-sequence Key-Value (KV) cache across all $L$ layers that caps concurrent serving batch size and makes decoding memory-bandwidth bound; (2) quadratic $O(N^2 D)$ training and prefill scaling; and (3) catastrophic out-of-distribution perplexity degradation when evaluating beyond the training horizon $N_{\text{train}}$.

Purely recurrent architectures—including linear attention (Katharopoulos et al., 2020), selective state-space models (Mamba-1/2; Gu & Dao, 2024; Dao & Gu, 2024), and fast-weight delta networks (Schlag et al., 2021; Yang et al., 2024b; Yang et al., 2025)—replace the $O(N)$ KV cache with a fixed-size state. However, positive-gated diagonal recurrences cannot track parity (which requires a negative eigenvalue) or non-abelian groups such as $S_3$ (Merrill et al., 2024; Grazzi et al., 2025), and purely recurrent models often lag full attention on exact in-context retrieval as capacity grows. Hybrid backbones such as Jamba (Lieber et al., 2024) and Samba (Ren et al., 2024) interleave recurrent and attention layers, yet typically combine positive-eigenvalue diagonal SSMs with RoPE or sliding-window attention, retaining either RoPE's length collapse or truncated context windows.

### 1.1 Key Contributions
1. **Distinctive Hybrid Mechanism:** DEEPA combines negative-eigenvalue gated delta-rule matrix memory ($b_t \in (0,2)$; Grazzi et al., 2025) with a parallel, isometric multi-scale causal dyadic sheaf diffusion path ($d \in \{1,2,4,8,16\}$) and unrotated ALiBi attention (Press et al., 2022) in a $3:1$ ratio (the best among the ratios tested: $3:1$ vs. $4:1$ and pure baselines). Unlike Jamba, Samba, or standard Gated DeltaNet, DEEPA operates without a causal $q,k$ token-shift/convolution mixer (we show a causal $q,k$ token shift degrades delta-rule erasure by $+0.046\text{ bpb}$) and preserves unrotated shift-invariant feature coordinates across recurrent and attention layers.
2. **FlashAttention-Compatible ALiBi Folding & Compiler Fusion:** By folding ALiBi's linear distance bias directly into query/key vectors (exact to $4\times 10^{-5}\text{ bpb}$), combining Triton chunkwise WY delta kernels with zero-copy dyadic slices, and applying whole-model `torch.compile`, DEEPA trains $1.16\text{--}1.57\times$ faster than FlashAttention RoPE GPT ($1.06\text{--}1.46\times$ faster than folded-kernel Pure ALiBi) from `4k` to `16k`, cuts inference KV cache by $4\times$, and boosts 8 GB serving throughput by $3.7\text{--}4.8\times$ vs. RoPE GPT ($3.1\text{--}4.2\times$ vs. Pure ALiBi).
3. **Multi-Modal, Scaling, Extrapolation, and Mechanism Evaluation:** Across 8 real-world domains (`enwik8`, `WikiText-103`, Python code, Hindi, Chinese, DNA, CIFAR-10 pixels, 16 kHz $\mu$-law audio), 4 scales ($1.85\text{M}\to 38.34\text{M}$ at matched $5\text{ tokens/param}$), 18,000-step Chinchilla-ratio convergence, 3-seed $16\times$ extrapolation, Long Range Arena, and a 9-task converged synthetic suite, DEEPA establishes strong quality and efficiency gains alongside clearly defined boundaries.

---

## 2. Related Work and Positioning Against Existing Hybrids

**Table 1: Architectural taxonomy contrasting DEEPA against existing hybrid and recurrent backbones across six design axes.**

| Architecture | Recurrent State Eigenvalues | Local Context Path | Attn Ratio / Type | Pos. Encoding | Decode State / $16\times$ Extrap. |
|---|---|---|---|---|---|
| **RoPE GPT** (Su et al., 2024) | None | Full Attention | $0:1$ Full | RoPE (rotates $q,k$) | $O(L N)$ / Collapses |
| **Pure ALiBi** (Press et al., 2022) | None | Full Attention | $0:1$ Full | ALiBi (unrotated) | $O(L N)$ / Stable |
| **Jamba** (Lieber et al., 2024) | Diagonal $\lambda \in (0,1)$ (Mamba) | Depthwise Conv1d ($k=4$) | $7:1$ Full | RoPE | $\frac{1}{8}O(LN)$ / RoPE-bound |
| **Samba** (Ren et al., 2024) | Diagonal $\lambda \in (0,1)$ (Mamba) | Depthwise Conv1d ($k=4$) | $1:1$ SWA | RoPE + Window | $O(LW)$ / Window-bound |
| **Gated DeltaNet** (Yang et al., 2025) | Projection $\lambda \in [0,1]$ ($\beta \in (0,1)$) | Short Conv1d on $q,k,v$ | Pure Rec. (no attn) | None | $O(L d_h^2)$ / Retrieval-limited |
| **DEEPA (Ours)** | **Reflection $\lambda \in [-1,1]$** | **Parallel Dyadic Sheaf ($d \in \{1..16\}$, $U^\top U=I$)** | **$3:1$ Full (`dddaddda`)** | **Flash-folded ALiBi** | **$\frac{1}{4}O(LN)$ / Stable $16\times$** |

---

## 3. The DEEPA Architecture and Mathematical Formulation

**Table 2: Complete notation and symbol glossary for DEEPA (default $14.44\text{M}$ `h6` configuration: $D=384, L=8, H_{\text{mix}}=6, d_h=64, H_{\text{attn}}=8, d_a=48$).**

| Symbol | Shape / Domain | Definition and Role |
|---|---|---|
| $B, T, D$ | $\mathbb{N}$ | Batch size, sequence length, and model hidden dimension ($D=384$). |
| $L, \mathcal{P}$ | $L=8$, `dddaddda` | Total blocks and repeating $3:1$ layer pattern (`d`=Sheaf-Delta, `a`=ALiBi). |
| $H_{\text{mix}}, d_h$ | $6, 64$ | Number of Sheaf-Delta mixer heads and per-head dimension ($H_{\text{mix}} d_h = D$). |
| $H_{\text{attn}}, d_a$ | $8, 48$ | Number of ALiBi attention heads and per-head dimension ($H_{\text{attn}} d_a = D$). |
| $x_t \in \mathbb{R}^D$ | $[B, T, D]$ | Pre-RMSNorm residual stream representation (stalk) at token position $t$. |
| $\hat{q}_t, \hat{k}_t, v_t$ | $[B, H_{\text{mix}}, T, d_h]$ | L2-normalized query/key vectors ($\|\hat{q}_t\|_2=\|\hat{k}_t\|_2=1$) and value vector. |
| $S_t$ | $[B, H_{\text{mix}}, d_h, d_h]$ | Per-head associative matrix memory ("notebook" state) at step $t$. |
| $a_t, b_t$ | $(0.8, 1)$, $(0, 2)$ | Per-head scalar retention decay gate and Householder erasure/write step size. |
| $\mathcal{D}, d$ | $\{1, 2, 4, 8, 16\}$ | Exponentially spaced causal dyadic lags for local sheaf transport. |
| $U_{t,d}$ | $d_h \times d_h$ ($U^\top U=I$) | Orthogonal transport map formed by $M=2$ thin Householder reflections. |
| $g_{t,d}$ | $(0, 1)$ | Learned sigmoid edge gate weighting transported message from position $t-d$. |
| $o_t^{\text{delta}}, o_t^{\text{dyadic}}$ | $[B, H_{\text{mix}}, T, d_h]$ | Outputs of the global delta-rule branch and local dyadic sheaf branch. |
| $m_h$ | $2^{-8h/H_{\text{attn}}}$ | Fixed geometric ALiBi slope for attention head $h \in \{1,\dots,H_{\text{attn}}\}$. |

### 3.1 Macro $3:1$ Topology, Delta Memory, Dyadic Sheaf, and Branch Fusion
See [fig1_architecture.pdf](file:///home/mukundan/learning/llm/Research/Research/hsnlm/paper/figs/fig1_architecture.pdf). Every block applies pre-RMSNorm residual updates $h_\ell = x_{\ell-1} + \text{Mixer}_\ell(\text{RMSNorm}(x_{\ell-1}))$, $x_\ell = h_\ell + \text{MLP}_\ell(\text{RMSNorm}(h_\ell))$, where Sheaf-Delta blocks (`d`) use SwiGLU (`mlp_mult=3.65`) and ALiBi blocks (`a`) and baselines use GELU (`mlp_mult=4.0`).

- **Global Gated Delta-Rule:** $\hat{q}_t = q_t/(\|q_t\|_2+\epsilon)$, $\hat{k}_t = k_t/(\|k_t\|_2+\epsilon)$, $b_t = 2\sigma(z_{b,t}) \in (0,2)$, $a_t = 0.8 + 0.2\sigma(z_{a,t}) \in (0.8,1)$ ($z_{a,t}$ initialized at $+3.0 \Rightarrow a_0 \approx 0.9905$, $\sim 73$-token half-life):
$$S_t = a_t S_{t-1}\!\left(I - b_t \hat{k}_t \hat{k}_t^\top\right) + b_t v_t \hat{k}_t^\top, \qquad o_t^{\text{delta}} = d_h^{-1/2}\, S_t \hat{q}_t.$$
At $b_t=2$, $(I - 2\hat{k}_t \hat{k}_t^\top)$ is a Householder reflection ($\lambda=-1$), enabling parity ($\mathbb{Z}_2$) and non-abelian $S_3$ state tracking. Decoding runs in $O(d_h^2)$ with a constant `61.4 KB` bf16 state per layer ($30,720$ values).
- **Local Dyadic Sheaf Diffusion:** $U_{t,d} = \prod_{m=1}^{2}(I - 2\hat{r}_{d,m}\hat{r}_{d,m}^\top)$, $o_t^{\text{dyadic}} = \sum_{d \in \{1,2,4,8,16\}} g_{t,d} U_{t,d} v_{t-d}$ (isometric $O(d_h)$ transport via zero-copy slices in training and a 16-token ring buffer in decoding).
- **Branch Fusion & Flash-Folded ALiBi:**
$$y_t = W_O \Big[\big(\text{RMSNorm}_{\text{delta}}(o_t^{\text{delta}}) + o_t^{\text{dyadic}}\big) \odot \text{SiLU}(W_g u_t)\Big].$$
Folding ALiBi's $-m_h(i-j) = (-m_h i)\cdot 1 + 1\cdot(m_h j)$ into $Q,K$ enables native FlashAttention execution (matching dense-mask loss within $4\times 10^{-5}\text{ bpb}$, yielding $1.13\text{--}1.89\times$ speedups from `2k` to `16k` for DEEPA [$0.90\times$ at `1k`] and $1.24\text{--}3.6\times$ for Pure ALiBi).

**Table 3: Analytical computational and memory complexity ($14.4\text{M}$ models, bf16 cache).**

| Model | Train Time | Decode Step | Cache Formula (bf16 values) | Cache `@16k` | Cache `@65k` |
|---|---|---|---|---|---|
| **Pure ALiBi / RoPE** | $O(L T D^2 + L T^2 D)$ | $O(L D^2 + L T D)$ | $8 \times (2 T D)$ | $201.3\text{ MB}$ | $805.3\text{ MB}$ |
| **ALiBi-DEEPA 3:1** | $O(L T D^2 + \frac{L}{4} T^2 D)$ | $O(L D^2 + \frac{L}{4} T D)$ | $6 \times 30{,}720 + 2 \times (2 T D)$ | **$50.7\text{ MB}$** | **$201.7\text{ MB}$** |
| **Pure SheafDelta** | $O(L T D^2 + L T D d_h)$ | $O(L D^2 + L D d_h)$ | $8 \times 30{,}720$ (const.) | **$0.49\text{ MB}$** | **$0.49\text{ MB}$** |

---

## 4. Experimental Protocol

All runs use a single RTX 2000 Ada Laptop GPU ($8\text{ GB}$ VRAM), `bfloat16`, `torch.compile`, vocab $V=256$, AdamW $\beta=(0.9,0.95)$, uniform `wd=0.05`, peak LR $2\times 10^{-3}$, 100-step warmup, cosine decay to $0.1\times$, grad clip $1.0$, $B=16$, $T=1024$. All standard deviations use population std (`ddof=0`), and **negative $\Delta\text{ bpb}$ indicates lower loss (better)** throughout. Early `enwik8` benchmarks scored the first 64 KB of the 1 MB validation slice; all multi-domain, scaling, and extrapolation tables report the **full 1 MB validation set** or a fixed $512\text{ KB}$ budget.

---

## 5. Experimental Results

### 5.1 Chinchilla-Ratio Convergence and Equal Wall-Clock on `enwik8`

**Table 4: 3-seed Chinchilla-ratio convergence ($18,000$ steps, $\approx 20.4\text{ tokens/param}$, $7.4$ repeated epochs over $40\text{ MB}$ `enwik8`), equal folded-kernel wall-clock ($13,300$ steps), and weight-decay ablation ($1,500$ steps; 64 KB slice).**

| Experiment / Model (3 seeds) | Steps | Val `@1024` | Per-Seed `@1024` | `@2048` | `@4096` | `tok/s`\* |
|---|---|---|---|---|---|---|
| **18k: ALiBi-DEEPA 3:1 (`h6`)** | 18,000 | **0.7794 ± .0025** | **.7820/.7761/.7801** | **0.7620 ± .0021** | **0.8246 ± .0034** | 80.6k |
| **13.3k (Iso-Time): DEEPA (`h6`)** | 13,300 | **0.7770 ± .0010** | **.7776/.7756/.7779** | **0.7585 ± .0012** | **0.8183 ± .0021** | 80.9k |
| **18k: Pure ALiBi** | 18,000 | 0.7863 ± .0049 | .7924/.7861/.7803 | 0.7664 ± .0047 | 0.8258 ± .0041 | 85.3k |
| 13.3k (Equal-Steps): Pure ALiBi | 13,300 | *[Scheduled overnight control run; to be inserted upon completion]* | | | | |
| 1.5k Uniform-WD: DEEPA (`h6`) | 1,500 | **1.0291 ± .0016** | 1.029/1.031/1.027 | **1.0092 ± .0022** | 1.0869 ± .0057 | 80.7k |
| 1.5k Split-WD: DEEPA (`h6`) | 1,500 | 1.0324 ± .0032 | 1.032/1.029/1.036 | 1.0121 ± .0020 | 1.0887 ± .0007 | 81.3k |
| 1.5k Uniform-WD: Pure ALiBi | 1,500 | 1.0369 ± .0012 | 1.038/1.036/1.036 | 1.0096 ± .0015 | **1.0855 ± .0019** | 85.5k |
| 1.5k Split-WD: Pure ALiBi | 1,500 | 1.0407 ± .0091 | 1.031/1.053/1.038 | 1.0133 ± .0085 | 1.0888 ± .0060 | 86.6k |

*\*Measured with dense-mask ALiBi. With Flash-ALiBi folding at $B=16, T=1024$, Pure ALiBi reaches $\sim 117.0\text{k tok/s}$ vs. DEEPA's $86.5\text{k tok/s}$. Equal wall-clock under the folded kernel corresponds to $13,300$ vs. $18,000$ steps ($\approx 42\text{ min}$ each); on the dense-mask runs actually executed, DEEPA also used $22\%$ less time ($44.9$ vs. $57.6\text{ min}$). One hypothesis for DEEPA's 13.3k run beating its own 18k run is mild over-repetition of the 40 MB pool at 7.4 epochs (pending Pure ALiBi 13.3k control).*

### 5.2 Parameter Scalability ($1.85\text{M} \to 38.34\text{M}$ at Matched 5 Tokens/Param)

**Table 5: Parameter scaling sweep on `enwik8` at matched $5\text{ tokens/param}$ (seed 0, full 1 MB validation; $\Delta = \text{DEEPA} - \text{Pure ALiBi}$, negative = DEEPA better).**

| Scale ($D/L/H_a/H_m$) | Model | Params | Steps (Tokens) | Full Val bpb | `@1024` (64K) | $\Delta$ Full Val | Peak MB |
|---|---|---|---|---|---|---|---|
| 2M (192/4/4/3) | ALiBi-DEEPA | 1.85M | 563 (9.2M) | **2.1117** | **1.7638** | **$-$0.2977** | 703 |
| 2M (192/4/4/–) | Pure ALiBi | 1.82M | 563 (9.2M) | 2.4094 | 2.0407 | — | 586 |
| 6M (256/8/4/4) | ALiBi-DEEPA | 6.44M | 1,966 (32.2M) | **1.5983** | **1.0523** | **$-$0.0213** | 1,773 |
| 6M (256/8/4/–) | Pure ALiBi | 6.36M | 1,966 (32.2M) | 1.6196 | 1.0749 | — | 1,444 |
| 14M (384/8/8/6) | ALiBi-DEEPA | 14.44M | 4,406 (72.2M) | **1.3938** | **0.8765** | **$-$0.0362** | 2,719 |
| 14M (384/8/8/–) | Pure ALiBi | 14.26M | 4,406 (72.2M) | 1.4301 | 0.9058 | — | 2,241 |
| 38M (512/12/8/8) | ALiBi-DEEPA | 38.34M | 11,701 (191.7M) | **1.2349** | **0.7304** | **$-$0.0275** | 1,634\* |
| 38M (512/12/8/–) | Pure ALiBi | 37.89M | 11,701 (191.7M) | 1.2624 | 0.7523 | — | 4,480 |

### 5.3 Zero-Shot Length Extrapolation to $16\times$ Training Length (3 Seeds)

**Table 6: Zero-shot length extrapolation on `enwik8` (3 seeds, trained at $T=1024$, scored on the identical $512\text{ KB}$ validation budget). See [fig2_extrap_scaling.pdf](file:///home/mukundan/learning/llm/Research/Research/hsnlm/paper/figs/fig2_extrap_scaling.pdf).**

| Model (3 seeds) | $L=1024$ | $L=2048$ | $L=4096$ | $L=8192$ | $L=16384$ |
|---|---|---|---|---|---|
| **ALiBi-DEEPA 3:1 (`h6`)** | **1.507 ± .008** | **1.487 ± .009** | **1.477 ± .010** | **1.472 ± .010** | **1.469 ± .010** |
| **Pure ALiBi** | 1.528 ± .003 | 1.500 ± .004 | 1.486 ± .004 | 1.479 ± .004 | 1.475 ± .004 |
| **RoPE GPT** | 1.588 ± .012 | 2.352 ± .070 | 3.454 ± .187 | 4.519 ± .179 | 5.299 ± .156 |

### 5.4 Multi-Domain and Multi-Modal Generalization (8 Datasets)

**Table 7: Multi-domain and multi-modal full-validation bpb (mean $\pm$ population std, 3 seeds unless noted; lower is better). See [fig3_multidomain.pdf](file:///home/mukundan/learning/llm/Research/Research/hsnlm/paper/figs/fig3_multidomain.pdf).**

| Domain (Metric) | ALiBi-DEEPA (`h6`) | Pure ALiBi | RoPE GPT | Pure SheafDelta |
|---|---|---|---|---|
| **Python Source Code** (bpb) | **1.1416 ± .0057** | 1.1883 ± .0041 | 1.2131 ± .0075 | 1.3745 ± .0096 |
| **Hindi Wikipedia** (UTF-8) | **0.8745 ± .0016** | 0.9586 ± .0072 | 1.0081 ± .0261 | 0.8766 ± .0006 |
| **Chinese Wikipedia** (UTF-8) | 2.0581 ± .0069 | **2.0250 ± .0072** | 2.2617 ± .0071 | 2.0833 ± .0026 |
| **DNA, hg38 chr20** (bits/base) | 1.7420 ± .0002 | 1.8389 ± .0105 | 1.7881 ± .0020 | **1.7384 ± .0010** |
| **CIFAR-10 Pixels** (bits/dim) | 4.4382 ± .0136 | 4.6570 ± .0098 | **3.9661 ± .0084**† | 4.4937 ± .0120 |
| **Speech 16 kHz $\mu$-law** (bps) | **3.3300 ± .0043** | 3.4436 ± .0112 | 3.4116 ± .0159 | 3.3311 ± .0056 |
| **WikiText-103** (1.5k, 3 seeds) | 1.6275 ± .0016 | **1.6154 ± .0020** | 1.7382 ± .0090 | — |
| **WikiText-103** (4.4k, 2 seeds) | **1.4250 ± .0017** | 1.4404 ± .0006 | — | — |

*†Diagnostic: `RoPE-DEEPA 3:1 (h6)` scores **3.9470 ± .0098** on CIFAR-10, beating RoPE GPT on $3/3$ seeds. Against Pure SheafDelta ($14.77\text{M}$, zero attention), ALiBi-DEEPA achieves 3 clear wins (code $-0.233$, Chinese $-0.025$, images $-0.056$), 2 ties within $\sim 1\sigma$ (Hindi, Audio), and 1 loss (DNA).*

### 5.5 Training Speed, Inference Decoding, and 8 GB Serving

**Table 8: Compiled training step latency ($B=1$, no checkpointing), prefill latency, and peak training VRAM ($14.4\text{M}$, FlashAttention-folded ALiBi for both DEEPA and Pure ALiBi vs. FlashAttention RoPE GPT). See [fig4_systems_overhead.pdf](file:///home/mukundan/learning/llm/Research/Research/hsnlm/paper/figs/fig4_systems_overhead.pdf).**

| Length $T$ | DEEPA Train | Pure ALiBi Train | RoPE Train | DEEPA Prefill | Pure ALiBi Prefill | RoPE Prefill | DEEPA MB | ALiBi / RoPE MB |
|---|---|---|---|---|---|---|---|---|
| 1,024 | 22.8 ms | **17.0 ms** | 18.3 ms | 5.0 ms | **2.8 ms** | 3.6 ms | 414.8 | **349.4** / 359.2 |
| 2,048 | 31.8 ms | **28.1 ms** | 31.1 ms | 6.3 ms | **5.8 ms** | 8.6 ms | 543.1 | **482.6** / 494.9 |
| 4,096 | **54.0 ms** | 57.4 ms | 62.5 ms | **14.5 ms** | 15.6 ms | 18.7 ms | 855.9 | **746.7** / 774.7 |
| 8,192 | **115.2 ms** | 137.9 ms | 150.9 ms | **31.2 ms** | 42.8 ms | 46.4 ms | 1,487.4 | **1,263.2** / 1,324.3 |
| 16,384 | **281.3 ms** | 410.5 ms | 440.4 ms | **76.6 ms** | 125.1 ms | 135.0 ms | 2,749.3 | **2,283.8** / 2,408.6 |

**Table 9: CUDA-graph autoregressive decoding latency (ms/token, best ALiBi implementation per regime: folded kernel at $B=1$, dense-mask path at $B=32$) and maximum 8 GB GPU serving capacity.**

| Context | Batch | DEEPA (ms) | Pure ALiBi | RoPE GPT | DEEPA 8 GB Cap. | Pure ALiBi 8 GB Cap. | RoPE 8 GB Cap. |
|---|---|---|---|---|---|---|---|
| 1,024 | 1 / 32 | 0.90 / **2.16** | **0.68** / 3.33 | 0.89 / 4.23 | — | — | — |
| 4,096 | 1 / 32 | **0.95** / **4.11** | 0.95 / 12.18 | 1.09 / 14.34 | **576 seq, 8,537 tok/s** | 144 seq, 2,781 tok/s | 144 seq, 2,283 tok/s |
| 16,384 | 1 / 32 | **1.22** / **12.71** | 1.91 / 46.75 | 1.91 / 56.05 | **144 seq, 2,653 tok/s** | 36 seq, 700 tok/s | 36 seq, 578 tok/s |
| 65,536 | 1 / 32 | **2.19** / **47.47** | 6.01 / OOM | 5.51 / OOM | **36 seq, 688 tok/s** | 9 seq, 163 tok/s | 9 seq, 144 tok/s |

### 5.6 Architectural Ablations

**Table 10: 3-seed mixer-internal ablations ($1,500$ steps, `enwik8` 64 KB slice): `h6` is adopted for matched quality, $4.7\times$ tighter seed variance, and higher throughput.**

| Variant (3 seeds each) | Val `@1024` | `@2048` | `@4096` | `tok/s` | Verdict |
|---|---|---|---|---|---|
| v1 ALiBi-DEEPA ($H_{\text{mix}}=8, d_h=48$) | **1.0273 ± .0076** | **1.0067 ± .0066** | **1.0831 ± .0050** | 78.5k | Baseline |
| **`h6` ($H_{\text{mix}}=6, d_h=64$)** | **1.0291 ± .0016** | **1.0092 ± .0022** | **1.0869 ± .0057** | **80.7k** | **Adopt ($4.7\times$ tighter std)** |
| `decay` (`linspace(-1, 4.5)`) | 1.0452 ± .0091 | 1.0257 ± .0103 | 1.1037 ± .0113 | 75.4k | Reject ($+0.018$) |
| `h6_postnorm_scaled` | 1.0587 ± .0069 | 1.0401 ± .0069 | 1.1173 ± .0083 | 82.0k | Reject ($+0.030$) |
| `postnorm` (unscaled) | 1.0715 ± .0148 | 1.0539 ± .0163 | 1.1306 ± .0217 | 74.5k | Reject ($+0.044$) |
| `h6_token_shift` ($q,k$ shift) | 1.0751 ± .0103 | 1.0566 ± .0099 | 1.1317 ± .0059 | 77.6k | Reject ($+0.046$) |

### 5.7 Long Range Arena, Causal $\beta$ Control, and Converged Synthetic Suite

**Table 11: Long Range Arena test accuracy ($D=256, L=4, 5,000$ steps, $B=32$, mean $\pm$ population std over 2 seeds).**

| LRA Task | ALiBi-DEEPA (`h6`) | Pure ALiBi | RoPE GPT |
|---|---|---|---|
| ListOps ($\le 2\text{k}$) | 19.9 ± 0.3% | 19.4 ± 0.6% | 18.2 ± 0.3% |
| IMDb bytes (`4k`) | **75.8 ± 0.3%** | 63.7 ± 0.1% | 54.6 ± 3.1% |
| sCIFAR grayscale (`1k`) | **39.6 ± 0.1%** | 36.8 ± 0.5% | 34.9 ± 0.9% |

**Table 12: Causal parity control ($D=128, L=4$, curriculum $8\to 64$, up to 20k steps, 3 seeds, identical parameter counts).**

| Model / $\beta$ Regime | Acc `@64` (Train) | Acc `@256` ($4\times$) | Steps to Solve |
|---|---|---|---|
| ALiBi-DEEPA, $\beta \in (0,1)$ | 99.9 / 99.3 / 99.8% | 66.4 / 64.5 / 63.7% | 11.5k / 12.5k / 12.5k |
| Pure SheafDelta, $\beta \in (0,1)$ | 99.8 / 99.9 / 99.6% | 64.2 / 64.4 / 64.2% | 11.5k / 11.0k / 17.5k |
| **ALiBi-DEEPA, $\beta \in (0,2)$** | **100 / 100 / 100%** | **90.2 / 86.6 / 90.5%** | **2.0k / 2.0k / 2.0k** |
| **Pure SheafDelta, $\beta \in (0,2)$** | **100 / 100 / 100%** | **93.0 / 80.6 / 98.8%** | **2.0k / 2.0k / 2.0k** |

**Table 13: Converged synthetic capability suite ($D=128, L=4$ [1 attention layer in hybrid], except `Copy_paper` which uses the paper $14.4\text{M}$ [$D=384, L=8$] model; up to 20k steps, LR grid, 3 seeds; accuracy \% per seed, `[steps]`, `x` = unsolved).**

| Task | ALiBi-DEEPA 3:1 (Ours) | Pure ALiBi | RoPE GPT | Pure SheafDelta |
|---|---|---|---|---|
| **MQAR-8** (`T64`) | **100/100/100 [1.5–3.5k]** | 99/99/100 [6–9k] | 99/100/99 [9–9.5k] | 99/99/100 [2.5–13.5k] |
| **MQAR-32** (`T512`) | **100/100/100 [2.5k all]** | 100/100/100 [3–6.5k] | 9/9/8 [x] | 100/99/100 [4.5–6k] |
| **MQAR-64** (`T512`) | 100/100/6 [4.5–6k] | 100/6/100 [3.5–9k] | 6/6/6 [x] | 100/6/100 [10k] |
| **Parity** (`64→256`) | **100→90/87/90 [2k]** | 46–57→31–52 [x] | 54–56→44–52 [x] | **100→93/81/99 [2k]** |
| **$S_3$ Group** (`64→256`) | **99–100→93/95/48 [4–13k]** | 29–30→20/20/20 [x] | 28→20/20/19 [x] | **99–100→92/89/96 [2.5–6k]** |
| **$S_5$ Group** (`64→256`) | 4/2/3→2/1/1 [x] | 2/2/2→1/1/1 [x] | 2/2/2→1/1/1 [x] | 4/6/6→2/2/2 [x] |
| **Copy\_paper** (`128→512`) | 100→100/98/38 [2–4k] | **100→96/95/87 [0.5k]** | 100→0/0/0 [0.5k] | 99–100→14/18/15 |
| **Copy** (`128→512`) | 100/2/100→96/0/59 | **100→99/98/99 [1k]** | 100→0/0/0 [0.5k] | 99→24/26/19 |
| **SelCopy** (`T256`) | 99/99/99 [3.5–10.5k] | 100/99/99 [7–10k] | 99/100/99 [4–6.5k] | **100/100/99 [3–4k]** |

ALiBi-DEEPA achieves 2 clear LRA wins (IMDb $+12.1\%$ vs. Pure ALiBi, sCIFAR $+2.8\%$) and ties at chance on ListOps; capping $\beta \in (0,1)$ provides direct causal evidence that negative eigenvalues drive $4\times$ parity generalization; and ALiBi-DEEPA 3:1 is the only model with non-trivial success across all three synthetic families (recall, parity/$S_3$ state tracking, and length-general copying), while Pure ALiBi copies more reliably at $4\times$ ($87\text{--}96\%$) and $4\times$ faster. A learning-rate robustness grid for copying and MQAR-64 is scheduled to quantify seed sensitivity.

---

## 6. Limitations and Future Work

1. **Short-Context Kernel Overhead ($T \le 1\text{k}$):** At $B=16, T=1024$, small-kernel launch overhead makes DEEPA $\sim 26\%$ slower than folded-kernel Pure ALiBi ($86.5\text{k}$ vs. $117.0\text{k tok/s}$) and $\sim 12\%$ slower than RoPE GPT ($98.5\text{k tok/s}$), while using $14\%$ more training VRAM at `16k` than RoPE GPT ($2,749$ vs. $2,409\text{ MB}$) and $20\%$ more than folded-kernel Pure ALiBi ($2,284\text{ MB}$). Fusing the Householder dyadic transport and output projection into a single Triton kernel is the primary systems next step.
2. **Monotonic ALiBi Penalty on 2D Raster Strides:** As shown by the CIFAR-10 `RoPE-DEEPA` diagnostic ($3.947$ vs. `ALiBi-DEEPA` $4.438\text{ bpd}$), ALiBi's monotonic distance decay handicaps fixed long-stride 2D dependencies.
3. **Copying Speed and Seed Sensitivity:** DEEPA learns exact string copying $4\times$ slower than Pure ALiBi ($2\text{--}4\text{k}$ vs. $0.5\text{k}$ steps) and exhibits seed/LR sensitivity on MQAR-64 and copying (failing one seed on each).
4. **Unsolved $S_5$ and Chance-Level ListOps:** All four single-step models fail $NC^1$-complete $S_5$ (motivating multi-reflection products as future work), and all models remain near chance on LRA ListOps.
5. **Byte-Level Scope Only:** All experiments are byte-level ($V=256$); subword (BPE) pretraining and downstream zero-shot benchmarks remain to be evaluated.
6. **Scale Beyond $38\text{M}$ Converged Parameters:** While scaling is validated from $1.85\text{M}$ to $38.34\text{M}$ at $5\text{ tokens/param}$ (and VRAM fits $206\text{M}$ in $5.3\text{ GB}$), full Chinchilla training at $100\text{M}\text{--}1\text{B}$ scale remains future work.

---

## 7. Conclusion

We presented **DEEPA** (**D**yadic-Delta **E**nhanced-sheaf, **E**fficient **P**airing with **A**LiBi), a $3:1$ hybrid sequence model pairing negative-eigenvalue gated delta-rule matrix memory and parallel isometric dyadic sheaf diffusion with FlashAttention-folded ALiBi attention. Across 8 real-world datasets, 4 parameter scales ($1.85\text{M}\to 38.34\text{M}$), 18,000-step and equal-wall-clock convergence, 3-seed $16\times$ extrapolation, Long Range Arena ($2/3$ clear wins, $1$ tie at chance), and a 9-task converged synthetic suite, ALiBi-DEEPA outperforms Pure ALiBi on $7/8$ domains and all 4 parameter scales, achieves 3 wins, 2 ties, and 1 loss against Pure SheafDelta across 6 domains, is the sole model with non-trivial success across recall, parity/$S_3$ state tracking, and $4\times$ copying, cuts inference KV cache by $4\times$, increases 8 GB serving throughput by $3.7\text{--}4.8\times$ vs. RoPE GPT ($3.1\text{--}4.2\times$ vs. Pure ALiBi), and accelerates `4k–16k` training by $1.16\text{--}1.57\times$ over FlashAttention RoPE GPT ($1.06\text{--}1.46\times$ over folded-kernel Pure ALiBi).

---

## Declarations
- **Funding:** No external funding was received for conducting this study.
- **Competing Interests:** The authors declare no competing financial or non-financial interests.
- **Data Availability:** All datasets used in this study are publicly available (`enwik8`, `WikiText-103`, `CIFAR-10`, `GRCh38/hg38`, `Speech Commands`, Wikipedia dumps, and `CodeParrot`).
- **Code Availability:** Complete training scripts, Triton-fused kernels, and benchmark suites will be released publicly upon acceptance.

---

## References
*(Identical 30 numbered references as in [DEEPA_Springer.tex](file:///home/mukundan/learning/llm/Research/Research/hsnlm/paper/DEEPA_Springer.tex).)*
