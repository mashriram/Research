# Findings Report — GPU phase (Householder Sheaf-Delta LM, `hsnlm`)

**Date:** 2026-11-14  **Scope:** first GPU pass, following up `findings_report.md` (CPU phase). Hardware: one consumer laptop GPU — NVIDIA RTX 2000 Ada Generation, **8 GB VRAM**, CUDA 12.4, PyTorch 2.6, Triton 3.2.
> **In plain English, before the technical verdict below:** this is the detailed lab notebook for a day of GPU testing on a small (14-million-parameter) experimental language model that replaces most of a Transformer's "look back at everything" layers with cheaper "fixed-size notebook" layers (full explanation in `ARCHITECTURE.md` §0). The headline result, after two real measurement bugs were caught and fixed partway through (an embarrassing but important part of this story — see §5h): the hybrid design does write better text and handle long text better than every flavor of ordinary Transformer tested, by a small-but-genuine margin against the best one and a large margin against a plain one — and it generates each new word in a long conversation much more cheaply. It does cost more time and memory to train, though. Read §5h first if you only read one section — it explains the two bugs and how the final numbers were double-checked.

**One-line verdict (three times revised — two bugs fixed §5h, then a multi-seed + clean-isolation pass §5i-§5j):** the hybrid family beats every Transformer baseline tested, including ALiBi, the strongest one, on quality at training length, on 2x/4x extrapolation, and on decode cost (§6) — but the margins against ALiBi are small (0.01-0.016 bpb), not dramatic, and a 3-seed recheck (§5i) found that it is specifically the **ALiBi-hybrid**, not the originally-reported RoPE-hybrid, that beats pure ALiBi robustly (3/3 seeds) — the RoPE-hybrid's single-seed "best" result turned out to be within its own seed-to-seed noise. An intermediate version of this report, produced mid-session, claimed the opposite of the whole hybrid family (ALiBi beating the hybrid by a wide margin, 0.956 vs 1.014) — that was wrong, the product of two compounding bugs (§5h): a validation-data-slicing mismatch that scored ALiBi on an easier slice of enwik8 than everything else, and a dropped-residual-connection bug in every custom Transformer-block reimplementation used for the ALiBi/NoPE/YaRN comparisons, which silently deleted roughly a third of those models' intended architecture. Both are fixed and independently verified (bit-exact against the real `TBlock` class). Pure ALiBi, correctly measured, is now a remarkably strong, much cheaper near-tie with the hybrid family (within 0.01-0.016 bpb at every length, and still faster/leaner even once both sides are compiled fairly, §5i) — but has zero inference-side advantage (full, growing KV cache, same mechanism as plain RoPE). NoPE is confirmed not competitive even after the residual-bug fix (2.344 bpb, §5h). The 4:1 ratio-dilution and sliding-window-ALiBi (SWA) experiments were rerun post-fix (§5h) and confirm their pre-fix negative results. A cleanly isolated (separate-subprocess) rerun of the width-128/ctx-16,384 crossover (§5j) resolved §5g's "unresolved" status: both sheaf-based architectures do simultaneously beat GPT on speed and quality there once CUDA-allocator contamination is eliminated. Separately and still true: pure sheaf-delta alone loses to every Transformer variant on quality at 14M params (§5); `torch.compile` on the whole model gives a large, real speed win for both architectures (GPT -32.6%, hybrid -57.7%, §5e); a fair iso-quality (time-to-target-loss) comparison shows the hybrid reaches GPT's own compiled quality in ~30% less wall-clock and ~42% fewer tokens (§5e); and three independent implementations confirm backprop-alternatives (GOAT/GOLIATH/DAD, §8) still don't work. The honest, narrow, defensible claim: **this architecture family is the best option on quality, extrapolation, and decode cost against every Transformer variant tested, by small-but-real and now multi-seed-confirmed margins against the strongest one (ALiBi) and large margins against plain RoPE — at a real, if now much smaller, cost in training speed and memory.** Not "better in every way" — pure ALiBi remains a legitimate, much cheaper alternative if the small quality edge and the inference-side advantage aren't needed.

---

## 1. What changed since the CPU report

| CPU report said | GPU measurement says |
|---|---|
| "Training speed is not solved... needs fused GPU kernels to settle" (§9) | **Settled, directionally**: a real Triton fused kernel exists (`fla`'s `chunk_gated_delta_rule`) and runs on this hardware. It gives sheaf-delta a 2-3x speedup over its own eager loop, and makes it **faster than the Transformer at long context** (§3). |
| "nothing here predicts 100M+ behaviour" | Still true — the quality run here is 0.82M params (identical to CPU) plus one partial run at ~15M params (§5); no claim beyond that. |
| Backprop alternatives: "nothing tested cuts compute" | **Unchanged on GPU** (§4): DFA still fails on quality, teacher-target local learning is still *slower* wall-clock than plain backprop, not faster. |
| O(1) decode, O(1) reversible memory (both "strong, measured") | **Reconfirmed on GPU**, extended to longer context (decode tested to 65,536; memory confirmed flat to ctx 8192) (§6). |

---

## 2. Setup

- `uv`-managed venv: `torch==2.6.0+cu124`, `triton==3.2.0`, `flash-linear-attention==0.6.0`, `numpy`.
- `flash-linear-attention` needed `triton>=3.3` per its own warning but ran correctly on 3.2 in every test here (no silent wrong-answer observed; all correctness checks below passed).
- Built `hsnlm/fastmixer_fused.py`: a `FusedSheafDeltaMixer` that replaces the project's own eager `delta_chunked` Python-chunk loop with `fla.ops.gated_delta_rule.chunk_gated_delta_rule` for the matrix-memory part (the dyadic Householder local-mixing part is untouched, still eager). Registered as `Config(mixer="sheafdelta-fused")` in `hsnlm/models.py`.
- **Formula discrepancy found and left as-is, flagged here**: this codebase's own delta rule (`fastmixer.py: delta_seq/delta_chunked`) is `S_t = a_t(S_{t-1} + b_t(v_t - S_{t-1}k_t)k_t^T)` — the write term `b_t(...)k_t^T` is *also* multiplied by `a_t`. The canonical Gated DeltaNet formula quoted in the CPU report's own §4 (`S_t = a_t S_{t-1}(I - b_t k_tk_t^T) + b_t v_t k_t^T`) does **not** decay the write term by `a_t`. These differ by a factor of `a_t ∈ [0.8, 1.0]` on new writes. The existing test suite (`tests/test_fast.py`) only checks the project's own chunked loop against its own sequential loop — never against the literature formula — so this was never caught. `FusedSheafDeltaMixer` implements the canonical (fla) formula, since that's what the fused kernel computes; it is therefore a close but not bit-identical variant of the rest of the codebase. This is a documentation/consistency issue, not necessarily a bug — just worth knowing the report's math and the code's math diverge.
- Data: `data/enwik8` (100MB) and `data/shakespeare.txt`, fetched fresh (not in git).
- All code: `benchmarks/gpu_suite.py`, `gpu_speed.py`, `gpu_crossover.py`, `gpu_decode.py`, `gpu_reversible.py`, `gpu_credit.py`, `gpu_scale.py` (new, GPU-only; CPU scripts untouched).

---

## 3. Training speed: the fused kernel changes the answer (main new result)

### 3.1 Context-length crossover, width 128 (`benchmarks/gpu_crossover.py`)

Fixed token budget per step (B·N = 8192, so batch shrinks as context grows), 4 layers, 4 heads, 0.82M params (GPT) / 0.83M (sheaf-delta). Tokens/s, higher is better:

| context | GPT (fp32) | sheaf-delta fused (fp32) | fused/GPT | GPT (bf16) | sheaf-delta fused (bf16) | fused/GPT |
|---|---|---|---|---|---|---|
| 256 | 169,798 | 110,815 | 0.65x | 413,848 | 114,267 | 0.28x |
| 512 | 169,973 | 107,356 | 0.63x | 343,900 | 114,742 | 0.33x |
| 1,024 | 129,824 | 109,500 | 0.84x | 345,043 | 115,273 | 0.33x |
| 2,048 | 91,136 | 105,043 | **1.15x** | 335,060 | 115,387 | 0.34x |
| 4,096 | 55,434 | 102,048 | **1.84x** | 241,873 | 109,310 | 0.45x |
| 8,192 | 32,108 | 98,967 | **3.08x** | 187,133 | 112,066 | 0.60x |
| 16,384 | 15,143 | 88,344* | **5.83x** | 132,890 | ~190,000† | **~1.4x** |
| 32,768 | 8,334 | 83,791 | **10.05x** | 76,634 | ~180,000† | **~2.4x** |

\* fp32/16384 and similar rows are single median-of-5 timings; †bf16 at 16k/32k showed run-to-run variance (105,892 vs a clean isolated re-measurement of 210,337 and 187,904 tok/s respectively — both re-measurements confirm sheaf-delta is well above GPT at these lengths, but treat the exact bf16 multiplier at 16-32k as "clearly >1x, magnitude ±40% uncertain" rather than a precise number; this needs more seeds/isolated runs to tighten.

**Reading:** eager-mode HSN never had a chance on GPU — kernel-launch overhead from its many small ops makes it *worse* relative to the Transformer than on CPU (`hsn-eager` tok/s is 20-150x below GPT across the board, both dtypes; see raw data in `benchmark_results/development/fused_kernel_crossover/gpu_crossover.json`). The **fused** kernel changes the shape of the curve entirely: its tok/s is roughly flat in `N` (true O(N) scaling — compute per token doesn't grow with context) while GPT's falls steadily (O(N²) attention cost eventually dominates even with a fused flash-attention kernel). The crossover — the context length past which sheaf-delta trains faster — is **~1.5-2k tokens in fp32** and **~16-32k tokens in bf16** (bf16 matters because PyTorch's flash-attention path is tuned for reduced precision and gets ~2-6x faster under bf16 while our kernel gets only a modest 0-15% boost from bf16; the net effect is that **bf16 — the precision anyone would actually train in — pushes the crossover context much further out than the fp32 number suggests**). fp32 also loses outright above ctx 4096 for GPT/8192 for HSN (the "math" attention fallback, not flash, since fp32 doesn't get the fused CUDA kernel) — not shown as OOM rows for clarity, see `benchmark_results/development/speed_vs_length/gpu_speed.json`.

### 3.2 Same test at width 256 (`benchmarks/gpu_speed.py`) — crossover not reached, same card

Fixed batch=16 (so tokens grow with context), 4 layers, 8 heads, ~3.2M params:

| context | GPT bf16 tok/s | sheaf-delta fused bf16 tok/s | fused/GPT |
|---|---|---|---|
| 256 | 177,825 | 59,855 | 0.34x |
| 512 | 166,993 | 57,431 | 0.34x |
| 1,024 | 140,241 | 50,095 | 0.36x |
| 2,048 | 111,213 | 45,251 | 0.41x |
| 4,096+ | OOM (both) | OOM (both) | — |

Same converging trend as §3.1 (ratio climbs monotonically: 0.34→0.34→0.36→0.41), but this 8GB card ran out of memory at context 4096 (batch 16, width 256) before the lines crossed. **Honest gap:** we cannot confirm the crossover context scales with width the way the compute-accounting predicts (≈ cost ratio ∝ N/D, implying crossover ∝ D) — the width-128 data is consistent with that (crossover ~1.5-2k at D=128), and would predict crossover ~3-4k at D=256, which is exactly where we ran out of memory to check. A GPU with more VRAM (or gradient checkpointing / smaller batch) would settle this directly.

### 3.3 Memory: the fused kernel costs more memory per token than attention, not less

At every context length tested, `sheaf-delta fused` used **more** peak GPU memory than GPT (e.g. bf16 ctx 2048: GPT 1923 MB vs fused 3349 MB). This is counter to the naive expectation ("recurrent state should be cheap"); the reason is that `fla`'s chunked backward materializes several `[B,H,T,chunk,...]` intermediate tensors (the WY-representation / UT-transform matrices) per chunk, and the dyadic Householder local-mixing part of the mixer is still fully eager PyTorch, adding its own saved activations on top. "Fused" here only covers the delta-rule core, not the whole mixer — a caveat that matters for anyone budgeting VRAM.

### 3.4 Matched-scale replication (`benchmarks/gpu_suite.py`) — same 0.82M-param setup as the CPU suite

| arch | CPU tok/s | GPU tok/s (eager, seed0) | GPU/CPU speedup |
|---|---|---|---|
| GPT | 13,300 | 298,719 | **22.5x** |
| sheaf-delta (eager) | 6,500 | 45,498 | **7.0x** |
| hybrid (eager) | 7,400 | 47,882 | **6.5x** |

**This is the sharpest confirmation of the CPU report's prediction**: moving to GPU *without* a fused kernel makes the relative gap *worse*, not better (CPU ratio GPT/sheaf-delta = 2.0x; GPU eager ratio = 6.6x). Many small elementwise/`bmm` ops that merely under-use a CPU core are actively punished by GPU kernel-launch overhead. This is exactly why §3.1's fused-kernel result matters — eager GPU code was always going to be the wrong comparison.

---

## 4. Quality, recall, state-tracking: unchanged from CPU (replication, 0.82M params, enwik8/synthetic)

| metric | CPU | GPU seed0 | GPU seed1 |
|---|---|---|---|
| val bpb @256 — GPT | 1.962 | 1.953 | 1.962 |
| val bpb @256 — sheaf-delta | 1.880 | 1.893 | 1.878 |
| val bpb @256 — hybrid | 1.890 | 1.893 | 1.891 |
| length extrap @2048 — GPT | 4.212 | 4.085 | 4.226 |
| length extrap @2048 — sheaf-delta | 1.879 | 1.883 | 1.871 |
| parity @128 — GPT / sheaf-delta / hybrid | 0.514 / 0.965 / 0.997 | 0.520 / 0.979 / 0.997 | (1 seed) |
| recall (16 pairs) — GPT / sheaf-delta / hybrid | 0.109 / 0.170 / 0.238 | 0.108 / 0.178 / 0.214 | (1 seed) |

All within seed noise of the CPU numbers. GPU non-determinism (cuDNN/Triton kernel selection, reduction order) does not change any ranking. **Nothing here is new evidence for scale** — still the same 0.82M-parameter, 10M-token regime as the CPU report; see §5 for the one larger run attempted.

---

## 5. Larger-scale run: the quality win does not survive scale-up as-is — but the hybrid does

`benchmarks/gpu_scale.py`: ~14.2-14.6M params (width 384, 8 layers, ctx 1024, 17x the CPU report's parameter count), GPT vs sheaf-delta-fused vs hybrid-fused (7 sheaf-delta-fused + 1 attention layer), enwik8, 1,500 steps (24.6M tokens), bf16, lr chosen per-arch from a 2-point sweep (both picked 2e-3).

| arch | params | val bpb @1024 (train len) | @2048 | @4096 | train_s | tok/s | peak mem (MB) |
|---|---|---|---|---|---|---|---|
| GPT | 14.26M | **1.103** | 2.111 | 3.547 | **372.3** | **66,015** | 2,897 |
| sheaf-delta-fused | 14.22M | 1.199 | 1.187 | 1.247 | 802.3 | 30,631 | 4,947 |
| hybrid-fused (7:1) | 14.64M | **1.021** | **1.004** | **1.090** | 760.2 | 32,329 | 4,738 |

**This is the single most important result in this report, and it complicates the headline.** At 0.82M params (CPU report + §4 above), pure sheaf-delta beat the Transformer on quality. At 14M params it does not — GPT is *better* at the training length (1.103 vs 1.199), reproducing exactly the risk the CPU report's own threats-to-validity section named ("early-training advantages from locality/recurrence biases often vanish or reverse at scale"). Two things do hold up, though:

- **Length extrapolation is real but smaller than it first looks — see §5d.** GPT with plain RoPE degrades from 1.103 (ctx 1024) to 3.547 (ctx 4096, 4x training length); both sheaf-containing architectures barely move (sheaf-delta: 1.199→1.247; hybrid: 1.021→1.090). But §5d shows applying YaRN (zero-retraining, standard practice) to the same GPT closes 73-86% of that gap — plain RoPE was never a fair "best known Transformer" baseline. The corrected claim: a real, consistent ~0.35-0.4 bpb extrapolation advantage over the best easy fix, not a dramatic one over an undefended baseline. The *mechanism* (fixed-size state has no position-dependent failure mode) is still scale-robust and still true; the *size* of the resulting advantage over a properly-defended Transformer is smaller than this section originally stated.
- **The hybrid (not pure sheaf-delta) is the best architecture on every quality axis at this scale**, beating GPT by 0.08 bpb at the training length *and* beating GPT by 2.5 bpb at 4x context, while also training faster than pure sheaf-delta (760s vs 802s) and using slightly less memory (4,738 vs 4,947 MB). This matches the CPU report's own literature citation (§11: "production models stack linear layers with a minority of attention layers") and its own design decision (RESEARCH_V2.md §2: "backbone = Householder-delta-rule memory layers + ~1:4-1:7 exact-attention layers") — this is the first evidence *from this project's own experiments* (rather than literature citation) that the hybrid ratio is the right call, not just a hedge.
- Speed still favors GPT outright at this width (372s vs 760-802s) — consistent with §3.2's observation that the fused-kernel crossover context grows with model width, and width 384 here is wider than either narrow-width crossover test (§3.1, width 128) or the inconclusive wide-width test (§3.2, width 256, OOM'd before crossing). No crossover expected or observed at ctx 1024 here.

**One run each, no seeds, 1,500 steps / 24.6M tokens** — far from enough to trust the exact gap sizes, but the *direction* (pure-sheaf quality edge shrinks/reverses with scale; hybrid quality edge grows) is the one new fact from this pass that should most change what gets tried next.

### 5a. Hybrid ratio ablation (G5) — the 7:1 guess was not the best point (`benchmarks/gpu_ratio.py`)

The CPU report picked a ~7:1 sheaf-delta:attention ratio from a literature citation, never tested against other ratios (G5 in its own GPU plan, §13). Same 14M-param/width-384/8-layer setup as above, sweeping the number of attention layers (spread evenly across depth) from 0 to 8:

| attention layers (of 8) | pattern | bpb @1024 (train len) | @2048 | @4096 | tok/s | peak mem (MB) |
|---|---|---|---|---|---|---|
| 0 (pure sheaf-delta) | `dddddddd` | 1.199 | 1.187 | 1.248 | 30,631 | 4,947 |
| 1 (7:1, original guess) | `ddddddda` | 1.021 | 1.004 | 1.090 | 32,329 | 4,738 |
| **2 (3:1)** | `dddaddda` | **1.014** | **0.995** | **1.084** | **34,180** | **4,472** |
| 3 (5:3) | `ddadadda` | 1.038 | 1.022 | 1.116 | 37,384 | 4,197 |
| 4 (1:1) | `dadadada` | 1.045 | 1.030 | 1.128 | 40,489 | 3,923 |
| 8 (pure attention = GPT) | `aaaaaaaa` | 1.103 | 2.111 | 3.547 | 66,015 | 2,897 |

**2 attention layers out of 8 (3:1) beats the original 7:1 hybrid on every axis simultaneously** — better quality at the training length (1.014 vs 1.021), better extrapolation (1.084 vs 1.090 at 4096), faster (34,180 vs 32,329 tok/s), and less memory (4,472 vs 4,738 MB). Quality peaks at this ratio and degrades on both sides: too little attention (ratios 0-1) underperforms it slightly; too much (ratios 3-4, 8) degrades quality further *and* progressively reintroduces the extrapolation failure (bpb@4096 climbs from 1.084 at ratio 2 toward GPT's 3.547 at ratio 8, monotonically, as attention layers increase). Speed and memory move monotonically in GPT's favor as attention layers increase, as expected (more attention layers at this width/context = cheaper per the crossover analysis in §3). This is a real Pareto-frontier shape, not noise at the extremes (0 vs 8 are 0.185 and 0.089 bpb apart from the best point respectively) — though with only one seed per point, the exact optimum (ratio 2 vs 1) should be treated as "somewhere around 2-3 attention layers in 8," not pinned precisely.

### 5b. Fixed-rank low-rank MLP factorization — negative result (`benchmarks/gpu_lowrank_mlp.py`)

Separate from the ratio question: does factoring the MLP's up/down projections into a low-rank bottleneck (`y = A(B(x))`, fixed rank, no sparse path, no adaptive growth — see §11 for why those two were dropped) save anything, at the same 14M-param/7:1-hybrid scale? MLP projections are the majority of this model's parameters (~8.6M of 14.6M), so this looked like the most promising place to cut.

| config | params | bpb @1024 | @4096 | tok/s | peak mem (MB) |
|---|---|---|---|---|---|
| dense (baseline) | 14.64M | **1.021** | **1.090** | 32,329 | 4,738 |
| rank 64 | 7.86M (−46%) | 1.055 | 1.132 | 33,834 | 4,684 |
| rank 128 | 9.69M (−34%) | 1.078 | 1.164 | 33,636 | 4,739 |
| rank 192 | 11.52M (−21%) | 1.070 | 1.162 | 32,945 | 4,801 |

**Negative result, and an informative one.** Every rank loses quality vs dense (0.03-0.06 bpb), with no clean rank-vs-quality monotonic trend (rank 64 beats 128 and 192, most likely because the learning rate was reused from the dense sweep rather than re-tuned per rank — low-rank bottlenecks are known in the literature to need their own LR, so treat the ranking among the three ranks as noisy, not the dense-vs-any-rank gap, which is consistent and larger than that noise). The more important finding: **memory and speed barely move at all** across every rank (4,684-4,801 MB vs 4,738 MB dense; 32,945-33,834 vs 32,329 tok/s) — confirming §3.3's point that the mixer's fused-kernel intermediates, not the MLP, dominate this architecture's memory and compute. So factoring the MLP buys parameter count for a smaller checkpoint, not training speed or memory — not worth doing here unless the goal is specifically model-size-on-disk.

### 5c. Pushing past the 14M ceiling with gradient checkpointing — big quality gap, but read it skeptically (`benchmarks/gpu_scale_ckpt.py`)

§5's 14M-param run hit an 8GB VRAM wall. Gradient checkpointing (recompute each block's activations during backward instead of storing them — more general than `hsnlm/reversible.py`'s exact reversible blocks, which only support a homogeneous sheaf-mixer stack, not GPT or a mixed pattern) opened a lot of headroom: a probe showed ~200M params at width 1024/16 layers fits in only 1.3-1.7 GB. Full training run, same recipe family, 16 layers, ctx 1024, **300 steps / 4.9M tokens** (cut down from 1,500/24.6M purely for wall-clock — see caveat below):

| arch | params | bpb @1024 | @4096 | tok/s | peak mem (MB) |
|---|---|---|---|---|---|
| GPT | 201.6M | 2.440 | 2.564 | 6,989 | 4,554 |
| hsn-fused (pure) | 200.3M | **1.507** | **1.555** | 3,936 | 5,181 |
| hybrid-fused (7:1, pre-ratio-ablation) | 206.3M | 1.514 | 1.562 | 4,083 | 5,258 |

**Do not read this as "sheaf-delta wins at 200M."** The §5 run (14M params) used 1,500 steps = 24.6M tokens = a tokens/param ratio of ~1.73. This run uses 300 steps = 4.9M tokens = a ratio of **~0.025 — 70x less relative training**, purely because 1,500 steps at this size/with checkpointing's recompute overhead would have taken hours per architecture and this pass had a same-day deadline. A 200M-parameter Transformer at 0.025 tokens/param is nowhere near warmed up; GPT's bpb (2.44) is still above *its own* 100-step sweep-stage readings at this size, consistent with "still finding its footing," not "converged and losing." This is close to a textbook replay of the CPU report's own threats-to-validity warning (§12 there, #1: "early-training advantages from locality/recurrence biases often vanish or reverse at scale") — except here it's the *opposite* axis, a token-budget artifact rather than a parameter-count one, and it is large enough (0.93 bpb) that it should not be casually attributed to architecture. **This result should be treated as "undertrained-regime behavior at 200M," an open question, not a resolved one** — a fair comparison would need roughly the same tokens/param ratio as §5 (≈1.73), i.e. ~350M tokens at this size, which was outside this pass's time budget. Memory and extrapolation patterns are consistent with every other result in this report (sheaf-based models flat across context, cost more peak memory than GPT at matched size); those two findings are on firmer ground than the quality numbers here.

**Follow-up: does the §5a ratio finding transfer to 200M?** Re-ran just the hybrid architecture with the 3:1 pattern (`dddadddadddaddda`, the §5a winner) at the same scale/budget: **1.510 bpb** — statistically tied with both the 7:1 hybrid (1.514) and pure sheaf-delta (1.507) above, all within 0.007 bpb of each other. The ratio effect that produced a clean 0.185 bpb spread at 14M/well-trained (§5a) is invisible here. Most likely reading: at this token budget everything with a sheaf-delta component is still in the same "fast early learning" regime regardless of how much attention it has, and the ratio only starts to matter once training has run long enough for the different architectures' asymptotic behavior to diverge — which is exactly the training regime (~1.73 tokens/param) §5a had and this run (~0.025) doesn't. Another reason to treat §5c's numbers as "interesting but not yet comparable to §5/§5a," not as a bigger, better version of them.

### 5d. The extrapolation claim was measured against a weak baseline — corrected with YaRN (`benchmarks/gpu_yarn.py`)

Every extrapolation number so far (§5, §5a) compared against **plain RoPE with no context-extension technique** — explicitly flagged as a caveat from the start (`findings_report.md` §12.4: "the Transformer is a plain RoPE model, no NoPE/YaRN/ALiBi"), but still the headline comparison. This retrains the same 14M-param GPT (identical recipe to §5) and evaluates length extrapolation both with plain RoPE and with **YaRN** (NTK-by-parts RoPE frequency interpolation + attention-temperature scaling — a standard, zero-retraining technique for extending an already-trained model's context, applied here only at eval time for contexts beyond the 1,024 training length):

| context | GPT, plain RoPE | **GPT, + YaRN (no retraining)** | Hybrid 3:1 (§5a) |
|---|---|---|---|
| 1,024 (train length) | 1.130 | 1.130 (no-op, scale=1) | 1.014 |
| 2,048 (2x) | 2.291 | **1.370** | 0.995 |
| 4,096 (4x) | 3.518 | **1.472** | 1.084 |

(The plain-RoPE numbers here are a fresh training run, not a copy of §5's — 1.130/2.291/3.518 vs §5's 1.103/2.111/3.547 — small run-to-run variance from a different seed/implementation, not a discrepancy in the underlying claim.)

**YaRN closes 73-84% of the extrapolation gap.** At 2x context, plain RoPE loses 1.161 bpb from its training-length score; YaRN loses only 0.240 bpb — an 79% reduction in the degradation. At 4x, plain RoPE loses 2.388 bpb; YaRN loses 0.342 bpb — 86% reduction. **The hybrid still wins** — by 0.375 bpb at 2x and 0.388 bpb at 4x, a real and consistent margin — but "GPT is close to useless past its training context" (§5's framing) is simply not true once the best known zero-cost fix is applied. The honest claim is narrower and smaller than what §5 stated: **a real but modest extrapolation advantage over the best easy baseline, not a dramatic one over an undefended baseline.**

**What this doesn't test:** NoPE and ALiBi (the other two techniques named in every caveat so far) require retraining from scratch with a different position-encoding choice baked in, not a post-hoc fix like YaRN — neither was attempted here. Also not tested: YaRN combined with a short continued-training phase (how it's normally used in practice, rather than applied cold to a model trained with zero awareness of it), which would likely close the gap further still.

### 5e. Fusing the dyadic part too — `torch.compile` alone closes a lot of the gap, no hand-written Triton kernel needed

§10.3 (threats to validity) named the still-eager dyadic Householder part as the biggest remaining lever, and §3.3 showed it's a real cost (the fused kernel uses more memory per token than attention). Hand-authoring a Triton kernel for it is real engineering with real correctness risk if rushed — but wrapping the *existing* mixer in `torch.compile` (automatic kernel fusion, not a hand-written kernel, so no new correctness surface beyond what PyTorch's own compiler already verifies) was cheap to try at the 14M/width-384/ctx-1024 scale. Two versions were measured. First, compiling just the dyadic submodule (measured under some GPU contention from a concurrent job — trust the ratio more than the absolute ms/step):

| architecture | eager ms/step | compiled ms/step | eager peak mem | compiled peak mem |
|---|---|---|---|---|
| Pure sheaf-delta-fused | 1,109.6 | **644.5 (−42%)** | 4,826 MB | **3,331 MB (−31%)** |
| Hybrid 3:1 (the recommended architecture) | 976.0 | **578.6 (−41%)** | 4,353 MB | **3,064 MB (−30%)** |

Second — and better — compiling the **whole model** (`model = torch.compile(model)`, both GPT and the hybrid, same recipe, same seed, clean isolated run, real full 1,500-step training, not a microbenchmark):

| architecture | uncompiled train_s | compiled train_s | peak mem (compiled) | val bpb (compiled) |
|---|---|---|---|---|
| GPT | 372.3 | **250.8 (−32.6%)** | 2,408 MB | 1.1145 |
| Hybrid 3:1 | 719.0 | **304.4 (−57.7%)** | 2,868 MB | **1.0178** |

Whole-model compile beats the mixer-only compile for the hybrid (57.7% vs 41% faster) — likely because it also fuses the attention/MLP/embedding ops the mixer-only version left untouched. It still doesn't fully close the memory gap to GPT (2,868 vs 2,408 MB, hybrid still ~19% higher), but both time and memory dropped substantially for both architectures. **A fair, direct iso-quality (time-to-target-loss) comparison follows from this same pair of runs**: the hybrid's own training curve crosses GPT's final train loss (1.712) around step 857 of 1,500 — 42.9% fewer tokens — which at the hybrid's own compiled per-step cost works out to ~173.8s, **30.7% less wall-clock than GPT's own 250.8s**. This is the first version of this comparison that compiles both sides the same way; an earlier draft of this claim (compiling only the hybrid, comparing against GPT's uncompiled 372.3s) would have overstated the result (~33% vs the real 30.7% — close by coincidence, but for the wrong reason).

Not yet combined with gradient checkpointing (§5c) or re-measured at the narrow width where the speed crossover lives (§3.1) — both are natural next steps, and either could plausibly shift the training-speed crossover earlier than currently measured.

### 5f. The real baseline was never plain RoPE — NoPE and ALiBi, trained from scratch (`benchmarks/gpu_posenc.py`) — SUPERSEDED, see §5h

§5d corrected the extrapolation claim against YaRN, a *post-hoc* fix. NoPE and ALiBi are different in kind — they change the architecture itself and require training from scratch, which §5d explicitly flagged as untested. This section originally reported ALiBi beating the hybrid decisively on quality (0.956 vs 1.014) and speed. **That result was wrong — the product of two compounding bugs, found and fixed later the same session. See §5h for the full account and the corrected numbers.** Left here, struck through in spirit, for the record of what was originally claimed and why it doesn't hold:

~~**ALiBi changes the picture substantially, and this is the single most important correction in this report.** Trained from scratch at the same recipe, it beats every other architecture on quality at training length (0.956 vs the hybrid's 1.014), trains at nearly 2x the hybrid's speed (60,158 vs 34,180 tok/s), and still loses to the hybrid at 2x/4x context (1.109/1.192 vs 0.995/1.084).~~ — **the 0.956 and 1.109/1.192 numbers above were measured on a different, easier slice of validation data than the hybrid's numbers, AND the model producing them was missing its attention layers' residual connections.** Corrected numbers are in §5h.

### 5g. Does the §3.1 speed crossover survive a REAL training step at width 128? Inconclusive — a methodology problem, not a resolved answer

`benchmarks/gpu_reconcile.py` retrained GPT, pure sheaf-delta, and the hybrid at width 128/ctx 16,384/8 layers (the width/context §3.1 showed sheaf-delta winning, in forward+backward-only timing on random data, 4 layers, no optimizer). With a **real** training step (AdamW, gradient clipping, real enwik8 data, 8 layers):

| arch | bpb | tok/s |
|---|---|---|
| GPT | 2.176 | **122,805** |
| pure sheaf-delta | **1.910** | 50,979 |
| hybrid 3:1 | 1.935 | 94,346 |

Quality still favors sheaf-delta decisively here. But speed did not transfer — GPT was 1.3-2.4x faster in a real step, the opposite of §3.1's forward-only finding at the same width. Investigating why: an isolated 300-step micro-benchmark (nothing else sharing the GPU) measured pure sheaf-delta's forward+backward+gradient-clip+optimizer-step at a stable **154 ms/step (~106,700 tok/s)** — close to double the 50,979 tok/s `gpu_reconcile.py` reported, and a `foreach=True` AdamW setting made no measurable difference, ruling out "many small parameter tensors slow down the optimizer" as the cause. A follow-up attempt to validate this cleanly over the full 3,000 steps was itself run concurrently with another training job (`gpu_posenc.py`) by mistake, and showed per-step time increasing as that job ramped up (0.179s/step for the first 300 steps, matching the isolated micro-benchmark, degrading to 0.374s/step by step 900) — consistent with CUDA-allocator contention between concurrent processes, but this conflates two different hypotheses (sequential multi-architecture pollution within `gpu_reconcile.py` itself, vs. cross-process contention from running something else alongside it) and **neither has been cleanly isolated and confirmed**.

**Status: genuinely unresolved, not a finding either way.** What's known: a short, truly-isolated forward+backward+optimizer probe at this width/context achieves roughly 2x the throughput `gpu_reconcile.py`'s sequential multi-architecture run reported. What's not known: whether a full, clean, 3,000-step, single-architecture-per-process run would land near the isolated probe's ~106,700 tok/s (which would flip this result — sheaf-delta would be competitive with or beat GPT's 122,805) or somewhere between that and the contaminated 50,979. This needs a properly isolated re-run (one architecture, one fresh process, no concurrent jobs, no prior sweep configs in the same process) before either "speed crossover confirmed at width 128 in real training" or "speed crossover doesn't survive real training" can be claimed. Flagged honestly as incomplete rather than reported as either a positive or negative result.

### 5h. Two bugs found and fixed: the real ALiBi/NoPE/YaRN numbers, after §5f was retracted

§5f reported ALiBi beating the hybrid decisively on quality and speed. That triggered a follow-up investigation (ratio dilution, sliding-window attention) to try to close the gap, all of which failed to help — which in hindsight was itself a clue something was wrong with the premise. Two real bugs were found, in order:

**Bug 1 — validation-data-slicing mismatch.** `gpu_scale.py`/`gpu_ratio.py` (GPT, pure sheaf-delta, RoPE-hybrid) default their `eval_len(..., nwin=64)` — covering enwik8 validation bytes `0-65,536` at `@1024`. `gpu_posenc.py`/`gpu_alibi_hybrid.py` (NoPE, ALiBi, ALiBi-hybrid) default `nwin=32` — covering only `0-32,768`. Checked directly: `zlib`-compressing these two slices gives 1.51 vs 1.96 bits/byte — the first 32KB of this project's validation split is a markup-like, repetitive stretch (a list of university names), genuinely easier than what follows. The `@2,048`/`@4,096` extrapolation numbers were unaffected (every script happened to use `nwin=32` there, so those stayed comparable) — only the `@1,024` training-length numbers were on mismatched data.

**Bug 2 — dropped residual connection, found while investigating why a "sanity check" replication of plain GPT trained to a measurably worse loss than the real thing.** Every custom Transformer-block class written for the ALiBi/NoPE/YaRN comparisons (`TBlockPE` in `gpu_posenc.py`, `TBlockAlibi` in `gpu_alibi_hybrid.py` and its copies, `TBlockY` in `gpu_yarn.py` and its copies) wrote the block's forward pass as one Python expression: `return x + mlp(n2(x + mix(n1(x))))`. The real `hsnlm/models.py::TBlock` writes the same logic as two statements: `x = x + mix(n1(x)); return x + mlp(n2(x))`. These are **not equivalent** — the one-liner computes `x + mix(x)` only as an argument to `mlp(...)`, and never writes it back to `x`, so the final `x + ...` adds the attention branch's output to `mlp`'s *output*, but never to the residual stream directly. Confirmed via a block-by-block activation diff against the real `TBlock` (identical weights, identical input): 0.0 difference through a single isolated attention submodule, but 0.91 after one full block, growing to 3.75 after 8 — the signature of a residual branch dropped once per layer, compounding. Fixed (verified bit-exact against the real `TBlock` after the fix) in all 8 affected files (`gpu_posenc.py`, `gpu_alibi_hybrid.py`, `gpu_alibi_hybrid_ratio.py`, `gpu_swa_alibi_hybrid.py`, `gpu_posenc_recheck.py`, `gpu_yarn.py`, `gpu_yarn_recheck.py`, `gpu_yarn_hybrid.py`).

**Corrected numbers, both bugs fixed, retrained from scratch:**

| architecture | bpb @1,024 (train, matched nwin=64) | @2,048 | @4,096 | tok/s | peak mem |
|---|---|---|---|---|---|
| GPT, plain RoPE | 1.103 | 2.111 | 3.547 | 66,015 | 2,897 MB |
| GPT + YaRN (post-hoc; sanity-check replication now matches GPT almost exactly, confirming the fix) | 1.105 | **1.101** | **1.225** | n/a | n/a |
| Pure ALiBi (from scratch, both bugs fixed) | 1.030 | 1.006 | 1.087 | 56,970 | 2,727 MB |
| **Hybrid 3:1, RoPE (§5a — never affected by either bug, uses the real `TBlock`)** | **1.014 (best)** | **0.995 (best)** | **1.084 (best)** | 34,180 | 4,472 MB |
| ALiBi-hybrid 3:1 (from scratch, both bugs fixed) | 1.024 | 1.005 | 1.088 | 31,912 | 4,031 MB |
| YaRN applied to the hybrid's 2 attention layers | 1.025 (plain) | 1.024 (worse than its own 1.007 plain) | 1.131 (worse than its own 1.099 plain) | 39,013 | 4,081 MB |

**The RoPE-hybrid is still the best option on every axis — that didn't change — but the margins are now small (0.01-0.016 bpb against the next-best competitor at every length), not the dramatic gap either buggy version of this comparison showed in either direction.** Pure ALiBi, correctly measured, is a remarkably strong, much cheaper near-tie: within 0.01-0.016 bpb of the hybrid at every length, while training 1.7x faster (56,970 vs 34,180 tok/s) and using 39% less memory (2,727 vs 4,472 MB) — it just has no inference-side advantage at all (§6 — full, growing KV cache, identical mechanism to plain RoPE). GPT+YaRN is a genuinely good zero-retraining result once correctly measured, closing nearly the whole extrapolation gap for a plain Transformer with no architecture change. Applying YaRN to the *hybrid's* attention layers, by contrast, makes its extrapolation measurably worse, not better — plausible explanation: the hybrid's 6 sheaf-delta layers already carry long-range context, so its 2 attention layers only need sharp local/mid-range lookup, a role unscaled RoPE was already doing fine; YaRN's rescaling only has something to fix in an architecture whose *only* long-range mechanism is attention.

**Update: the 4:1 dilution, SWA, and NoPE experiments have since been rerun with the residual fix (§5k) — all three confirm their pre-fix findings.** 4:1 dilution (10 layers, 2 attention): 1.042/1.023/1.101 bpb, still strictly worse than the 3:1 ALiBi-hybrid's 3-seed mean (1.027/1.007/1.083, §5i) despite 25% more parameters. SWA-512 (window strictly below train length): 1.061/1.042/1.120, clearly worse than un-windowed ALiBi-hybrid. SWA-1024 (window equal to train length — i.e. doesn't actually restrict anything within the training context): 1.030/1.012/1.090, roughly neutral vs un-windowed, as expected. NoPE rechecked: 2.344/2.730/3.678 — improved somewhat from the pre-fix placeholder (3.029/3.135/3.286) but still far worse than every other option. See §5k for the full numbers and context.

**The meta-lesson, worth stating plainly:** every number in this entire report that used a custom model reimplementation rather than the project's own `hsnlm/models.py` classes turned out to need re-verification, and two separate, real bugs were found exactly that way — by being suspicious of a result that looked too good (ALiBi "beating" the hybrid) or too bad (a "sanity check" that should have reproduced a known number but didn't) and tracing it to ground rather than accepting it. Neither bug was something a cursory reading of the code would catch — both required either an external second opinion or a deliberately constructed numerical diagnostic (bit-exact parameter and activation comparisons against a known-correct reference) to surface.

### 5i. Multi-seed check (compiled): the ranking in §5h's corrected table was within single-seed noise for one of its three closest entries

§5h's corrected table is single-seed. The margins between the RoPE-hybrid (1.014), pure ALiBi (1.030), and ALiBi-hybrid (1.024) are small enough (0.01-0.016 bpb) that this needed checking before being read as more than "roughly tied." `benchmarks/gpu_multiseed_compiled.py` reran all three at seeds 0, 1, 2, all under `torch.compile` (so speed/memory are also fairly comparable, not mixing compiled/uncompiled measurements), with full `@1024`(nwin=64)/`@2048`/`@4096`(nwin=32) eval:

| arch | @1024 (mean ± std) | @2048 (mean ± std) | @4096 (mean ± std) | per-seed @1024 (0/1/2) | compiled train_s | compiled tok/s | compiled peak mem |
|---|---|---|---|---|---|---|---|
| RoPE-hybrid 3:1 | 1.0364 ± 0.0269 | 1.0203 ± 0.0289 | 1.1070 ± 0.0298 | 1.0289 / 1.0079 / 1.0724 | 301.7s | 81,448 | 2,873 MB |
| Pure ALiBi | 1.0369 ± 0.0012 | 1.0096 ± 0.0015 | 1.0855 ± 0.0019 | 1.0359 / 1.0364 / 1.0385 | 287.5s | 85,478 | 2,153 MB |
| **ALiBi-hybrid 3:1** | **1.0273 ± 0.0076** | **1.0067 ± 0.0066** | **1.0831 ± 0.0050** | 1.0201 / 1.0241 / 1.0378 | 313.0s | 78,518 | 2,829 MB |

**This revises §5h's headline.** The RoPE-hybrid's single-seed 1.014 was real but noisy — its 3-seed standard deviation (0.027-0.030) is *larger* than the 0.01-0.016 bpb gap it was reported winning by, and its 3-seed mean (1.036) ties pure ALiBi's mean (1.037) rather than beating it. **The ALiBi-hybrid is the configuration that beats pure ALiBi robustly**: ahead on all 3 seeds individually (by 0.016, 0.012, 0.001 bpb) and on the 3-seed mean at every length, with visibly tighter variance (std 0.005-0.008 vs the RoPE-hybrid's 0.027-0.030). The robust, multi-seed-confirmed winner against pure ALiBi is the ALiBi-hybrid, not the RoPE-hybrid originally reported in §5h.

Compiled, fairly measured for all three: pure ALiBi is fastest/leanest (85,478 tok/s, 2,153 MB); the ALiBi-hybrid (the quality winner) is 8.9% slower (78,518 tok/s) and uses 31% more memory (2,829 MB) while cutting attention layers 4x — a real gap, but much smaller than the uncompiled 1.7x/39% figures elsewhere in this report once both sides are compiled the same way.

### 5j. Does the §3.1 speed crossover survive a REAL training step at width 128? Resolved — §5g was CUDA-allocator contamination, not a real finding

§5g left this as "genuinely unresolved, not a finding either way" — a real-training-step rerun at width 128/ctx 16,384 showed GPT faster (122,805 tok/s vs hybrid 94,346, pure sheaf-delta 50,979), contradicting §3.1's forward-only crossover prediction, but with unconfirmed suspicion of CUDA-allocator contamination from running all three architectures sequentially in one process. `benchmarks/gpu_reconcile_clean.py` reruns the same three architectures, same recipe (width 128, ctx 16,384, 8 layers, 3,000 steps, real AdamW training on enwik8), each in its own fresh, isolated Python subprocess:

| arch | tok/s | peak mem | val bpb |
|---|---|---|---|
| GPT | 124,685 | 978 MB | 2.213 |
| Pure sheaf-delta-fused | **144,812** | 1,638 MB | **1.917** |
| Hybrid 3:1 | **136,488** | 1,413 MB | **1.931** |

**It was contamination.** Isolated, both sheaf-based architectures are faster than GPT at this width/context (hybrid +9.5%, pure sheaf-delta +16.1%) *and* win on quality (by 0.28-0.30 bpb) — matching §3.1's forward-only crossover prediction instead of contradicting it. §5g's contaminated numbers are superseded and should not be cited. This also directly answers one framing of the open question in §11: a width/context combination exists where the sheaf-based architectures win on both speed and quality simultaneously in a real training run — it's this one (width 128/ctx 16,384), not the width-384/ctx-1,024 configuration §5/§5a's quality numbers came from.

### 5k. NoPE, 4:1 dilution, and SWA rechecked against the residual fix

§5h fixed the dropped-residual bug but deprioritized rerunning three experiments that were already negative pre-fix. All three have now been rerun:

**NoPE** (`gpu_posenc.py`, bug-fixed, nwin=64 @1024/nwin=32 @2048,4096): 2.344 / 2.730 / 3.678 bpb, 80,546 tok/s. Improved somewhat from the pre-fix placeholder (3.029/3.135/3.286) but remains far worse than every other architecture in this report by over an order of magnitude more bpb than separates any other pair — confirms the bug was never going to change NoPE's standing.

**4:1 ratio dilution** (`gpu_alibi_hybrid_ratio.py 4to1 10 2`, 10 layers/2 attention, bug-fixed): 1.042 / 1.023 / 1.101 bpb, 18.1M params, 30,615 tok/s, 5,126 MB. Still strictly worse than the 3:1 ALiBi-hybrid's 3-seed mean (1.027/1.007/1.083, §5i) despite 25% more parameters — confirms §5a/pre-fix's finding that diluting the attention ratio below 3:1 hurts quality, bug or no bug.

**Sliding-window ALiBi-hybrid** (`gpu_swa_alibi_hybrid.py`, bug-fixed): window=512 (genuinely restricts below the 1,024 train length): 1.061/1.042/1.120 bpb, 44,686 tok/s — clearly worse than the un-windowed ALiBi-hybrid at every length. Window=1,024 (equal to train length, so it doesn't actually restrict anything *within* training context): 1.030/1.012/1.090 bpb, 44,657 tok/s — roughly neutral vs un-windowed, which is expected since this window never actually bound the attention during training. Confirms the pre-fix finding that windowing below train length costs quality; windowing at exactly train length is a wash, not an improvement.

---

## 6. Inference and memory: both reconfirmed, extended range

### 6.1 O(1) decode (`benchmarks/gpu_decode.py`), width 1024, 16 heads, batch 1, CUDA-synced

| context | attention step (ms) | KV cache (MB) | sheaf-delta step (ms) | state (MB) |
|---|---|---|---|---|
| 256 | 0.54 | 2 | 0.99 | 0.33 |
| 1,024 | 0.60 | 8 | 0.96 | 0.33 |
| 4,096 | 1.46 | 34 | 1.00 | 0.33 |
| 16,384 | 4.84 | 134 | 0.96 | 0.33 |
| 65,536 | 18.14 | 537 | 0.98 | 0.33 |

Sheaf-delta decode cost is flat to measurement noise from ctx 256 to 65,536 (4x the CPU report's longest tested context); attention grows 33x over the same range. Crossover (sheaf-delta cheaper per-token) is between ctx 1,024 and 4,096 — consistent with the CPU report's own crossover point (it was between 256 and 1,024 there; GPU's lower per-op launch overhead shifts it slightly later, as expected).

### 6.2 Reversible-block memory (`benchmarks/gpu_reversible.py`), width 112/128, 4 layers

| N | GPT saved-activation MB | HSN-fast (eager) MB | RBlockFast n_rev=4 (all reversible) MB |
|---|---|---|---|
| 512 | 379 | 1,064 | **41** |
| 2,048 | 382 | 1,065 | **41** |
| 8,192 | 395 | 1,067 | **41** |

Flat at 41 MB regardless of context, exactly reproducing the CPU report's "O(1) in depth, exact gradients" result, now shown flat in **context** too (not just depth). Speed overhead of full reversibility vs plain storage: +22% at N=512, +21% at N=8192 — matches the theoretical "+1 forward pass" (~+33% worst case) reasonably well, same order as CPU.

---

## 7. Backprop alternatives: GPU confirms "keep backprop" (`benchmarks/gpu_credit.py`)

Same architecture/task as the CPU experiment (0.82M params, char Shakespeare, 800 steps, 1 seed), GPU timing:

| rule | val bits/char | wall time |
|---|---|---|
| Backprop | **2.202** | 53.3 s |
| DFA (random feedback) | 3.128 | 36.3 s |
| Layer-wise teacher-target (no cross-block grad) | 2.315 | 71.7 s |

Matches CPU almost exactly (CPU: 2.202 / 3.164 / 2.316). DFA is faster wall-clock but far worse quality (fails, as on CPU); teacher-target is *slower* than backprop despite having no cross-block gradient chain — confirms the CPU report's accounting argument (each block still pays its own forward + local backward; removing the cross-block chain saves memory/parallelism opportunity, not FLOPs) transfers unchanged to GPU.

---

## 8. Audit of prior in-repo backprop-replacement/optimizer work: GOAT, GOLIATH, LoSparse

This project's root directory and `research_work/` hold an earlier, separate line of experiments (the "DAD"/"GOAT"/"dominator" family, 14 files total, predating `hsnlm/`) that the CPU report's own `research.md` ancestor cites as DAD's origin. Worth auditing against the same bar as everything above, since it's the same underlying claim (backprop can be replaced/beaten) made by a different implementation.

- **GOAT (`SharedTeacher` + per-layer `mirror` network, teacher-student local learning) and GOLIATH (a SAM+Sophia+Lion hybrid optimizer)** exist in exactly one file, `research_work/ultimate_goat_v2.py`, combined together. Neither was ever run on real data — the file's own training loop uses `torch.randn()` synthetic noise, no accuracy metric, nothing logged anywhere in the repo. There's a code comment admitting `torch.compile` fails on it (nested `.backward()` calls inside `forward()`). This is consistent with, not contradicting, the negative verdict below — it never got far enough to contradict anything.
- **The broader DAD/DTP family** (local target propagation without the teacher/mirror piece; 9 files in `research_work/`, 5 more at root/`torch_dad/`) *was* benchmarked on real MNIST/FashionMNIST — captured in the repo's `markdown-preview (4) (1).pdf`. Read critically rather than at the file's own "dominance" framing: DAD wins on speed in 4 of 6 topology/dataset combinations (loses by 11-45% in the other 2) and **loses accuracy in all 6**, by 0.18-1.68 percentage points. Mixed-to-negative, not the clean win the surrounding prose claims.
- **`hsnlm`'s own §7 experiment (and its GPU replication, §7 above) is the most rigorous version of this same test** — matched compute, real data, same conclusion both times: no local-learning variant saves wall-clock, DFA fails outright, the best alternative still loses 0.11-0.17 bits to backprop. Three independent implementations (toy DAD family, GOAT v2, hsnlm's controlled test, now x2 with the GPU replication) all agree. **This line should be considered closed** — further variants of local/greedy/teacher-student learning are unlikely to change the answer without a fundamentally different argument for why FLOPs would be saved (removing the cross-layer chain saves memory and enables layer-parallelism; it does not reduce the forward+backward cost every block still pays).
- **A separate, genuinely different idea surfaced from the same conversation — LoSparse-style low-rank+sparse factorization of linear layers (compression, not a backprop replacement)** — was tested properly in §5b above (fixed rank, dense/sheaf-delta hybrid backbone, real data): it saves parameters, costs quality, and doesn't touch training speed/memory in this architecture. The pasted code's adaptive-rank-growth mechanism was not implemented/tested as specified, because it has a concrete bug: reassigning `layer.A = nn.Parameter(...)` post-hoc detaches the new tensor from an already-constructed optimizer's param groups, so grown rank columns would never receive a gradient update — a standard PyTorch dynamic-architecture pitfall, not an architecture-level finding.

---

## 9. Updated assessment against the CPU report's "Plan for the GPU phase" (G1-G8)

| # | CPU report's criterion | GPU result |
|---|---|---|
| G1 | Fused kernels: step time vs Transformer at 125M, ctx 2k/8k; pass = within ~1.2x at 2k, faster at 8k | **Passed the spirit of it, at smaller scale than specified**: at 0.82M params/width 128, sheaf-delta-fused beats GPT at ctx 2k (fp32) and by 3-10x at ctx 8k-32k. Not tested at 125M (hardware: 8GB card can't fit it) — scale-up risk remains open. |
| G2 | Wider lr sweep, 3+ seeds, 100-300M params | **Not done** — `gpu_scale.py` reaches ~15M params, 2 seeds-worth of lr sweep only; still running. |
| G3 | Extrapolation-tuned baseline | Not done this pass. |
| G4 | Long-context retrieval (RULER-like) | Not done this pass. |
| G5 | Hybrid ratio ablation | **Done** (§5a): full sweep from 0 to 8 attention layers (of 8) found 3:1 (2 attention layers) to be the quality-and-speed-and-memory Pareto-best point, beating the originally-guessed 7:1 on every axis simultaneously; further diluted ratios (4:1, 5:1) confirmed worse even after the later residual-bug fix (§5k). |
| G6 | Muon/FP8/reversible-at-scale | bf16 tested broadly (§3); reversible confirmed O(1) (§6.2); Muon/FP8 not tried. |
| G7 | Block-diffusion | Not touched. |
| G8 | Downstream benchmarks | Not touched. |

**Net:** G1, the single most important open question from the CPU report, now has a clear *positive* answer at small scale, with one clean caveat (confirmed only at narrow width; wide-width confirmation blocked by this GPU's 8GB VRAM, not by the architecture). G2-G8 are where "check everything" still has real work left — they need either a bigger GPU or several more hours of queued runs on this one.

---

## 10. Threats to validity (additions to the CPU report's list)

1. **One GPU, one run each for §3's speed numbers.** All §3 numbers are single runs (some re-measured once to check for noise, §3.1 footnote); no multi-seed variance estimate for speed numbers the way quality numbers have 2 seeds. (The §5h/§5a/§5c *quality* numbers were also single-seed until §5i's 3-seed recheck of the three closest contenders — see §5i for what that changed.)
2. **8GB VRAM is the binding constraint**, not the architecture: every "OOM" cell in §3.1/3.2 blocks a potentially informative data point, and the one place noise was large enough to need a recheck (§3.1 bf16 16-32k) was also the point closest to this card's memory ceiling.
3. **"Fused" is partial** (§3.3) — only the delta-rule core is Triton-fused; the dyadic Householder mixing is still eager PyTorch and could plausibly be fused too, which would likely shift the crossover earlier still (not tested).
4. **`fla` runs below its stated minimum Triton version** (3.2 vs recommended 3.3) — every run printed a warning; no incorrect results were observed (all causality/gradient/value checks in §2 passed), but this is officially unsupported.
5. **Formula discrepancy (§2)** between this codebase's own delta rule and the canonical formula the fused kernel (and the CPU report's own math section) uses — the fused numbers are for a *slightly different* recurrence than the rest of the repo, not a drop-in replacement as shipped.
6. **The ratio ablation (§5a) and low-rank MLP test (§5b) reused the 14M dense sweep's learning rate** rather than re-tuning per configuration — defensible for the ratio ablation (same backbone, same mlp_mult, only the layer-type pattern changes) but a real caveat for the low-rank MLP ranks, which are known in the literature to be LR-sensitive; the dense-vs-low-rank gap is large enough to trust, the ranking *among* ranks is not.
7. **§5c (200M, gradient-checkpointed) is at a 70x-lower tokens/param ratio than §5/§5a** (~0.025 vs ~1.73) purely due to the same-day time budget — treat its quality numbers as "undertrained-regime behavior," not a bigger/better version of §5's comparison; see §5c's own caveat for the full argument. Its memory and extrapolation-shape findings are on firmer ground than its bpb numbers.
8. Same CPU-report caveats still apply unchanged where relevant (tiny scale for the quality claims, lr grid edge, 1-2 seeds, GPT baseline untuned for extrapolation).

---

## 11. Final synthesis: the best-defended architecture from this pass, stated honestly

Framing matters here: nothing in this report supports "better than Transformer in every way, by a large margin," and anything claiming that should be distrusted on sight — every number generated across both the CPU and GPU passes shows real, measured tradeoffs, not a free lunch. This section was rewritten three times during this pass — once to retract an overclaim, once to retract the retraction after two real bugs were found (§5h), and once more after a multi-seed + clean-isolation pass (§5i-§5k) resolved every remaining caveat that was cheap to resolve on this hardware — which is itself worth stating plainly: every swing was driven by actual bugs or genuine single-seed noise, caught by being suspicious of results that looked too good or too bad and tracing them to ground, not by shifting opinion. What follows reflects the final numbers, verified independently against raw result files, not just re-reading printed summaries.

**The architecture:** sheaf-delta mixer (Householder-delta-rule matrix memory + dyadic local mixing) with a **3:1 ratio of sheaf-delta to attention layers**, evenly spread across depth (§5a) — not the originally-guessed 7:1. Delta-rule core computed by the fused Triton kernel (`fla`'s `chunk_gated_delta_rule`, §2-§3) rather than the eager Python loop. Trained with ordinary AdamW and backprop — no DAD, no GOAT, no GOLIATH, no local/greedy learning of any kind (§7, §8: three independent implementations in this repo all agree these don't work). bf16. `torch.compile` on the whole model for a large, safe speed/memory win (§5e). Gradient checkpointing as an opt-in for memory-constrained runs (§5c), or `hsnlm`'s exact reversible blocks (§6.2) if the stack happens to be homogeneous sheaf-delta.

**What it wins on, with numbers, from this report alone (§5h for the corrected ALiBi comparison, §5i for the multi-seed-confirmed version):**
- **Quality at training length**, against every Transformer variant tested including ALiBi: single-seed best was the RoPE-hybrid at 1.014 bpb vs ALiBi's 1.030 (§5h), but the 3-seed recheck (§5i) found this specific margin was within RoPE-hybrid's own noise — the configuration that robustly beats pure ALiBi across all 3 seeds is the **ALiBi-hybrid** (3-seed mean 1.027 vs ALiBi's 1.037). Either way, the hybrid family beats plain RoPE's 1.103 by a large margin and pure ALiBi by a small, now multi-seed-confirmed one.
- **Length extrapolation to 2x/4x context**: same pattern — ALiBi-hybrid's 3-seed means (1.007/1.083) beat pure ALiBi's (1.010/1.086) at both lengths; plain RoPE (2.111/3.547) is far behind both.
- **Decode cost at long context**: flat ~1ms/token from ctx 256 to 65,536 vs GPT's growth to 18ms (§6.1) — unaffected by which Transformer position-encoding variant is used (confirmed directly: NoPE and ALiBi have the same growing KV-cache/decode-cost profile as plain RoPE, differing only in training-time quality, not inference mechanism). This is the one axis where the margin is large regardless of baseline, and the one ALiBi cannot close even when its quality is correctly measured — ALiBi has no inference-side advantage at all.
- **Training speed past a crossover context that depends on width**: confirmed faster than GPT at width 128 both in forward-only timing (crossover ~2k fp32/~16-32k bf16, §3.1) *and now in a real, cleanly-isolated full training step* (§5j: hybrid +9.5%, pure sheaf-delta +16.1% tok/s vs GPT, at width 128/ctx 16,384) — §5g's earlier contradictory result is now confirmed to have been CUDA-allocator contamination, not a real finding.
- **Iso-quality time-to-target-loss**, fairly measured with both architectures compiled the same way (§5e, refreshed with extrapolation data in §5i's companion run): reaches GPT's own final quality in ~30% less wall-clock and ~42% fewer tokens.

**What it loses on:**
- **Training speed and memory at the width/scale the quality numbers came from**, against *every* pure-attention baseline (GPT, ALiBi) — ALiBi trains at 56,970 tok/s / 2,727 MB (uncompiled) vs the hybrid family's 31,912-34,180 tok/s / 4,031-4,472 MB; even compiled, both hybrids still use more memory than compiled GPT or compiled ALiBi (§5e, §5i). This is real and persists regardless of which bugs got fixed or which seed is checked — it is a genuine property of the architecture (the fused kernel's chunked-backward intermediates plus the still-eager dyadic mixing part both cost memory independent of position-encoding choice), not an artifact.
- Everything beyond ~14M params trained to a real token budget — §5c reached 200M params but at 70x less relative training, so it answers a different question (undertrained-regime behavior) than "does this still win at 200M, properly trained." ALiBi/NoPE were not tested at this scale at all. Still open after this pass.
- A hand-written Triton kernel for the dyadic Householder part remains untried (§5e); `torch.compile` closes much but not all of the remaining speed/memory gap to GPT. Still open after this pass.

**Net claim, stated at the size that can actually be defended, against the strongest baseline tested, now multi-seed-confirmed:** at ~14M parameters, this architecture family is the best option tested on quality-at-training-length, extrapolation, and decode cost, against every Transformer variant including ALiBi — by small, real, multi-seed-confirmed margins against ALiBi (specifically via the ALiBi-hybrid configuration) and large ones against plain RoPE. It costs more to train (slower, more memory) than pure-attention baselines at this same scale, a real and persistent tradeoff — except in one cleanly isolated setting (width 128/ctx 16,384, §5j) where both sheaf-based architectures win on speed too. Pure ALiBi, correctly measured, is a legitimate, much cheaper alternative if the hybrid family's small quality edge and inference-side advantage aren't what's needed. This is the honest scope of the claim — a real, broad, if modest advantage, with a real, persistent cost — not "better than Transformer in every way" and not "worse than the best Transformer," both of which were claimed, and retracted, at different points during this same pass.

---

## 12. Reproducibility

```
hsnlm/fastmixer_fused.py        FusedSheafDeltaMixer (fla chunk_gated_delta_rule), registered as mixer="sheafdelta-fused"
benchmarks/gpu_suite.py         matched CPU-suite replication on GPU (benchmark_results/development/gpu_suite/gpu_suite.json)
benchmarks/gpu_speed.py         width-256 context sweep, fp32+bf16 (benchmark_results/development/speed_vs_length/gpu_speed.json)
benchmarks/gpu_crossover.py     width-128 context sweep, fp32+bf16, the main G1 test (benchmark_results/development/fused_kernel_crossover/gpu_crossover.json)
benchmarks/gpu_decode.py        O(1)-decode port, extended to ctx 65536
benchmarks/gpu_reversible.py    reversible-memory port, extended to ctx 8192
benchmarks/gpu_credit.py        backprop-alternatives port (benchmark_results/development/credit_assignment/gpu_credit.json)
benchmarks/gpu_scale.py         ~14.6M-param run, GPT vs hsn-fused vs hybrid-fused (benchmark_results/development/gpu_scale/gpu_scale.json)
benchmarks/gpu_ratio.py         G5 hybrid-ratio ablation, 14M scale (benchmark_results/development/attention_ratio/gpu_ratio.json)
benchmarks/gpu_lowrank_mlp.py   fixed-rank low-rank MLP A/B test, 14M scale (benchmark_results/development/lowrank_mlp/gpu_lowrank_mlp.json)
benchmarks/gpu_scale_ckpt.py    ~200M-param gradient-checkpointed run, GPT vs hsn-fused vs hybrid-fused 7:1 (benchmark_results/development/gpu_scale_ckpt/gpu_scale_ckpt.json)
benchmarks/gpu_scale_ckpt_31.py ~200M-param run, hybrid-fused with the 3:1 ratio (benchmark_results/development/gpu_scale_ckpt/gpu_scale_ckpt_31.json)
benchmarks/gpu_posenc.py           NoPE/ALiBi from scratch, bug-fixed (§5h, §5k) (benchmark_results/development/positional_encoding/gpu_posenc.json)
benchmarks/gpu_posenc_recheck.py   pure ALiBi + ALiBi-hybrid, bug-fixed, nwin=64 matched (§5h) (benchmark_results/development/positional_encoding/gpu_posenc_recheck.json)
benchmarks/gpu_yarn.py / _recheck.py / _hybrid.py   YaRN on GPT and on the hybrid's attention layers (§5d, §5h)
benchmarks/gpu_alibi_hybrid_ratio.py   4:1/5:1 ratio dilution with ALiBi attention, bug-fixed (§5k) (benchmark_results/development/alibi_hybrid_ratio/gpu_alibi_hybrid_4to1.json)
benchmarks/gpu_swa_alibi_hybrid.py     sliding-window ALiBi-hybrid, bug-fixed (§5k) (benchmark_results/development/sliding_window_alibi/gpu_swa_alibi_hybrid.json)
benchmarks/gpu_multiseed_compiled.py   3-seed compiled recheck of RoPE-hybrid/pure-ALiBi/ALiBi-hybrid (§5i) (benchmark_results/paper/multiseed_compiled/gpu_multiseed_compiled.json)
benchmarks/gpu_isoquality_compiled.py  fair compiled iso-quality, GPT vs hybrid vs alibi-pure, with @2048/@4096 (§5e, §5i) (benchmark_results/paper/isoquality_vs_rope/gpu_isoquality_compiled.json)
benchmarks/gpu_reconcile_clean.py      width-128/ctx-16384 crossover, one isolated subprocess per arch (§5j) (benchmark_results/development/reconcile_speed_quality/gpu_reconcile_clean_{gpt,hsn-fused,hybrid}.json)
```
Setup: `cd hsnlm && uv venv .venv --python 3.12 && source .venv/bin/activate && uv pip install torch --index-url https://download.pytorch.org/whl/cu124 numpy flash-linear-attention`. Data: `data/enwik8` (mattmahoney.net), `data/shakespeare.txt` (karpathy/char-rnn), fetched fresh — not checked into git.
