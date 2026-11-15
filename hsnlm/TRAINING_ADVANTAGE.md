# Training Advantage — Hybrid Sheaf-Delta LM vs matched Transformer

Companion to `ARCHITECTURE.md`. All numbers below are measured (CPU: `findings_report.md`; GPU: `findings_report_gpu.md`), not projected. "Hybrid" below always means the 3:1 sheaf-delta:attention configuration unless noted.

> **In plain English:** this document asks "does our hybrid design actually produce *better writing* than an ordinary Transformer, trained the same way, same size?" Short answer: yes, by a small real margin, against every ordinary-Transformer variant tested, including the best one found (ALiBi). The hybrid also needs noticeably fewer practice examples to reach a given quality level, and — once a speed trick called `torch.compile` is applied fairly to both — actually reaches that quality in *less total training time* too, despite each individual training step costing a bit more. The real downside: training the hybrid uses more GPU memory and runs somewhat slower per step than training an ordinary Transformer of the same size. "bpb" below means "bits per byte" — think of it as a golf score for how well the model predicts text: lower is better. See `ARCHITECTURE.md` §0 for a full glossary.

## 1. Quality

**Methodology note, read before the table (two real bugs were found and fixed in the same session that produced it):**
1. **A data-slicing mismatch**: `gpu_scale.py`/`gpu_ratio.py` (GPT, pure sheaf-delta, RoPE-hybrid) evaluated `@1024` over validation bytes `0-65,536`; `gpu_posenc.py`/`gpu_alibi_hybrid.py` (NoPE, ALiBi, ALiBi-hybrid) defaulted to `0-32,768` — a measurably easier slice of enwik8 (zlib 1.51 vs 1.96 bits/byte; the hybrid's two families agreed on `@2048`/`@4096`, which were never affected). This alone made ALiBi look far better than it was.
2. **A dropped-residual bug**: every custom Transformer-block reimplementation written for the ALiBi/NoPE/YaRN comparisons (`gpu_posenc.py`, `gpu_alibi_hybrid.py`, `gpu_yarn.py`, and copies) collapsed two statements into one — `return x + mlp(n2(x + mix(x)))` instead of `x = x + mix(x); return x + mlp(n2(x))` — which silently drops the attention branch's own residual connection. Verified bit-exact against the real `TBlock` class once fixed; before the fix, every affected model was training with roughly a third of its intended architecture missing, confirmed by a block-by-block activation trace (diff grows from 0.9 after layer 1 to 3.8 after layer 8).

Both bugs were found, fixed, and every affected number below was **retrained from scratch twice** (once per bug) — this table is the result after both fixes, not a patch of the original numbers.

| Scale | Metric | GPT (RoPE) | GPT + YaRN (post-hoc) | NoPE (scratch, bug-fixed, rechecked) | ALiBi (scratch) | **Hybrid 3:1 (RoPE)** | ALiBi-hybrid 3:1 | Pure sheaf-delta |
|---|---|---|---|---|---|---|---|---|
| 0.82M params, CPU, enwik8 | val bpb @ train length | 1.962 | not tested | not tested | not tested | 1.890 (7:1, not 3:1) | not tested | **1.880** |
| 14.3-14.6M params, GPU, enwik8 | val bpb @ train length | 1.103 | 1.105 (sanity-check replication) | 2.344 | 1.030 | **1.014 (best single seed; ties on 3-seed mean, see §1a)** | 1.024 (best on 3-seed mean, see §1a) | 1.199 |
| 14.3-14.6M params, GPU | val bpb @ 2x train length | 2.111 | 1.101 | 2.730 | 1.006 | **0.995 (best single seed)** | 1.005 | 1.187 |
| 14.3-14.6M params, GPU | val bpb @ 4x train length | 3.547 | 1.225 | 3.678 | 1.087 | **1.084 (best single seed)** | 1.088 | 1.248 |
| 14.3-14.6M params, GPU | train tok/s (uncompiled) | 66,015 | n/a | 80,546 | 56,970 | 34,180 | 31,912 | 30,631 |
| 14.3-14.6M params, GPU | peak mem (uncompiled) | 2,897 MB | n/a | n/a | 2,727 MB | 4,472 MB | 4,031 MB | 4,947 MB |

**Single-seed, the RoPE-hybrid is the best option on every axis in this table** — that conclusion survived both bug fixes — but §1a below (a 3-seed, compiled recheck) found that this specific ranking between the RoPE-hybrid, pure ALiBi, and ALiBi-hybrid does not survive multiple seeds: the RoPE-hybrid's margin here is within its own seed-to-seed noise, and the **ALiBi-hybrid**, not the RoPE-hybrid, is the configuration that robustly beats pure ALiBi. Read this table as "single-seed, directionally consistent with but superseded in exact ranking by §1a," not as the final word on hybrid-vs-ALiBi-vs-ALiBi-hybrid. ALiBi has no inference-side advantage at all regardless (full, growing KV cache, identical mechanism to plain RoPE — see `INFERENCE_ADVANTAGE.md`), so the hybrid family's real value proposition stands either way: *a small, consistent quality/extrapolation edge plus a genuine decode-cost advantage* over pure ALiBi, not a dominant quality win.

**NoPE has now been rechecked with the dropped-residual fix (bug-fixed number above, 2.344/2.730/3.678).** It improved somewhat from its pre-fix placeholder (was 3.029/3.135/3.286) but remains far worse than every other option by over an order of magnitude more bpb than separates any other pair in this table — confirms the prediction that the bug was very unlikely to change NoPE's standing.

**GPT + YaRN is a genuinely strong, zero-retraining result, now correctly measured:** applying YaRN (NTK-by-parts RoPE frequency interpolation, applied only at eval time to an already-trained plain-RoPE model, no retraining) closes nearly the entire extrapolation gap for a plain Transformer — 1.101 vs the hybrid's 0.995 at 2x context, far closer than the pre-fix numbers (1.370) suggested. Interestingly, applying the *same* YaRN technique to the hybrid's own 2 attention layers makes its extrapolation measurably *worse* (1.024 vs the hybrid's own plain-RoPE 1.007 at 2x) — plausible explanation: the hybrid's 6 sheaf-delta layers already carry long-range context natively, so its 2 attention layers only need to do sharp local/mid-range lookup, a role unscaled RoPE is already well-suited for; YaRN's frequency rescaling helps a pure-attention model (which has no other mechanism for long range) but only distorts a role that wasn't broken in the hybrid. This is an explanation for an observed result, not an independently proven mechanism.

### 1a. Multi-seed check (compiled): does the hybrid/ALiBi/ALiBi-hybrid ranking survive more than one seed?

The quality numbers in §1's table are a single seed each — flagged as the last open caveat (open question #7, §5). This reruns the three closest contenders — RoPE-hybrid 3:1, pure ALiBi, ALiBi-hybrid 3:1 — at seeds 0, 1, 2, all under `torch.compile` (so the numbers are also directly, fairly comparable on speed/memory, not mixing compiled and uncompiled measurements):

| arch | @1024 (mean ± std) | @2048 (mean ± std) | @4096 (mean ± std) | per-seed @1024 (0/1/2) | compiled train_s | compiled tok/s | compiled peak mem |
|---|---|---|---|---|---|---|---|
| RoPE-hybrid 3:1 | 1.0364 ± 0.0269 | 1.0203 ± 0.0289 | 1.1070 ± 0.0298 | 1.0289 / 1.0079 / 1.0724 | 301.7s | 81,448 | 2,873 MB |
| Pure ALiBi | 1.0369 ± 0.0012 | 1.0096 ± 0.0015 | 1.0855 ± 0.0019 | 1.0359 / 1.0364 / 1.0385 | 287.5s | 85,478 | 2,153 MB |
| **ALiBi-hybrid 3:1** | **1.0273 ± 0.0076** | **1.0067 ± 0.0066** | **1.0831 ± 0.0050** | 1.0201 / 1.0241 / 1.0378 | 313.0s | 78,518 | 2,829 MB |

**What this settles:** the RoPE-hybrid's single-seed "best" result (1.014, §1) was real but noisy — its seed-to-seed standard deviation (0.027) is *larger* than the 0.01-0.016 bpb gap it was reported winning by, and its 3-seed mean (1.036) actually ties pure ALiBi's mean (1.037) rather than beating it. **The ALiBi-hybrid is the one configuration that wins cleanly and reproducibly**: it beats pure ALiBi on all 3 seeds individually (by 0.016, 0.012, and 0.001 bpb) and on the 3-seed mean at every context length tested (@1024: 1.027 vs 1.037; @2048: 1.007 vs 1.010; @4096: 1.083 vs 1.086), with noticeably tighter variance (std 0.005-0.008 vs RoPE-hybrid's 0.027-0.030). **This revises §1's headline**: the robust, multi-seed-confirmed winner against pure ALiBi is the ALiBi-hybrid, not the RoPE-hybrid — the RoPE-hybrid's apparent edge in the single-seed table was within noise.

**Bonus — the compiled speed/memory gap to pure ALiBi, now fairly measured for all three:** compiled, pure ALiBi trains at 85,478 tok/s / 2,153 MB; the ALiBi-hybrid (the quality winner above) is 8.9% slower (78,518 tok/s) and uses 31% more memory (2,829 MB) while cutting attention layers 4x (2 of 8 vs 8 of 8) — a real but much smaller gap than the uncompiled 1.7x/39% figures elsewhere in this document, now that both sides are compiled the same way.

**Supporting evidence (synthetic tasks, both CPU and GPU, consistent):**
- Associative recall (multi-query key→value): sheaf-delta and hybrid learn faster than the Transformer baseline at matched steps (e.g. 16 pairs: Transformer 0.108-0.109, sheaf-delta 0.170-0.178, hybrid 0.214-0.238) — read as "learns faster here," not "attention can't recall," since all three are undertrained on this task.
- State tracking (parity): the Transformer baseline never learns it (~0.5 = chance) in the step budget tested; sheaf-delta and hybrid solve it (0.96-1.00) and generalize to 4x sequence length. This is the cleanest mechanistic evidence for the delta-rule's negative-eigenvalue transitions (§4 of `findings_report.md`) over diagonal gating.

## 2. Speed — real, but context- and width-dependent, and not yet shown at the quality-winning width

With the fused Triton kernel (`fla`'s `chunk_gated_delta_rule`), sheaf-delta training speed **crosses over and beats the Transformer past a context-length threshold**:

| Width | fp32 crossover | bf16 crossover | Confirmed how far past crossover? |
|---|---|---|---|
| 128 | ~1.5-2k tokens | ~16-32k tokens | Up to 32x faster (fp32, ctx 32,768); ~1.4-2.4x faster (bf16, ctx 16-32k, noisy — see `findings_report_gpu.md` §3.1 footnote) |
| 256 | not reached by ctx 2,048 (OOM beyond) | not reached by ctx 2,048 (OOM beyond) | Trend converging (ratio climbs from 0.34x to 0.41x as context grows) but this 8GB card ran out of memory before crossing |
| 384 (the width used for the §1 quality numbers) | **not tested at the quality-winning ratio** — the 14M comparison used ctx 1,024, where GPT was still faster (372s vs 760-802s) | | |

**The honest gap:** the quality win (§1, width 384, ctx 1,024) and the speed win (width 128, ctx ≥2k) have never been demonstrated in the exact same width/context combination. But §2a below now confirms, in a cleanly isolated real training run at width 128/ctx 16,384, that both the hybrid *and* pure sheaf-delta can simultaneously beat GPT on speed and quality in the same run — resolving the question of whether such a combination exists at all, even though it's not literally the same width=384 config §1's quality numbers came from.

### 2a. Confirming the crossover survives a real full training step, cleanly isolated — resolves open question #4 (§5)

`findings_report_gpu.md` §5g flagged a real problem: a full real-training-step rerun at this same width-128/ctx-16,384 setting showed GPT *faster* than both sheaf-based architectures — the opposite of the forward-only crossover numbers above — but that script ran all three architectures sequentially in one long-lived process, and CUDA-allocator contamination/fragmentation was suspected but never cleanly confirmed as the cause. This reruns the same three architectures (GPT, pure sheaf-delta-fused, hybrid 3:1), each in its own fresh, isolated Python subprocess:

| arch | tok/s | peak mem | val bpb |
|---|---|---|---|
| GPT | 124,685 | 978 MB | 2.213 |
| Pure sheaf-delta-fused | **144,812** | 1,638 MB | **1.917** |
| Hybrid 3:1 | **136,488** | 1,413 MB | **1.931** |

**This resolves §5g: it was contamination, not a real finding.** Isolated, both sheaf-based architectures are faster than GPT at this width/context (hybrid +9.5%, pure sheaf-delta +16.1%) *and* win on quality (by 0.28-0.30 bpb), matching the forward-only crossover prediction instead of contradicting it. The earlier contaminated run's numbers (GPT 122,805 tok/s vs hybrid 94,346, pure sheaf-delta 50,979) are superseded and should not be cited.

**Why eager-mode numbers don't matter here:** without the fused kernel, sheaf-delta is 20-150x slower than the Transformer on GPU (kernel-launch overhead on many small ops is punished harder on GPU than CPU) — worse than the CPU-measured 2-3x gap, not better. The entire speed story depends on the fused kernel existing and being used; see `ARCHITECTURE.md` §4 for what's still unfused.

## 3. Memory — checkpointing is a real, demonstrated lever

At the exact 14M-param/width-384/ctx-1024 configuration used for the §1 quality numbers:

| arch | peak memory, no checkpointing | peak memory, with checkpointing | reduction |
|---|---|---|---|
| Transformer | 2,897 MB | 660 MB | 4.4x |
| Pure sheaf-delta | 4,947 MB | 913 MB | 5.4x |
| Hybrid (3:1) | 4,472 MB | 908 MB | 4.9x |

Gradient checkpointing (recompute each block's forward during backward instead of storing activations) cuts absolute memory 4.4-5.4x for every architecture tested, with the usual ~20-33% compute cost for the recomputation. It **narrows but does not close** sheaf-delta's relative memory disadvantage (913/660 = 1.38x more than the Transformer, down from 4947/2897 = 1.71x without checkpointing) — the fused kernel's own intermediates (the WY-representation chunk tensors) plus the still-eager dyadic mixing part both cost memory independent of whether checkpointing is on. `hsnlm`'s exact reversible blocks (`RevLM`, homogeneous sheaf-delta stacks only) achieve the same O(1)-in-depth memory property with exact (not recomputed-and-discarded) gradients, if the stack doesn't need attention layers mixed in.

**A second, independent lever: `torch.compile`.** Two versions of this were measured. First, compiling just the still-eager dyadic mixer submodule (`findings_report_gpu.md` §5e): pure sheaf-delta 42% faster / 31% less memory; the hybrid 41% faster / 30% less memory. Second, and better — compiling the **whole model** (`model = torch.compile(model)`, both GPT and the hybrid, same recipe): GPT dropped from 372.3s to 250.8s (**-32.6%**), and the hybrid dropped from 719.0s to 304.4s (**-57.7%**, a bigger win than the mixer-only compile found) — real, measured full training runs, not a microbenchmark. This is the cheap, safe version of "fuse the dyadic part too" (`ARCHITECTURE.md` §4); the full hand-written-Triton-kernel version remains untried and would likely do better still, but was judged too risky to rush. The two levers (checkpointing, compile) haven't been combined or measured together.

### 3a. Iso-quality (time-to-target-loss): the hybrid needs fewer tokens *and* less wall-clock time to match GPT, once both are compiled fairly

A natural question the raw tok/s numbers above don't answer directly: if GPT trains faster per step, does it still reach the *same quality* sooner? Measured directly (both architectures `torch.compile`d the same way — compiling only one side would bias this comparison, so both were compiled uniformly for this specific test):

| | GPT (compiled) | Hybrid 3:1 (compiled) | Pure ALiBi (compiled, 3-seed mean, §1a) |
|---|---|---|---|
| Wall-clock for 1,500 steps | 251.4s | 303.9s | 287.5s |
| Final val bpb @1024 | 1.1014 | 1.0305 | 1.0369 |
| @2048 | 2.0244 | 1.0143 | 1.0096 |
| @4096 | 3.3221 | 1.0981 | 1.0855 |
| tok/s | 97,772 | 80,874 | 85,478 |
| Peak memory | 2,408 MB | 2,866 MB | 2,153 MB |

The hybrid needed only **~864 of its own 1,500 steps (42.4% fewer tokens)** to reach GPT's final train loss — and because each hybrid step, compiled, isn't that much slower than GPT's, that token efficiency translates into real wall-clock savings too: **~175.1s to reach GPT's quality, vs GPT's own 251.4s (30.3% less wall-clock)**. This was checked against the actual recorded training curves (50-step resolution, linearly interpolated between the two bracketing points), not inferred from a single end-point. Plausible mechanism, consistent with `findings_report.md` §7.4's discussion of the sheaf-delta memory doing a form of online closed-form updating inside the forward pass — each token teaches this architecture more per step than it teaches a plain Transformer, enough to outweigh its somewhat higher per-step cost. This is a real, fairly-measured result — a dramatically larger "time-to-quality" claim (e.g. built by compiling only one side) would not be. Pure ALiBi's compiled numbers are now in the same table for reference (§1a has the full 3-seed quality comparison) — it's faster and leaner than both GPT-matched architectures here but wasn't run through the same iso-quality crossing-point analysis, since the quality race is between the hybrid family and ALiBi, not between either and GPT.

**Separately, switching from the 7:1 ratio to the measured-best 3:1 ratio is a free compute saving, not a tradeoff:** 3:1 is simultaneously faster (34,180 vs 32,329 tok/s) *and* leaner (4,472 vs 4,738 MB) than 7:1, on top of being higher quality (§1). There was no cost to finding and using the better ratio.

## 4. Backprop alternatives: closed question, confirmed three ways

No tested method reduces training compute relative to plain backprop, and the two most "decoupled" methods actively lose quality:

| Method | Result | Where tested |
|---|---|---|
| DAD-style greedy local loss per block | No wall-clock savings; -0.17 bit vs backprop | CPU, `findings_report.md` §7.1 |
| DFA (random fixed feedback) | Fails outright (-0.9 to -1 bit) | CPU and GPU (`findings_report.md` §7.1; `findings_report_gpu.md` §7) |
| Layer-wise teacher-target regression | **Slower** wall-clock than backprop despite no cross-block gradient chain | CPU and GPU, same numbers both times (2.316/2.315 bpc) |
| GOAT (shared-teacher + per-layer mirror network) + GOLIATH (SAM+Sophia+Lion optimizer) | Never benchmarked on real data by the prior implementation (synthetic smoke test only); consistent with, not contradicting, the above | Audited, not re-run — `findings_report_gpu.md` §8 |

**Why this keeps happening:** backprop computes the full gradient for ~2x the forward cost; every alternative tested still requires each block to do its own forward + local backward, so removing the cross-block dependency chain saves memory and enables layer-parallelism, not FLOPs (`findings_report.md` §7.3). This is an accounting argument, not an empirical coincidence — it is unlikely to reverse with a new variant of the same idea family.

## 5. Open questions (do not treat as settled)

1. **Resolved, after two rounds of bug-fixing (§1):** the hybrid family beats a matched Transformer on quality — including ALiBi, the strongest baseline — but the margin is small (0.01-0.016 bpb), not the dramatic win the original framing claimed nor the loss a since-corrected intermediate version of this document claimed. Two real bugs (a validation-data-slicing mismatch, and a dropped-residual bug in every custom ALiBi/NoPE/YaRN reimplementation) drove the swings; both are now fixed and verified bit-exact against the real `TBlock` class.
2. **Resolved — NoPE rechecked against the dropped-residual fix (§1).** Bug-fixed number: 2.344/2.730/3.678 bpb @1024/2048/4096 — improved somewhat from the pre-fix placeholder (3.029/3.135/3.286) but still far worse than every other option by over an order of magnitude more bpb than separates any other pair in this report. Confirms the bug was never going to change NoPE's standing.
3. **Resolved — the 4:1 ratio-dilution and sliding-window-ALiBi (SWA) experiments both rerun with the residual fix.** 4:1 dilution (10 layers, 2 attention): 1.042/1.023/1.101 bpb — still strictly worse than the 3:1 ALiBi-hybrid's 3-seed mean (1.027/1.007/1.083, §1a) despite 25% more parameters, confirming the pre-fix finding. SWA-512 (window below train length): 1.061/1.042/1.120 — clearly worse than un-windowed ALiBi-hybrid, confirming windowing below train context costs quality. SWA-1024 (window equal to train length, i.e. not really restricting anything within training context): 1.030/1.012/1.090 — roughly neutral vs un-windowed, as expected since the window never actually bound anything at train length.
4. **Resolved — the 3:1 ratio's quality win does coexist with a real speed win in at least one clean setting (§2a).** A cleanly isolated (separate-subprocess) full training run at width 128/ctx 16,384 showed both pure sheaf-delta and the hybrid simultaneously beating GPT on speed (+16.1%/+9.5%) *and* quality (0.28-0.30 bpb) — confirming `findings_report_gpu.md` §5g's earlier contradictory result was CUDA-allocator contamination, not a real finding. Not literally the same width=384 config as §1's quality numbers, but establishes that such a combination is real and exists.
5. **Resolved (the training-duration half) — does the win survive much more training at the same 14M-param scale, or is it an early-training artifact?** This was the single most-repeated standing risk in this document (the CPU report's own threats-to-validity warning: "early-training advantages from locality/recurrence biases often vanish or reverse with more training"). `benchmarks/gpu_chinchilla_15m.py` reran `h6` (the current-best ALiBi-hybrid+`mixer_heads=6` config) against pure ALiBi at 18,000 steps instead of 1,500 (~20.5 "tokens"/param — in practice ~7.4 repeated epochs over the fixed 40M-byte enwik8 training slice, not fresh unique data, stated plainly rather than oversold). Result: `h6` 0.7820/0.7637/0.8276 bpb @1024/2048/4096 vs pure ALiBi's 0.7924/0.7720/0.8307 — **the gap did not shrink.** It held at @1024 (0.0078 at 1,500 steps → 0.0104 at 18,000) and actually widened at @2048 (0.0004 → 0.0083) and @4096 (a noise-level disadvantage → a real +0.0031 advantage). One seed, not three — a multi-seed check at this step count would still be worth doing before treating the exact margin as final — but the central risk (does this evaporate with real training?) is directly addressed and the answer is no.
6. **Still open (the param-count half) — does the win survive past 14M params?** A ~200M-param attempt landed at a 70x-lower tokens/param ratio than the 14M comparison (time-budget artifact, not an architecture result) — see `findings_report_gpu.md` §5c for why those specific numbers should not be read as a scale answer either way. Reaching even a modest tokens/param ratio at 50-86M params would need several hours per architecture on this hardware — deliberately not attempted yet, to avoid reproducing the same undertrained-regime artifact §5c already found. Also: ALiBi/NoPE were never tested at this scale at all.
6. **Would fusing the dyadic Householder part fully close the speed/memory gap?** Still open. `torch.compile` on the whole model (§3, §3a) gives a real, large partial answer (32.6%/57.7% faster for GPT/hybrid respectively) without a hand-written kernel; the full hand-written-Triton version remains untried.
7. **Resolved — multi-seed check (§1a).** Reran the three closest contenders (RoPE-hybrid, pure ALiBi, ALiBi-hybrid) at seeds 0/1/2, compiled. Result: the RoPE-hybrid's single-seed edge was within noise (std 0.027, larger than the gap it was reported winning by); the **ALiBi-hybrid** is the configuration that robustly beats pure ALiBi, winning on all 3 seeds individually and on the 3-seed mean at every context length, with tighter variance (std 0.005-0.008).

**Remaining genuinely open items, after this pass: #5 (scale beyond 14M) and #6 (hand-written Triton kernel for the dyadic part).** Everything else flagged as an open caveat in earlier versions of this document has now been measured and resolved one way or the other.
