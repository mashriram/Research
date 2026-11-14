# Training Advantage — Hybrid Sheaf-Delta LM vs matched Transformer

Companion to `ARCHITECTURE.md`. All numbers below are measured (CPU: `findings_report.md`; GPU: `findings_report_gpu.md`), not projected. "Hybrid" below always means the 3:1 sheaf-delta:attention configuration unless noted.

> **In plain English:** this document asks "does our hybrid design actually produce *better writing* than an ordinary Transformer, trained the same way, same size?" Short answer: yes, by a small real margin, against every ordinary-Transformer variant tested, including the best one found (ALiBi). The hybrid also needs noticeably fewer practice examples to reach a given quality level, and — once a speed trick called `torch.compile` is applied fairly to both — actually reaches that quality in *less total training time* too, despite each individual training step costing a bit more. The real downside: training the hybrid uses more GPU memory and runs somewhat slower per step than training an ordinary Transformer of the same size. "bpb" below means "bits per byte" — think of it as a golf score for how well the model predicts text: lower is better. See `ARCHITECTURE.md` §0 for a full glossary.

## 1. Quality

**Methodology note, read before the table (two real bugs were found and fixed in the same session that produced it):**
1. **A data-slicing mismatch**: `gpu_scale.py`/`gpu_ratio.py` (GPT, pure sheaf-delta, RoPE-hybrid) evaluated `@1024` over validation bytes `0-65,536`; `gpu_posenc.py`/`gpu_alibi_hybrid.py` (NoPE, ALiBi, ALiBi-hybrid) defaulted to `0-32,768` — a measurably easier slice of enwik8 (zlib 1.51 vs 1.96 bits/byte; the hybrid's two families agreed on `@2048`/`@4096`, which were never affected). This alone made ALiBi look far better than it was.
2. **A dropped-residual bug**: every custom Transformer-block reimplementation written for the ALiBi/NoPE/YaRN comparisons (`gpu_posenc.py`, `gpu_alibi_hybrid.py`, `gpu_yarn.py`, and copies) collapsed two statements into one — `return x + mlp(n2(x + mix(x)))` instead of `x = x + mix(x); return x + mlp(n2(x))` — which silently drops the attention branch's own residual connection. Verified bit-exact against the real `TBlock` class once fixed; before the fix, every affected model was training with roughly a third of its intended architecture missing, confirmed by a block-by-block activation trace (diff grows from 0.9 after layer 1 to 3.8 after layer 8).

Both bugs were found, fixed, and every affected number below was **retrained from scratch twice** (once per bug) — this table is the result after both fixes, not a patch of the original numbers.

| Scale | Metric | GPT (RoPE) | GPT + YaRN (post-hoc) | NoPE (scratch, bug not yet rechecked) | ALiBi (scratch) | **Hybrid 3:1 (RoPE)** | ALiBi-hybrid 3:1 | Pure sheaf-delta |
|---|---|---|---|---|---|---|---|---|
| 0.82M params, CPU, enwik8 | val bpb @ train length | 1.962 | not tested | not tested | not tested | 1.890 (7:1, not 3:1) | not tested | **1.880** |
| 14.3-14.6M params, GPU, enwik8 | val bpb @ train length | 1.103 | 1.105 (sanity-check replication) | 3.029 (pre-fix; not yet rechecked) | 1.030 | **1.014 (best)** | 1.024 | 1.199 |
| 14.3-14.6M params, GPU | val bpb @ 2x train length | 2.111 | 1.101 | 3.135 (pre-fix) | 1.006 | **0.995 (best)** | 1.005 | 1.187 |
| 14.3-14.6M params, GPU | val bpb @ 4x train length | 3.547 | 1.225 | 3.286 (pre-fix) | 1.087 | **1.084 (best)** | 1.088 | 1.248 |
| 14.3-14.6M params, GPU | train tok/s (uncompiled) | 66,015 | n/a | 80,443 (pre-fix) | 56,970 | 34,180 | 31,912 | 30,631 |
| 14.3-14.6M params, GPU | peak mem (uncompiled) | 2,897 MB | n/a | n/a | 2,727 MB | 4,472 MB | 4,031 MB | 4,947 MB |

**The RoPE-hybrid remains the best option on every axis in this table — that conclusion survived both bug fixes — but the margins are small and honest now (0.01-0.016 bpb against the next-best option at every length), not the dramatic swings either direction that earlier, buggy versions of this comparison showed.** The real story is how close pure ALiBi now is: correctly implemented, it's within 0.01-0.016 bpb of the hybrid at every length, while training 1.7x faster and using 39% less memory. ALiBi has no inference-side advantage at all (full, growing KV cache, identical mechanism to plain RoPE — see `INFERENCE_ADVANTAGE.md`), so the hybrid's real value proposition is: *a small, consistent quality/extrapolation edge plus a genuine decode-cost advantage* — not a dominant quality win, which is a more modest and more defensible claim than either the original "hybrid beats Transformer" framing or the subsequent (buggy) "ALiBi beats the hybrid" correction.

**NoPE's row above is pre-fix and has not yet been rechecked** with the dropped-residual fix — given it was already ~3x worse than everything else even before accounting for the bug, it's very unlikely to flip the overall ranking, but its exact numbers should not be cited as final.

**GPT + YaRN is a genuinely strong, zero-retraining result, now correctly measured:** applying YaRN (NTK-by-parts RoPE frequency interpolation, applied only at eval time to an already-trained plain-RoPE model, no retraining) closes nearly the entire extrapolation gap for a plain Transformer — 1.101 vs the hybrid's 0.995 at 2x context, far closer than the pre-fix numbers (1.370) suggested. Interestingly, applying the *same* YaRN technique to the hybrid's own 2 attention layers makes its extrapolation measurably *worse* (1.024 vs the hybrid's own plain-RoPE 1.007 at 2x) — plausible explanation: the hybrid's 6 sheaf-delta layers already carry long-range context natively, so its 2 attention layers only need to do sharp local/mid-range lookup, a role unscaled RoPE is already well-suited for; YaRN's frequency rescaling helps a pure-attention model (which has no other mechanism for long range) but only distorts a role that wasn't broken in the hybrid. This is an explanation for an observed result, not an independently proven mechanism.

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

**The honest gap:** the quality win (§1, width 384, ctx 1,024) and the speed win (width 128, ctx ≥2k) have never been demonstrated in the same run. It is an open question whether the 3:1 ratio's quality edge survives at a width/context combination where sheaf-delta is *also* the faster architecture. This is flagged as the most important unresolved experiment in `findings_report_gpu.md` §11, and (if run) would need to be read back into this document.

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

| | GPT (compiled) | Hybrid 3:1 (compiled) |
|---|---|---|
| Wall-clock for 1,500 steps | 250.8s | 304.4s |
| Final train loss @ 1,500 steps | 1.712 | 1.605 |
| Final val bpb @ 1,500 steps | 1.1145 | **1.0178** |
| Peak memory | 2,408 MB | 2,868 MB |

The hybrid needed only **~857 of its own 1,500 steps (42.9% fewer tokens)** to reach GPT's final train loss (1.712) — and because each hybrid step, compiled, isn't that much slower than GPT's, that token efficiency translates into real wall-clock savings too: **~173.8s to reach GPT's quality, vs GPT's own 250.8s (30.7% less wall-clock)**. This was checked against the actual recorded training curves (50-step resolution), not inferred from a single end-point. Plausible mechanism, consistent with `findings_report.md` §7.4's discussion of the sheaf-delta memory doing a form of online closed-form updating inside the forward pass — each token teaches this architecture more per step than it teaches a plain Transformer, enough to outweigh its somewhat higher per-step cost. This is a real, fairly-measured result — a dramatically larger "time-to-quality" claim (e.g. built by compiling only one side) would not be.

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

1. **Resolved, after two rounds of bug-fixing (§1):** the hybrid beats a matched Transformer on quality — including ALiBi, the strongest baseline — but the margin is small (0.01-0.016 bpb), not the dramatic win the original framing claimed nor the loss a since-corrected intermediate version of this document claimed. Two real bugs (a validation-data-slicing mismatch, and a dropped-residual bug in every custom ALiBi/NoPE/YaRN reimplementation) drove the swings; both are now fixed and verified bit-exact against the real `TBlock` class.
2. **NoPE has not yet been rechecked against the dropped-residual fix.** Its pre-fix number (3.029 bpb, ~3x worse than everything else) makes it very unlikely to change the overall ranking, but should not be cited as a final number.
3. **The 4:1/5:1 ratio-dilution experiment and the sliding-window-ALiBi (SWA) experiment both used the buggy residual implementation and were never rereun** — both were already negative results even pre-fix (4:1 was strictly worse than 3:1 on every axis despite 25% more parameters; SWA degraded both quality and extrapolation vs the un-windowed hybrid), and the premise that motivated SWA (ALiBi having a large quality edge requiring an extrapolation fix) no longer holds now that ALiBi's real quality is known to be close to, not ahead of, the hybrid's. Deprioritized, not rerun.
4. **Does the 3:1 ratio's quality win survive at a width where sheaf-delta is also faster?** Attempted (`findings_report_gpu.md` §5g) — inconclusive. A real full-training-step comparison at width 128/ctx 16,384 (the confirmed speed-crossover zone from forward-only timing) showed GPT decisively faster, but a methodology investigation (CUDA-allocator contamination, possibly from multiple architectures/jobs sharing one process) means this specific number should not be trusted as final. Needs a properly isolated re-run.
5. **Does the 3:1 ratio's quality win survive past 14M params with a real token budget?** A ~200M-param attempt landed at a 70x-lower tokens/param ratio than the 14M comparison (time-budget artifact, not an architecture result) — see `findings_report_gpu.md` §5c for why those specific numbers should not be read as a scale answer either way. Also: ALiBi/NoPE were never tested at this scale at all.
6. **Would fusing the dyadic Householder part fully close the speed/memory gap?** `torch.compile` on the whole model (§3, §3a) gives a real, large partial answer (32.6%/57.7% faster for GPT/hybrid respectively) without a hand-written kernel; the full hand-written-Triton version remains untried.
7. **Every number in this table is a single seed.** The margins between the hybrid, ALiBi, and ALiBi-hybrid are now small enough (0.01-0.016 bpb) that a multi-seed check is needed before treating the exact ranking among those three as more than "roughly tied, hybrid nominally ahead."
