# DEEPA v2 bonus ablation — mixer-internal upgrades from `improvements-scope.md`

**Status: exploratory follow-up, not part of the locked v1 result set.** v1 (`TRAINING_ADVANTAGE.md`, `findings_report_gpu.md`) is untouched by this — every new `Config` field defaults to exact v1 behavior, verified via the existing test suite (`tests/test_fast.py`, `tests/test_equiv.py`, both still pass) plus a dedicated bit-exact `step()==forward()` rollout check for each new flag before any ablation number was trusted.

## Where this came from

`improvements-scope.md` (written by ChatGPT, reviewed independently by Gemini/Antigravity) proposed a set of mixer-internal upgrades to `SheafDeltaMixer`/`FusedSheafDeltaMixer`, ranked by risk. This tests the three ChatGPT/Gemini jointly flagged as free/near-free:

1. **`H=6, dh=64`** instead of `H=8, dh=48` at `D=384` — aligns `fla`'s Triton tiles to a power of 2, +33% per-head state capacity.
2. **Multi-scale decay-bias init** (`linspace(-1.0, 4.5, H)`) instead of the flat `3.0` every head currently gets.
3. **Post-sum `RMSNorm`** (`onorm(o + dyadic)` instead of `onorm(o) + dyadic`) — intended to stop the delta-rule readout from amplifying noise before combining with the dyadic branch.

**A real confound was caught and fixed before running anything:** `Config.heads` is shared between `SheafDeltaMixer` (the `d` layers) and the ALiBi attention module (the `a` layers). Setting `heads=6` globally would have changed the 2 attention layers' head count too — not an isolated mixer ablation. Fixed by adding a separate `mixer_heads` field (falls back to `heads` when unset, so default behavior is unchanged) that only `SheafDeltaMixer` reads.

## Implementation (all additive, all opt-in, all off by default)

`hsnlm/models.py::Config` gained three fields: `mixer_heads: int = 0`, `decay_init: str = "flat"`, `postnorm: bool = False`. `hsnlm/fastmixer.py` and `hsnlm/fastmixer_fused.py` branch on these in `SheafDeltaMixer.__init__`/`forward`/`step`; with every field at its default, the code takes the exact v1 code path (confirmed: `tests/test_fast.py`/`test_equiv.py` pass unchanged, and a rollout check with all-default `Config` reproduces v1's existing bit-exact `step()==forward()` behavior).

## A real, independent bug found while verifying this (orthogonal to the ablation, fixed separately)

While bit-exactness-checking `FusedSheafDeltaMixer`, `step()` and `forward()` turned out **not** to be bit-exact even at v1 defaults (max-abs-diff 0.049 at fp32 on a 64-step random rollout) — confirmed pre-existing via `git stash` (same diff before any of this ablation's changes), and confirmed never covered by the existing test suite. Root cause (diagnosed collaboratively, confirmed by reading `fla`'s own reference implementation, `fla.ops.gated_delta_rule.naive.naive_recurrent_gated_delta_rule`): `chunk_gated_delta_rule` (used in `forward()`) defaults `scale=None`, which resolves internally to `dh**-0.5`, applied as `q = q * scale` before the readout `o = q @ h`. `step()`'s manual einsum had no equivalent scaling. **Fixed in `step()`** (not `forward()` — `forward()` is what every real training run, including this whole ablation, actually learned under; changing it would invalidate already-collected numbers): `o = einsum(...) * s.dh ** -0.5`. Verified: fp32 max-abs-diff drops from 0.049 to 0.0008; bf16 from a much larger systematic offset to 0.009 (ordinary rounding-level noise, same order as the eager `SheafDeltaMixer`'s own long-standing ~8e-6 at fp32 scaled up for bf16/chunked-kernel noise). This is a decode-path correctness fix only — it doesn't touch `forward()`, so it has zero effect on any training/quality number anywhere in this project, including the numbers below (training and eval both go through `forward()` exclusively).

## Results: 3 seeds each, compiled, full eval, same recipe as `gpu_multiseed_compiled.py`'s `alibi-hybrid` (the v1 winner, reused as baseline rather than rerun)

| variant | @1024 mean ± std | @2048 mean ± std | @4096 mean ± std | train_s | tok/s | peak mem |
|---|---|---|---|---|---|---|
| **baseline (v1 ALiBi-hybrid 3:1)** | 1.0273 ± 0.0076 | 1.0067 ± 0.0066 | 1.0831 ± 0.0050 | 313.0 | 78,518 | 2,828.7 MB |
| h6 (`mixer_heads=6`) | 1.0291 ± 0.0016 | 1.0092 ± 0.0022 | 1.0869 ± 0.0057 | 304.4 | 80,736 | 2,720.2 MB |
| decay (multiscale init) | 1.0452 ± 0.0091 | 1.0257 ± 0.0103 | 1.1037 ± 0.0113 | 326.0 | 75,378 | 2,827.0 MB |
| postnorm | 1.0715 ± 0.0148 | 1.0539 ± 0.0163 | 1.1306 ± 0.0217 | 329.7 | 74,534 | 2,753.4 MB |
| combined (all three) | 1.0716 ± 0.0197 | 1.0545 ± 0.0214 | 1.1329 ± 0.0171 | 324.4 | 75,757 | 2,718.2 MB |

Raw data: `results/gpu_v2_ablation.json`, reproducible via `benchmarks/gpu_v2_ablation.py [h6] [decay] [postnorm] [combined]`.

## Verdict: only one of the three "free" upgrades actually delivers

**`h6` (decoupled `mixer_heads=6`, `dh=64`): adopt.** Quality is a statistical tie with baseline (1.0291 vs 1.0273, well within either's seed-to-seed noise) but noticeably *tighter* variance (std 0.0016 vs 0.0076 at @1024), and it's a real, measured efficiency win: 2.8% faster (304.4s vs 313.0s), 3.8% less memory (2,720 vs 2,829 MB). Zero quality cost, real efficiency gain — the one item from `improvements-scope.md` that delivered as advertised.

**`decay` (multi-scale decay-bias init): reject as implemented.** Worse on every length (+0.018 bpb @1024, +0.019 @2048, +0.021 @4096), consistently across all 3 seeds. Root cause: the original proposal's `linspace(-1.0, 4.5, H)` was designed assuming a plain `sigmoid` decay gate (half-life ~14 tokens at bias 3.0); this codebase's actual gate is `0.8 + 0.2*sigmoid(...)` (documented and independently verified earlier this session), so the v1 flat bias of `3.0` already gives every head a ~73-token half-life. Applying the proposed linspace instead pushes 5 of 8 heads down to 4-33-token half-lives — short enough to duplicate what the dyadic branch's `{1,2,4,8,16}`-token lags already cover — while cutting the matrix-memory's long-range retention capacity for those heads. A differently-tuned multi-scale range (e.g. centered higher, informed by the *actual* decay-gate formula) might still be worth trying, but this specific proposal, measured as specified, is a clear regression.

**`postnorm`: reject as implemented, and the worst result of the three.** Worse on every length (+0.044 @1024, widening to +0.050 @4096), and *noisier* than baseline, not more stable (std 0.0148 vs 0.0076). Root cause: `chunk_gated_delta_rule`'s implicit `dh**-0.5` scaling (the same factor behind the `step()` bug above) means the delta-rule readout `o` that `postnorm` sums with the dyadic branch is already ~7x smaller in raw amplitude than the dyadic branch, before any normalization. Summing them pre-normalization under `postnorm=True` forces one `RMSNorm` to arbitrate between two branches at a 7x amplitude mismatch the original proposal didn't account for — the opposite of its intended effect (removing noise amplification). A corrected version (compensating for the `dh**-0.5` factor, or using the un-scaled delta-rule output) is a different, untested variant, not what's measured here.

**`combined` tracks `postnorm` almost exactly** (1.0716 vs postnorm's 1.0715 at @1024) — confirms `postnorm`'s damage dominates and that `decay`'s extra harm doesn't meaningfully compound on top of it at this scale, while `combined` keeps roughly `h6`'s efficiency profile (train_s/mem closer to `h6` than to `postnorm`/`decay`, since `h6`'s savings are structural and independent of the other two flags).

## Round 2 (layered on top of `h6`, the only round-1 win): both rejected too

Two follow-ups were proposed to fix the round-1 failures directly, both implemented as additive `Config` flags (`postnorm_scaled`, `token_shift`) with full `step()`/`forward()` bit-exactness re-verified before trusting any number (eager and fused mixers, every flag individually and combined — diffs at ordinary float-noise level, same as the round-1 checks). Also fixed, independently of this ablation: `FusedSheafDeltaMixer.step()` was missing a `dh**-0.5` readout scale that `chunk_gated_delta_rule` applies by default in `forward()` — a real, pre-existing train/decode mismatch (confirmed via `git stash` to predate any of this ablation's changes), fixed in `step()` only since `forward()` is what every real training run has already learned under.

| variant | @1024 mean ± std | @2048 mean ± std | @4096 mean ± std | train_s | tok/s | peak mem |
|---|---|---|---|---|---|---|
| **h6 (still the best)** | **1.0291 ± 0.0016** | **1.0092 ± 0.0022** | **1.0869 ± 0.0057** | 304.4 | 80,736 | 2,720.2 MB |
| h6 + postnorm_scaled | 1.0587 ± 0.0069 | 1.0401 ± 0.0069 | 1.1173 ± 0.0083 | 299.7 | 81,994 | 2,644.9 MB |
| h6 + token_shift | 1.0751 ± 0.0103 | 1.0566 ± 0.0099 | 1.1317 ± 0.0059 | 316.8 | 77,565 | 3,169.6 MB |
| h6 + both | 1.0539 ± 0.0183 | 1.0352 ± 0.0185 | 1.1136 ± 0.0143 | 312.3 | 78,694 | 3,097.3 MB |

**`postnorm_scaled` (`onorm((o * dh**0.5) + dy)`, undoing the kernel's implicit readout scale before summing): reject.** The amplitude-mismatch fix was real and correctly applied (verified bit-exact), but fixing it didn't rescue the idea — still 0.030 bpb worse than `h6` alone. So the problem was never purely the `dh**-0.5` bug; forcing the *combined* delta+dyadic signal through one normalization appears to genuinely suppress useful signal that v1's design (normalize the delta branch independently, then add the dyadic branch on top) doesn't. It is cheaper and faster (299.7s/2,645MB, both better than `h6` alone) — just not worth it at this quality cost.

**`token_shift` (learnable per-head `alpha_q`/`alpha_k` mixing `x_t` with raw `q_{t-1}`/`k_{t-1}` before `F.normalize`, 1-token state added to `.step()`): reject, and the worst result of this whole ablation.** Worse on every length (+0.046 bpb @1024) *and* costs real speed/memory (+4.1% slower, +16.5% more memory) — a clean negative on every axis, not just a quality/efficiency tradeoff. Plausible reason: this was applied uniformly to all 6 sheaf-delta layers, but the "single-byte key collision" problem it was meant to fix is specific to Layer 1 (only there does `x_t` represent a single raw byte; by layer 2+, `x_t` is already a rich contextual representation, where blending in `t-1` just adds redundant noise instead of resolving a collision that no longer exists). A layer-1-only version is a different, untested experiment.

**`h6 + both` partially but not fully cancels the two regressions** (1.0539, vs a naive additive prediction of ~1.105) — some positive interaction between the two changes, but still clearly worse than `h6` alone, and the noisiest variant in the whole ablation (std 0.0183; one seed ties `h6`'s own level almost exactly, the other two don't) — not a combination to trust even if its mean looked better.

**Verdict, unchanged after two rounds: `h6` alone is still the best known configuration.** Final recommended config: `Config(dim=384, layers=8, heads=8, mixer_heads=6, mlp_mult=3.65, pattern="dddaddda", mixer="sheafdelta-fused")` with ALiBi attention in the 2 `a` layers — quality-tied with the un-modified v1 hybrid, 2.8% faster, 3.8% leaner, 4.7x tighter seed variance.

## Does `h6`'s edge survive much more training? Yes — checked separately (`gpu_chinchilla_15m.py`)

Every number above is at 1,500 steps (~1.73 "tokens"/param), the same recipe as every other result in this project — leaving open whether the win is an early-training artifact (the CPU report's own standing warning). Reran `h6` vs pure ALiBi at 18,000 steps (~20.5 "tokens"/param, in practice ~7.4 repeated epochs over the fixed 40M-byte enwik8 slice): `h6` 0.7820/0.7637/0.8276 bpb @1024/2048/4096 vs pure ALiBi's 0.7924/0.7720/0.8307 — the gap held at @1024 (0.0078→0.0104) and widened at @2048 (0.0004→0.0083) and @4096 (a noise-level disadvantage → +0.0031). One seed, not three, but the central risk this checks for (vanishing with real training) didn't materialize. See `TRAINING_ADVANTAGE.md` open question #5 for the full writeup.

## Does split weight-decay (excluding 1D params + `.V` from decay) help? No — checked and rejected (`gpu_splitwd_3seed.py`)

`train_deepa_phase1.py`'s optimizer splits parameters into decay/no-decay groups (no decay on 1D params or the Householder `.V`) instead of this project's uniform `weight_decay=0.05` on everything. A single run under it looked promising (`h6` at 1.0149 bpb @1024, notably better than `h6`'s uniform-WD 3-seed mean of 1.0291) — but checked properly with 3 seeds each, giving *both* `h6` and pure ALiBi the identical split-WD optimizer for a fair comparison: `h6` split-WD lands at 1.0324±0.0032 (nominally *worse* than uniform-WD's 1.0291±0.0016), and pure-ALiBi split-WD lands at 1.0407±0.0091 (also worse, and 8x noisier than uniform-WD's 1.0369±0.0012). **Rejected** — the single-run result was favorable noise. The core `h6`-beats-pure-ALiBi margin is unaffected either way (0.0078 bpb under uniform-WD, 0.0083 under split-WD) — the ranking is robust to this choice, just not improved by it.

## What's not yet tried

- A layer-1-only version of the token-shift (only the first sheaf-delta layer gets it) — the round-2 result suggests this narrower form is where any real benefit, if there is one, would have to live.
- A re-tuned `decay_init` range informed by the actual `0.8 + 0.2*sigmoid(...)` gate formula (e.g. targeting longer, not shorter, half-lives than v1's flat 73-token baseline).
- `h6` has not been tested combined with pure ALiBi or GPT baselines, nor at the width-128/ctx-16,384 configuration from `findings_report_gpu.md` §5j.
- Nothing in this document has been tested past 14M params or beyond this single 8GB GPU.
