# Inference Advantage — Hybrid Sheaf-Delta LM vs matched Transformer

Companion to `ARCHITECTURE.md` and `TRAINING_ADVANTAGE.md`. This is the one axis in this project with the cleanest, least-caveated evidence — it's about the *mechanism* of decoding, which doesn't depend on resolving the open training-scale questions in `TRAINING_ADVANTAGE.md` §5.

> **In plain English:** once a model is trained, "inference" (generating new text) is the thing users actually experience — every chatbot reply, every autocomplete suggestion. An ordinary Transformer has to keep a photocopy of the entire conversation so far (the "KV cache") and re-check it on every new word, so generating gets slower and more memory-hungry the longer a conversation runs. This architecture's "notebook" layers don't grow at all — generating the 100th word and the 100,000th word cost almost exactly the same. That's the single cleanest, most dependable advantage measured in this whole project — it follows directly from how the mechanism works, not from a model's training quality, so it holds true regardless of which flavor of ordinary Transformer it's compared against.

## 1. Why this is the strongest claim in the project

Training speed/quality findings depend on width, ratio, and token budget (see `TRAINING_ADVANTAGE.md` §1 — the hybrid wins on quality and extrapolation against every baseline including ALiBi, but by a small margin, and loses on training speed/memory against pure-attention baselines). Decode cost does not depend on any of that — it follows directly from the mechanism: attention must read a growing KV cache every step; the sheaf-delta layer carries a **fixed-size** state (a matrix memory plus a small dyadic buffer) that gets updated, not grown, every step. This was measured on CPU (`findings_report.md` §6) and reconfirmed on GPU at 4x the context range (`findings_report_gpu.md` §6.1), same shape both times.

**This holds regardless of which Transformer position-encoding variant it's compared against — confirmed directly, not assumed.** NoPE and ALiBi only change how attention *scores* are computed; neither changes what attention has to *store*. A direct decode-step comparison (width 1024, 16 heads, batch 1) shows RoPE and ALiBi have nearly identical per-step cost at every context length (e.g. 52.9 vs 57.3 ms at ctx 65,536 — ALiBi slightly slower, not faster), and NoPE is cheaper only at *short* context (skips RoPE's trig-function overhead — 0.25 vs 4.55 ms at ctx 256) but converges to nearly the same cost as RoPE by long context (51.2 vs 52.9 ms at ctx 65,536) once raw attention compute dominates. **Critically, all three have an identical, growing KV cache at every context length** — 2.1 MB at ctx 256 up to 536.9 MB at ctx 65,536, for RoPE, NoPE, and ALiBi alike. None of them touch the actual mechanism (attention must store every past key/value); they only change how the scores computed from that cache are biased. This is the clean, mechanistic reason the inference advantage in this document survives the §1 correction in `TRAINING_ADVANTAGE.md` even though the quality/speed advantage did not: inference cost is about the memory *mechanism*, and ALiBi/NoPE don't change the mechanism, only the training-time quality of using it.

## 2. The numbers (GPU, same hardware, same script, same run — `benchmarks/gpu_decode.py`)

Per-layer, single-token decode, batch 1, width 1024, 16 heads, CUDA-synced:

| context | attention step | GPT KV cache | sheaf-delta step | sheaf-delta state |
|---|---|---|---|---|
| 256 | 0.54 ms | 2 MB | 0.99 ms | 0.33 MB |
| 1,024 | 0.60 ms | 8 MB | 0.96 ms | 0.33 MB |
| 4,096 | 1.46 ms | 34 MB | 1.00 ms | 0.33 MB |
| 16,384 | 4.84 ms | 134 MB | 0.96 ms | 0.33 MB |
| 65,536 | 18.14 ms | 537 MB | 0.98 ms | 0.33 MB |

**Crossover point:** between ctx 1,024 and 4,096, attention step time overtakes sheaf-delta's (flat) cost. Below that, attention is actually a bit cheaper per token (less fixed overhead) — the advantage is specifically a *long-context* one, not universal.

**Past crossover, the gap only widens:** at 65,536 tokens of context, attention takes **18x longer per decoded token** and its KV cache is **1,628x larger** (537 MB vs 0.33 MB) than sheaf-delta's state. The state size literally does not appear in the table as a function of context — it's the same 0.33 MB at every row.

## 3. Why the memory number matters more than the latency number for real serving

In production LLM serving, the KV cache is usually the actual ceiling on throughput, not raw compute: it's held *per sequence*, so it multiplies by how many requests are being served concurrently. A model whose GPU can hold, say, 50 concurrent long-context Transformer sessions before running out of KV-cache memory could hold dramatically more sheaf-delta sessions in the same memory, because each one costs 0.33 MB instead of hundreds of MB at long context. This is a batch-serving/cost argument, not a single-request-latency one, and it gets stronger exactly as context length grows — which is also the direction real usage (long documents, long conversations, agent traces) has been moving.

## 4. The quality-winning hybrid (3:1), measured directly — full 8-layer stack, not just one isolated layer

§2 measured one bare mixer/attention layer in isolation. Here's the realistic number: a full 8-layer stack's total per-token decode time, batch 1, width 1024, comparing GPT (8 attention layers) vs pure sheaf-delta (8 sheaf-delta layers) vs the quality-winning hybrid (6 sheaf-delta + 2 attention, pattern `dddaddda`):

| context | GPT, 8 layers | pure sheaf-delta, 8 layers | **hybrid 3:1, 8 layers** | GPT KV cache | hybrid KV cache (2 attn layers) |
|---|---|---|---|---|---|
| 256 | 22.04 ms | 10.28 ms | **5.25 ms** | 16.8 MB | 4.2 MB |
| 1,024 | 23.33 ms | 9.46 ms | **5.96 ms** | 67.1 MB | 16.8 MB |
| 4,096 | 29.76 ms | 9.57 ms | **7.28 ms** | 268.4 MB | 67.1 MB |
| 16,384 | 91.54 ms | 9.58 ms | **13.64 ms** | 1,073.7 MB | 268.4 MB |
| 65,536 | (OOM in this project's runs) | — | **41.67 ms** | — | 1,073.7 MB |

**Updated after a later mixer optimization (`fastmixer.py`'s zero-gather dyadic rewrite + a vectorized `step()`, `findings_report_gpu.md` §5h) — the hybrid's decode cost roughly halved at every context length vs the original measurement** (e.g. 13.56→5.25 ms at ctx 256), extending cleanly out to ctx 65,536 for the first time. **The hybrid wins outright, at every context length tested, including the shortest one** — it's not a long-context-only advantage once a full realistic stack is measured. The gap widens with context as expected, though the hybrid's 2 full-attention layers still mean its own decode cost *grows* with context (it is not flat/O(1) — only the 6 sheaf-delta layers are) — by 65,536 it's still ~2x faster than GPT would be at that length (extrapolating GPT's own growth trend) and uses 4x less KV cache. Pure sheaf-delta is the only fully flat/O(1) option (flat ~9.5-10.3 ms regardless of context, no KV cache at all) — the hybrid trades some of that flatness for the quality win documented in `TRAINING_ADVANTAGE.md` §1, and still comes out clearly ahead of the Transformer on both speed and memory. (A separate experiment tried capping the hybrid's attention layers to a sliding window to make its decode cost fully flat too — it worked for decode cost, but measurably hurt training quality and extrapolation, so it was not adopted; see `findings_report_gpu.md` §5h and `TRAINING_ADVANTAGE.md` §5.)

## 5. The throughput primitive behind block/self-speculative decoding — measured, with a real finding

**What this is not:** a trained diffusion draft model, an acceptance-rate measurement, or a full speculative-decoding loop. **What this is:** the core mechanism block/self-speculative decoding (`RESEARCH_V2.md` §5) depends on — does processing `b` positions in one forward pass get cheaper per token as `b` grows, for an architecture whose single-token decode is dominated by fixed overhead? (`benchmarks/gpu_blockdecode.py`, width 1024, 8 layers, inference-only/no backward, fused kernel for the sheaf-delta parts.)

| block size `b` | GPT ms/token | pure sheaf-delta ms/token | hybrid 3:1 ms/token |
|---|---|---|---|
| 1 (naive one-at-a-time) | 12.40 | **44.64** | **35.42** |
| 8 | 1.64 | 5.73 | 4.58 |
| 32 | 0.55 | 1.45 | 1.15 |
| 128 | 0.23 | 0.39 | 0.36 |
| 512 | 0.18 | 0.21 | 0.23 |
| 2,048 | 0.187 (*worse* than b=512) | 0.200 | **0.197** |

**The finding:** at `b=1`, the fused sheaf-delta kernel is dominated by Triton kernel-launch overhead and is actually 3-4x *slower per token* than GPT — a real weakness, consistent with every other "eager/small-batch GPU overhead" finding in this report (`findings_report_gpu.md` §3, §3.4). But it improves **190-225x** from `b=1` to `b=2,048`, and keeps improving past the point where GPT's own improvement stalls and slightly reverses (0.181ms at b=512 → 0.187ms at b=2,048 — an early sign of O(N²) attention cost creeping back in). By `b=2,048` the hybrid (0.197ms/token) is within 5% of GPT and still on a downward trend; GPT is not.

**What this actually means:** block/self-speculative decoding isn't a nice-to-have for this architecture — it is fixing the single biggest weakness this report found in it (naive single-token GPU decode). It hasn't been shown to make sheaf-delta *beat* GPT's absolute per-token cost at block sizes a real speculative-decoding setup would use (typically 4-64 draft tokens, where GPT is still 2-8x cheaper in this test) — that would need either larger blocks than are typical for speculative decoding, or fixing the `b=1` overhead directly (better kernel warm-up/CUDA graphs), neither tested here. The honest claim: **the throughput primitive behaves exactly as the design doc predicted, and growing blocks closes the gap, but does not yet close it at realistic speculative-decoding block sizes.**

## 6. Multimodality and real diffusion serving — still design-only

Two things from `RESEARCH_V2.md` §3/§5 remain genuinely untested, and §5 above does not change that:
- **An actual trained block-diffusion draft model** — §5's test used teacher-forced forward passes on random tokens, not a model that's learned to propose a block of future tokens. The acceptance rate that would make a real speculative-decoding loop pay off (`RESEARCH_V2.md` §5 estimates ~3.3 tokens/pass at 70% per-token acceptance, γ=8 — a guess, not measured anywhere) is still unmeasured.
- **Multimodal data of any kind.** No image/audio/video tokenizer or dataset has been run through this architecture. The flat-state argument (§2-§4) is a property of the mechanism, not evidence from multimodal data.

## 7. Summary

| Claim | Status |
|---|---|
| O(1) decode cost in context length | **Measured, CPU and GPU, consistent** |
| O(1) decode memory (state) in context length | **Measured, CPU and GPU, consistent, extended to ctx 65,536 on GPU** |
| Crossover point where this starts winning | **Measured: ~1-4k tokens of context**, below which attention is actually cheaper |
| Same numbers for the quality-winning 3:1 hybrid specifically | **Measured (§4), updated after a mixer optimization**: wins at every context tested (256-65,536), decode cost roughly halved at every point vs the original measurement; still grows with context (not flat) because of its 2 full-attention layers — a sliding-window variant made it flat but cost quality, so wasn't adopted |
| Block/speculative-decode throughput primitive | **Measured (§5)**: real and large (190-225x improvement b=1→2048), closes but does not yet flip the gap vs GPT at realistic block sizes |
| Benefit materializes in real multimodal/diffusion serving | Still design argument only — no draft model, no acceptance rate, no multimodal data (§6) |
