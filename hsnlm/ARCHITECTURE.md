# DEEPA — Dyadic-Delta Enhanced Sheaf, Efficient Pairing Architecture

*(Formerly referred to as the "Hybrid Sheaf-Delta LM" throughout this project's earlier docs — same architecture, now with a name. "Dyadic-Delta" = the two halves of each memory layer (the local dyadic mixing add-on and the delta-rule matrix memory); "Enhanced Sheaf" = the "sheaf" naming already used throughout this codebase, enhanced by the fused GPU kernel that makes it fast; "Efficient" = the measured training/inference efficiency results; "Pairing" = the 3:1 pairing of memory layers with attention layers.)*

**Status:** the best-defended configuration from `findings_report.md` (CPU phase) and `findings_report_gpu.md` (GPU phase), as of 2026-11-15. Not a finished product — see the open questions at the end of `TRAINING_ADVANTAGE.md` and `INFERENCE_ADVANTAGE.md` before treating any number here as final.

---

## 0. In plain English — what this is, and why it exists

Every large language model (ChatGPT, Claude, etc.) is built from a stack of identical "blocks" that take in the text so far and predict the next word. The near-universal choice for how a block looks back at earlier text is **attention**: for every new word, the model compares it against *every single earlier word* in the conversation to decide what's relevant. This works extremely well, but it has one structural cost: the model has to keep a running list (the **KV cache**) of every word it has seen, and that list keeps growing as the conversation gets longer. Twice the conversation length means roughly twice the memory and, eventually, much more than twice the compute per new word — because every new word has to be compared against a longer and longer list.

This project builds and tests an alternative building block — a **sheaf-delta layer** — that does not keep a growing list at all. Instead of remembering every word individually, it keeps a small **fixed-size "notebook"** (technically, a matrix of numbers) that gets *updated*, not grown, with every new word — closer to how a person might jot down a running summary rather than photocopying the entire conversation so far. The notebook never gets bigger, no matter how long the conversation runs, so the cost of generating each new word stays flat instead of climbing.

The catch: a notebook that only keeps a compressed summary is worse at very precise, exact lookups than a model that can see the entire original text ("what *exact* word did I use three paragraphs ago?"). So this project doesn't replace attention entirely — it builds a **hybrid**: mostly notebook-style layers, with a few ordinary attention layers mixed in to handle the precise lookups the notebook isn't good at. The question this whole project spent a day answering, with real measurements rather than guesswork, is: *what's the best mix, and does it actually beat a well-built ordinary Transformer?*

**The one-paragraph answer:** yes, by a small but real and carefully verified margin, on writing quality and on handling text longer than what it was trained on, and by a large margin on the cost of generating each new word in long conversations. The real trade-off: it takes longer and more memory to *train* than a plain Transformer of the same size. Nothing here claims a free lunch — every number below has a matching cost somewhere else, and two real bugs were found and fixed while measuring this (see §2 "Final verdict" and `findings_report_gpu.md` §5h), so the numbers here were checked twice before being trusted.

**Glossary, for the technical sections below:**
- **bpb (bits per byte):** how well the model predicts the next character of text — lower is better, like a golf score. A model that's just guessing randomly among 256 byte values scores 8.0; a genuinely good model on English text scores around 1.0-1.1.
- **tok/s (tokens per second):** raw training or generation speed — higher is better.
- **KV cache:** the growing "photocopy of everything said so far" that ordinary attention needs to keep.
- **Extrapolation:** how well a model holds up when asked to handle more text than it was ever trained on (e.g. trained on 1,024-word excerpts, tested on 4,096-word ones).
- **Parameters:** the number of adjustable numbers inside the model — roughly, its "size." 14 million parameters here is tiny by modern standards (GPT-4-class models have hundreds of billions) — this is a controlled, small-scale experiment, not a production model.

---

## 1. What it is, technically

A causal language model built from two layer types, mixed in a fixed ratio, trained end-to-end with ordinary backpropagation (the standard way every mainstream neural network is trained):

- **Sheaf-delta layer** (`mixer="sheafdelta-fused"` in `hsnlm/models.py`): the "notebook" layer described above. Technically, a Householder-type gated delta-rule matrix memory (`S_t = a_t S_{t-1}(I - b_t k_tk_t^T) + b_t v_t k_t^T` — a fixed-size matrix `S` that gets updated every step, never grown) plus cheap "dyadic Householder" local mixing (a lightweight mechanism for blending in the last few words directly, on top of the notebook). The matrix-memory part runs through a **fused Triton kernel** (`flash-linear-attention`'s `chunk_gated_delta_rule`, wired in as `hsnlm/fastmixer_fused.py`) for GPU speed; the local-mixing part is still ordinary PyTorch code (see §4).
- **Attention layer**: a standard, off-the-shelf Transformer block (pre-normalization, RoPE positional encoding, `scaled_dot_product_attention`) — the same building block used in most production language models, unchanged.

**Mixing ratio: 3 sheaf-delta ("notebook") layers for every 1 attention ("full lookup") layer**, spread evenly through the model's depth (e.g. 8 layers total → pattern `dddaddda`, where `d` = notebook layer, `a` = attention layer). This ratio isn't a guess — it's the measured best point from testing every ratio from "all notebook, no attention" to "all attention, no notebook" (`findings_report_gpu.md` §5a), and it beat the ratio originally guessed from published papers (7 notebook layers to 1 attention layer) on every axis measured: quality, long-text handling, speed, and memory, all at once.

**Training:** AdamW optimizer, bf16 (a faster, lower-precision number format used throughout modern GPU training), ordinary backpropagation. No experimental "replace backprop" tricks of any kind — three separate attempts in this project (described in `TRAINING_ADVANTAGE.md` §4) all confirmed that nothing tried saves real computation or preserves quality, so backprop was kept.

**Memory policy (optional):** for training setups that are tight on GPU memory, two opt-in tricks are available — recomputing each layer's output during the backward pass instead of storing it (`torch.utils.checkpoint`), or (if the model is pure notebook-layers with no attention mixed in) an exact "reversible" variant (`hsnlm/reversible.py`). Both trade roughly 20-33% extra compute time for a 4-5x cut in memory use. Neither is the default. See `TRAINING_ADVANTAGE.md` §3.

---

## 2. Why each piece is in there, and the final verdict on whether it works

| Piece | Why it's there, in plain terms | Technical evidence |
|---|---|---|
| Notebook-style matrix memory, not a simpler "running average" | A simple running-average memory (the kind used in many cheap recurrent models) is mathematically provably unable to solve certain exact-tracking tasks (like checking whether a sequence has an even or odd number of flips — "parity"). This project's matrix-memory design can solve that task; simpler alternatives tested alongside it cannot. | `findings_report.md` §5.3, §4 |
| The small "blend in the last few words" add-on (dyadic local mixing) | Cheap, well-behaved, and handles very local context directly — but testing confirmed it is *not* where the model's ability to recall things from far back in the text comes from; that's the matrix memory's job. | `findings_report.md` §4 |
| 3:1 ratio of notebook layers to attention layers | All-notebook loses writing quality once the model is scaled up past the tiniest size tested, even though it won at the very smallest size; all-attention (an ordinary Transformer) degrades badly on text longer than it was trained on. The 3:1 mix wins on writing quality *and* long-text handling against every version of an ordinary Transformer tested — including the strongest one found (a Transformer variant called ALiBi) — by a real but small margin (see "final verdict" below). | `findings_report_gpu.md` §5, §5a, §5d, §5h |
| Fused Triton kernel for the notebook-memory math | Without this GPU-specific optimization, the notebook layers are 20-150x *slower* than attention on a GPU (lots of small operations that GPUs handle poorly one at a time); the optimization closes this gap and, past a certain conversation length, flips it so the notebook layers are the faster option. | `findings_report_gpu.md` §3 |
| Keep ordinary backpropagation | Every attempted alternative — three independent experiments across this project — either failed outright or didn't actually save computation; see §4 below. | `findings_report.md` §7; `findings_report_gpu.md` §7-§8 |

### Final verdict, stated plainly (full numbers in `TRAINING_ADVANTAGE.md` and `findings_report_gpu.md` §5h)

At the scale actually tested (14 million parameters — small, controlled, not production-scale), comparing the 3:1 hybrid against a careful lineup of ordinary-Transformer variants (plain, ALiBi, YaRN-extended, position-encoding-free):

- **Writing quality at the length it was trained on:** the hybrid wins against every variant tested, including the best one (ALiBi) — but only by a small, real margin (about 1.5% better bpb), not a dramatic one.
- **Handling text 2-4x longer than it was trained on:** same story — the hybrid wins against every variant, margin small but real.
- **Cost of generating each new word in a long conversation:** the hybrid wins by a large, clear margin — the notebook layers' cost stays flat as the conversation grows, while a pure Transformer's cost keeps climbing.
- **Training speed and training memory:** here the hybrid *loses* — an ordinary Transformer using the best position-encoding variant (ALiBi) trains about 1.7x faster and uses about 40% less memory. This cost is real and did not go away.

**Important correction history, stated for transparency:** midway through this project, a buggy measurement briefly suggested ALiBi *beat* the hybrid decisively on quality; later, a second bug (traced down and fixed, see `findings_report_gpu.md` §5h) revealed that comparison had been measuring a broken copy of the ALiBi model (missing a connection that's supposed to carry information around each layer) and comparing it against a different, easier slice of test data than the hybrid was scored on. Once both bugs were fixed and the attention baselines were retrained correctly, the original result — the hybrid wins, modestly — held up. This is mentioned here not as a footnote but because it's the main reason every number in this document can be trusted: they were checked, found wrong once, re-checked, and verified a second time before being written down as final.

---

## 3. What's explicitly NOT in the final architecture, and why

- **No experimental "replace backpropagation" tricks, no fancy low-rank compression layers.** All tested (on CPU, GPU, or both) and found not to help — see `TRAINING_ADVANTAGE.md` §4-§5.
- **No alternative output-layer math, no exotic "infinite-depth" solvers.** These were considered in the project's original design notes (`RESEARCH_V2.md`) but dropped by reasoning alone before spending any GPU time on them — see `findings_report.md` §10 for the specific reasoning on each. Not contradicted by anything measured since; simply never prioritized.
- **No "diffusion-style" parallel text generation yet.** Discussed as a promising future direction in §5 below and in `RESEARCH_V2.md`, but zero lines of code or measurement exist for it in this project. Stated honestly as an idea, not a result.

---

## 4. What's not yet optimized — and a cheap fix that already works well

The GPU speed optimization (§1) only covers the matrix-memory math itself. The "blend in the last few words" add-on is still ordinary, unoptimized code, and it's part of why the notebook layer currently uses *more* peak GPU memory per word than an attention layer, not less (`findings_report_gpu.md` §3.3). Hand-writing a custom GPU kernel for it would fully fix this, but that's serious low-level engineering with real risk of introducing subtle bugs if rushed — not attempted here.

**What was tried instead, and worked well: `torch.compile`** — a one-line PyTorch feature that automatically finds and fuses together the model's existing operations, without writing any new low-level code. Measured two ways, at the 14-million-parameter scale: compiling just the "blend in the last few words" add-on (notebook-only model: 42% faster / 31% less memory; the 3:1 hybrid: 41% faster / 30% less memory), and — a bigger win — compiling the **entire model** at once: an ordinary Transformer got **32.6% faster**, and the hybrid got **57.7% faster**, both measured over real, full training runs, not a quick synthetic test.

This doesn't fully close the memory gap to an ordinary Transformer even with the whole-model version (compiled Transformer: 2,408 MB; compiled hybrid: 2,868 MB, still about 19% more) — but it's a real, low-risk, easy win. One more genuinely useful fact fell out of this same test: when both models are compiled the same (fair) way, the hybrid reaches the *same* writing quality an ordinary Transformer tops out at using **30.7% less training time and 42.9% fewer training examples** — because each example teaches the hybrid slightly more per step, even though each step itself costs a bit more. (`TRAINING_ADVANTAGE.md` §3a has the full numbers and how this was checked fairly, rather than by comparing a compiled hybrid against an uncompiled Transformer, which would have overstated the result.)

---

## 5. Extensibility: multimodality and "diffusion-style" generation

**Everything in this section is reasoning from the architecture's measured properties, not a tested result.** Nothing in this project has actually run an image, audio, or "generate-many-words-at-once" experiment. Read this as "here's why it's plausible," not "here's proof it works."

### 5a. Beyond text (images, audio, etc.)

The model's core design is "numbers in, numbers out" with no built-in assumption that those numbers represent English text specifically. Images, audio, and video can be converted into sequences of discrete codes (a standard, well-established technique already used by most multimodal AI systems), and those code sequences would plug into the exact same layers used for text — no redesign needed. Two properties measured in this project specifically support this working well:
- **The notebook's fixed size** (it never grows, regardless of how long the input is) matters even more for audio/video, since those are typically converted into *far more* individual codes than the equivalent amount of text would be.
- **Good handling of longer-than-trained-on input** (measured on text) would matter the same way for long audio/video streams.
- None of this has actually been tried with real image or audio data. This is an argument from measured properties, not a finished result.

### 5b. Generating multiple words at once ("block diffusion")

A more ambitious future idea (from the project's original design notes, `RESEARCH_V2.md`): instead of generating one word at a time, generate a whole block of words at once in a "fill in the blanks" style, then just verify/correct them one pass later — plausible specifically because of two things this project did measure: the notebook's fixed-size memory means verifying a block doesn't cost more memory just because the conversation is long, and generating a word normally already keeps the per-word cost flat for the notebook layers. **None of this has been built or tested in this project.** It remains the most promising, least-tested idea in the whole research line — worth pursuing next, not something to claim as working today.

### 5c. Honest summary

The properties this project actually measured (fixed-size memory, flat per-word generation cost, good handling of longer-than-trained text) are exactly the right properties for both of the extensions above to work well — which makes sense, since both were explicit goals from the start of the project (`RESEARCH_V2.md`). But "has the right properties for X" and "X actually works" are two different claims, and only the first one has real evidence behind it here.
