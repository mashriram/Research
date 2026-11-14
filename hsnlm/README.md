# DEEPA — Dyadic-Delta Enhanced Sheaf, Efficient Pairing Architecture

An experimental language model architecture that replaces most of a Transformer's attention layers with a cheaper, fixed-size "memory" layer — tested, measured, and honestly reported, including the mistakes made along the way.

## What this is, in one paragraph

Ordinary Transformers (the architecture behind GPT, Claude, and nearly every modern language model) answer every new word by comparing it against *every word that came before* — a mechanism called attention. This works very well, but the cost of remembering the conversation (the "KV cache") grows forever as the conversation gets longer. **DEEPA** mixes in a different kind of layer — a **sheaf-delta layer** — that keeps a small, *fixed-size* running summary instead of a growing list, updated every word rather than grown. Three of these "memory" layers paired with one ordinary attention layer (a 3:1 ratio, found by testing every ratio, not guessed) turned out to be the best combination: **DEEPA beats every flavor of ordinary Transformer tested — including the strongest one — on writing quality and on handling text longer than it was trained on, by a small but real margin, and generates each new word far more cheaply in long conversations. It costs more time and GPU memory to train, though — that trade-off is real and didn't go away.**

Scale tested: 14 million parameters (small, controlled, research-scale — not a production model). Nothing here has been tested beyond that scale.

## Final verdict (full numbers in `TRAINING_ADVANTAGE.md` and `findings_report_gpu.md` §5h)

| Axis | Result |
|---|---|
| Writing quality (at the length it was trained on) | **DEEPA wins**, by a small margin, against every Transformer variant tested |
| Handling longer text than it was trained on | **DEEPA wins**, by a small margin, against every variant |
| Cost of generating each new word (long conversations) | **DEEPA wins**, by a large margin — its cost stays flat; a Transformer's keeps climbing |
| Training speed and memory | **DEEPA loses** — the best Transformer variant trains ~1.7x faster and uses ~40% less memory |

Not "better in every way" — a real, modest win on quality/long-text/generation-cost, at a real, persistent cost in training speed/memory. See `ARCHITECTURE.md` §2 for the full verdict with the story of how it was double-checked (two real measurement bugs were found and fixed before these numbers were trusted).

## Reading order

| Doc | What's in it |
|---|---|
| **`ARCHITECTURE.md`** | **Start here.** Full description of DEEPA, in plain English with a technical appendix. What it is, why each piece is there, the final verdict. |
| `TRAINING_ADVANTAGE.md` | Does DEEPA actually write better text than a matched Transformer, and how much does it cost to train? All the quality/speed/memory numbers. |
| `INFERENCE_ADVANTAGE.md` | The cleanest result in the project — generating text with DEEPA vs. a Transformer, measured directly. |
| `findings_report_gpu.md` | The full GPU-phase lab notebook — every experiment run, including dead ends, bugs found and fixed, and things that didn't work. The most detailed and most honest document in the project. |
| `findings_report.md` | The original, smaller-scale CPU-only experiments that started this project. |
| `RESEARCH_V2.md` | The original design document and brainstorm — ideas, some tested and confirmed, some dropped, some still just plans. |
| `RESULTS.md` | The very first raw benchmark numbers, kept for historical record. |
| `DEEPA_full_report.pdf` | All of the above, combined into a single PDF. |

## A note on how this was built

This project's numbers were checked, found wrong twice, and corrected twice before being written down as final — not because the architecture doesn't work, but because two real bugs (a mismatched data comparison, and a dropped connection in a comparison model) briefly made the results look dramatically better and then dramatically worse than they actually are. Both are documented in full, including how they were caught, in `findings_report_gpu.md` §5h. This is mentioned prominently, not buried, because it's the reason the final numbers can be trusted: every surprising result in this project was treated as a bug to investigate before being treated as a finding to report.

## Reproducing the results

```bash
cd hsnlm
uv venv .venv --python 3.12 && source .venv/bin/activate
uv pip install torch --index-url https://download.pytorch.org/whl/cu124 numpy flash-linear-attention
# data (not checked into git):
#   data/enwik8          — mattmahoney.net
#   data/shakespeare.txt  — karpathy/char-rnn
```

All benchmark scripts live in `benchmarks/`, and their raw results in `results/` (as JSON + logs). The scripts referenced throughout the docs above (e.g. `gpu_ratio.py`, `gpu_posenc.py`, `gpu_isoquality_compiled.py`) are runnable as-is once the data files above are in place. See `findings_report_gpu.md` §12 for the full list and what each one measures.

## Status

Research-scale (14M params), single-GPU (one consumer 8GB card), mostly single-seed. Not a production model. Open questions that still need answering before scaling this up are listed at the end of `TRAINING_ADVANTAGE.md` and `findings_report_gpu.md`.
