"""Builds benchmark_results/PHASE_A_RESULTS.md from the four Phase-A result JSONs (re-runnable; skips missing cells)."""
import json, os, statistics as st
K3 = ("alibi-hybrid-h6", "alibi-pure", "rope-gpt")
NAME = {"alibi-hybrid-h6": "ALiBi-DEEPA 3:1 (h6)", "alibi-pure": "Pure ALiBi", "rope-gpt": "RoPE GPT"}
def J(p): return json.load(open(p)) if os.path.exists(p) else {}
L = []; w = L.append
def ms(v, d=4): return f"{st.mean(v):.{d}f} ± {st.pstdev(v):.{d}f}" if len(v) > 1 else f"{v[0]:.{d}f}"

w("# Phase A results: reviewer-response experiments\n")
w("All runs: byte-level LM, vocab 256, 14.4M-param width-384 recipe unless stated (AdamW β=(0.9,0.95), uniform wd 0.05, "
  "lr 2e-3, 100-step warmup, cosine to 0.1×, clip 1.0, B=16, ctx 1024, bf16, torch.compile), single RTX 2000 Ada Laptop GPU (8 GB). "
  "bpb = bits per byte (lower is better). ± = population std (ddof=0) over seeds, the same convention as Table 9.\n")
w("**Evaluation-slice caveat (applies to every earlier table in the paper too).** The repo convention `val_bpb` scores only the first "
  "64 windows × 1024 = 64 KB of the 1 MB validation slice. On enwik8 that region is much easier than the rest: the seed-0 h6 model scores "
  "1.02 bpb on it but 1.21–1.90 on each later 64 KB block. Relative comparisons on the same windows remain valid, but absolute bpb on the "
  "64 KB slice is optimistic. The new tables therefore also report `full` = all 976 windows of the validation slice, or a fixed 512 KB budget.\n")

W = J("benchmark_results/paper/wikitext103_3seed/gpu_wikitext103_3seed.json")
if W:
    w("## 1. Additional dataset: WikiText-103 (raw, byte-level), 3 seeds, 1,500 steps (Reviewer items 4, 10)\n")
    w("Train = first 40 MB of the WikiText-103-raw train split; val = first 1 MB of the official validation split (sizes match enwik8).\n")
    w("| Model | Params | bpb @1024 (64 KB) | @2048 | @4096 | bpb full val @1024 (1 MB) | train tok/s | peak train MB |")
    w("|---|---|---|---|---|---|---|---|")
    for k in K3:
        rs = [W[f"{k}|seed{s}"] for s in range(3) if f"{k}|seed{s}" in W]
        if not rs: continue
        g = lambda m: [r[m] for r in rs]
        w(f"| {NAME[k]} | {rs[0]['params']/1e6:.2f}M | {ms(g('val_bpb'))} | {ms(g('val_bpb_len2048'))} | {ms(g('val_bpb_len4096'))} | "
          f"{ms(g('val_bpb_full'))} | {st.mean(g('tok_s')):,.0f} | {st.mean(g('peak_mem_mb')):,.0f} |")
    w("\n**Finding:** on WikiText-103, pure ALiBi beats the hybrid at every length and every seed (gap ≈ 0.03 bpb on the 64 KB slice, "
      "≈ 0.012 bpb on the full validation set, many times the seed std). The hybrid's enwik8 advantage does **not** transfer to this dataset "
      "at this budget. Both ALiBi models extrapolate stably to 2×/4× while RoPE GPT degrades sharply, the same pattern as on enwik8.\n")

E = J("benchmark_results/paper/length_extrapolation_16k/gpu_extrap_16k.json")
if E:
    LENS = (1024, 2048, 4096, 8192, 16384)
    w("## 2. Length extrapolation to 16× the training length (Reviewer item 7)\n")
    w("Seed 0, trained at ctx 1024 on enwik8 (1,500 steps), evaluated zero-shot. Each length is scored on the **same 512 KB** "
      "of validation text (512×1k, 256×2k, 128×4k, 64×8k, 32×16k windows), so the numbers are directly comparable across lengths.\n")
    w("| Model | " + " | ".join(f"L={l}" for l in LENS) + " | Δ(16k − 1k) |"); w("|---|" + "---|" * (len(LENS) + 1))
    for k in K3:
        r = E.get(f"{k}|seed0")
        if not r or f"val_bpb_len{LENS[-1]}" not in r: continue
        v = [r[f"val_bpb_len{l}"] for l in LENS]
        w(f"| {NAME[k]} | " + " | ".join(f"{x:.4f}" for x in v) + f" | {v[-1]-v[0]:+.4f} |")
    w("\nPer-position bpb inside a 16,384-byte window (mean over 32 windows, 1,024-position buckets; positions > 1,024 were never seen in training):\n")
    w("| Model | " + " | ".join(f"{i}k–{i+1}k" for i in range(16)) + " |"); w("|---|" + "---|" * 16)
    for k in K3:
        r = E.get(f"{k}|seed0", {})
        if "position_bpb_16k" in r: w(f"| {NAME[k]} | " + " | ".join(f"{x:.2f}" for x in r["position_bpb_16k"]) + " |")
    w("\n**Finding:** both ALiBi models keep improving as context grows (no degradation out to 16k), and the hybrid leads pure ALiBi by "
      "0.020–0.033 bpb at every length (single seed). RoPE GPT collapses immediately past its training length (bpb 1.58 → 5.48), "
      "and the per-position profile shows the collapse begins exactly at position 1,024. Eval peak VRAM at 16k (B=1): "
      + ", ".join(f"{NAME[k]} {E[f'{k}|seed0'].get('eval_peak_mem_mb_len16384', 0):,.0f} MB" for k in K3 if f"{k}|seed0" in E) + ".\n")

S = J("benchmark_results/paper/scaling_2m_to_38m/gpu_scaling_law.json")
if S:
    w("## 3. Parameter scaling at a matched 5 tokens/parameter (Reviewer item 6)\n")
    w("Seed 0. Both models at a scale get the identical step count (5 × hybrid params / 16,384 tokens per step). Train = enwik8 minus the "
      "validation slice (99 MB, ≤ 2 epochs); val = the paper's standard 1 MB slice. LR 2e-3 for ≤ 14M; at ~38M each model uses its best "
      "LR from {1e-3, 2e-3} (1,000-step sweep).\n")
    w("| Scale | dim / layers / attn heads / mixer heads | Model | Params | Steps | Tokens | LR | bpb @1024 (64 KB) | @2048 | @4096 | bpb full val | tok/s | peak MB | grad ckpt |")
    w("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for sc in ("2M", "6M", "14M", "38M"):
        for k in ("alibi-hybrid-h6", "alibi-pure"):
            r = S.get(f"{sc}|{k}")
            if not r: continue
            d, l, h, hm = r["cfg"]
            w(f"| {sc} | {d}/{l}/{h}/{hm if k != 'alibi-pure' else '–'} | {NAME[k]} | {r['params']/1e6:.2f}M | {r['steps']:,} | {r['tokens']/1e6:.1f}M | {r['lr']} | "
              f"{r['val_bpb']:.4f} | {r['val_bpb_len2048']:.4f} | {r['val_bpb_len4096']:.4f} | {r['val_bpb_full']:.4f} | {r['tok_s']:,.0f} | {r['peak_mem_mb']:,.0f} | {r['grad_ckpt']} |")
    w("\nHybrid advantage (pure − hybrid, positive = hybrid better):\n")
    w("| Scale | Δ bpb @1024 (64 KB) | Δ bpb full val |"); w("|---|---|---|")
    for sc in ("2M", "6M", "14M", "38M"):
        a, b = S.get(f"{sc}|alibi-hybrid-h6"), S.get(f"{sc}|alibi-pure")
        if a and b: w(f"| {sc} | {b['val_bpb']-a['val_bpb']:+.4f} | {b['val_bpb_full']-a['val_bpb_full']:+.4f} |")
    w("\n**Finding:** the hybrid beats pure ALiBi at all four scales (1.8M → 38M) at matched tokens, by 0.021–0.036 bpb on the full "
      "validation set from 6M upward; the margin does not shrink with scale. The 2M gap (~0.28–0.30) is an outlier: at only 563 steps pure "
      "ALiBi has barely left the early plateau, so it mostly reflects the hybrid's faster early learning, not a large-model trend. "
      "Caveats: one seed per cell; at 38M the hybrid's B=16 step exceeded the 7 GB budget and ran with activation checkpointing while "
      "pure ALiBi did not, so the 38M tok/s column is **not** a like-for-like speed comparison (quality is unaffected by checkpointing).\n")
    sw = sorted(k for k in S if k.startswith("sweep|"))
    if sw:
        w("\n38M LR sweep (1,000 steps, seed 100, bpb @1024): " + "; ".join(f"{k.split('|')[2]} lr={k.split('|')[3]}: {S[k]['val_bpb']:.4f}" for k in sw) + "\n")

P = J("benchmark_results/paper/training_overhead_profile/gpu_overhead_profile.json")
if P:
    C = (1024, 2048, 4096, 8192, 16384)
    w("## 4. Computational overhead and memory vs. sequence length (Reviewer items 5, 8)\n")
    w("14.4M models, B=1, bf16, torch.compile, AdamW. Median of 10 timed steps after 3 warm-up steps. "
      "`train` = full step (fwd+bwd+clip+AdamW); `prefill` = no-grad forward over the sequence.\n")
    for mode, title in (("train_nockpt", "Training step, no activation checkpointing"), ("train_ckpt", "Training step, per-block activation checkpointing"), ("prefill", "Inference prefill (forward only)")):
        w(f"**{title}** (ms / peak MB):\n")
        w("| Model | " + " | ".join(f"T={c}" for c in C) + " |"); w("|---|" + "---|" * len(C))
        for k in K3:
            cells = []
            for c in C:
                r = P.get(f"{k}|{c}|{mode}", {})
                cells.append(f"{r['ms']:.1f} ms / {r['peak_mem_mb']:,.0f} MB" if r.get("status") == "ok" else r.get("status", "–"))
            w(f"| {NAME[k]} | " + " | ".join(cells) + " |")
        w("")
    w("**Speed ratio vs. hybrid** (other model's ms ÷ hybrid ms; > 1 = hybrid faster), training without checkpointing:\n")
    w("| Baseline | " + " | ".join(f"T={c}" for c in C) + " |"); w("|---|" + "---|" * len(C))
    for k in ("alibi-pure", "rope-gpt"):
        w(f"| {NAME[k]} | " + " | ".join(f"{P[f'{k}|{c}|train_nockpt']['ms']/P[f'alibi-hybrid-h6|{c}|train_nockpt']['ms']:.2f}×" for c in C) + " |")
    w("\n**Analytical cost** (per model; inference cache = per-sequence state that must be kept during autoregressive decoding, bf16):\n")
    T2 = ("1024", "4096", "16384", "65536")
    w("| Model | Params | Weights bf16 | Train state fp32+Adam | Attn layers | SheafDelta layers | " + " | ".join(f"Cache @{t}" for t in T2) + " |")
    w("|---|---|---|---|---|---|" + "---|" * len(T2))
    for k in K3:
        a = P[f"analytic|{k}"]
        w(f"| {NAME[k]} | {a['params']/1e6:.2f}M | {a['param_bytes_bf16']/1e6:.1f} MB | {a['train_state_bytes_fp32_adam']/1e6:.1f} MB | {a['attn_layers']} | "
          f"{a['sheafdelta_layers']} | " + " | ".join(f"{a['cache_bytes_bf16'][t]/1e6:.2f} MB" for t in T2) + " |")
    w("\nCache formula: attention layer = 2·T·D values (K and V); SheafDelta layer = H·d_h² (matrix memory, 6·64·64) + 16·D (dyadic ring "
      "buffer) = 30,720 values, independent of T. The hybrid's cache is 6 × 30,720 + 2 × 2·T·384, i.e. ≈ ¼ of a full-attention model's "
      "at every length (the remaining growth comes from its 2 attention layers).\n")
    w("**Findings:** (i) Against pure ALiBi (the architecture it is meant to improve on), the hybrid's training step is 8% slower at 1k but "
      "1.3×–2.8× faster from 2k to 16k (the gap widens with length), with 12–52% lower peak memory at every length; prefill is likewise "
      "10% slower at 1k and 2.0–2.7× faster from 2k to 16k. (ii) RoPE GPT on FlashAttention "
      "is the fastest model at every length. The hybrid's speed advantage is over ALiBi's explicit-bias attention, which cannot use the "
      "FlashAttention kernel, not over attention in general; RoPE GPT, however, cannot extrapolate (Section 2). (iii) With activation "
      "checkpointing, the hybrid's step is 2.6–3.7× slower than without, a much larger recompute penalty than the attention models pay "
      "(1.0–1.3×); at B=1 the fused delta-rule kernel's recompute cost dominates. Checkpointing is therefore only worth it for the hybrid "
      "when memory is the binding constraint.\n")
w("## Implementation notes\n")
w("- Scripts: `benchmarks/gpu_wikitext103_3seed.py`, `gpu_extrap_16k.py`, `gpu_scaling_law.py`, `gpu_overhead_profile.py`; shared code in "
  "`benchmarks/deepa_common.py`; data prep `benchmarks/data_prep/build_wikitext103_bytes.py`; this file from `benchmarks/summarize_phaseA.py`.")
w("- Model classes and the training recipe are copied from `gpu_splitwd_3seed.py` (param counts reproduce exactly: 14,436,348 / 14,260,608).")
w("- ALiBi beyond 4,096 tokens uses query-chunked attention (1,024-row blocks, bias built per block, K/V sliced to the causal prefix). "
  "It is mathematically identical to the dense-bias path (max abs diff 0.0 at T = 2,048 and 4,096) and needed because the dense fp32 "
  "[H,T,T] bias is 8.6 GB at T = 16,384. FlexAttention was not usable (torch 2.6 rejects head dim 48).")
w("- Single-seed rows (extrapolation, scaling) carry seed noise of about ±0.002–0.005 bpb (see Section 1 stds); differences below ~0.01 should not be over-interpreted.")
open("benchmark_results/PHASE_A_RESULTS.md", "w").write("\n".join(L) + "\n"); print("\n".join(L))
