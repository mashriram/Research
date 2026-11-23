"""Builds benchmark_results/PHASE_B_RESULTS.md from the Phase B result JSONs. Safe to run at any time: sections whose runs have not
finished yet are marked PENDING (or show partial rows, clearly labelled). Std = population std (ddof=0), as in Table 9.
Usage: python benchmarks/summarize_phaseB.py"""
import json, os, statistics as st
J = lambda p: json.load(open(p)) if os.path.exists(p) else {}
NAME = {"alibi-hybrid-h6": "ALiBi-DEEPA 3:1 (h6) [paper model]", "alibi-pure": "Pure ALiBi", "rope-gpt": "RoPE GPT",
        "rope-hybrid-h6": "RoPE-DEEPA 3:1 (h6) [diagnostic]", "sheafdelta-pure": "Pure SheafDelta (no attention)"}
out = []
P = out.append
def ms(v, d=4): return f"{st.mean(v):.{d}f} ± {st.pstdev(v):.{d}f}" if len(v) > 1 else (f"{v[0]:.{d}f} (1 seed)" if v else "—")

P("# Phase B results: deeper benchmarks for the reviewer response\n")
P("Hardware: single RTX 2000 Ada Laptop GPU (8 GB). Byte-level LM, vocab 256. bpb = bits per byte, lower is better. "
  "± = population std over seeds. Unless stated, models are the paper recipe (dim 384, 8 layers, 14.4M params, AdamW "
  "(0.9, 0.95), wd 0.05, lr 2e-3, warmup 100, cosine, B 16, ctx 1024, bf16, torch.compile). ALiBi attention uses an exact "
  "fused-kernel formulation (validated to within 4e-5 bpb of the dense-mask path). `full` = all 976 windows of the 1 MB "
  "validation slice (the 64-window `val_bpb` repo convention is optimistic; see PHASE_A_RESULTS.md).\n")

# ---------------- 1. domains ----------------
D = J("benchmark_results/paper/multidomain_3seed/gpu_domains_3seed.json")
DOMS = [("code", "Python source code"), ("hindi", "Hindi Wikipedia (Devanagari UTF-8)"), ("zh", "Chinese Wikipedia (UTF-8)"),
        ("dna", "DNA, hg38 chr20 (bits/base)"), ("image", "CIFAR-10 pixels, raster RGB (bits/dim)"), ("audio", "Speech, 16 kHz 8-bit mu-law (bits/sample)")]
P("## 1. Multi-domain / multi-modal language modelling (Reviewer item 4)\n")
P("Six non-enwik8 domains, each 40 MB train + 1 MB disjoint validation, identical byte-level pipeline and recipe, "
  "1,500 steps, seeds 0/1/2. Value = full-validation bpb.\n")
KS = ["alibi-hybrid-h6", "alibi-pure", "rope-gpt", "sheafdelta-pure"]
P("| Domain | " + " | ".join(NAME[k] for k in KS) + " | Winner (margin) |"); P("|---" * (len(KS) + 2) + "|")
wins = {}
for d, lab in DOMS:
    vals = {k: [D[f"{d}|{k}|seed{s}"]["val_bpb_full"] for s in range(3) if f"{d}|{k}|seed{s}" in D] for k in KS}
    done = {k: v for k, v in vals.items() if len(v) == 3}
    cells = [(("**" if done and k == min(done, key=lambda a: st.mean(done[a])) else "") + ms(vals[k]) +
              ("**" if done and k == min(done, key=lambda a: st.mean(done[a])) else "")) if vals[k] else "pending" for k in KS]
    if len(done) >= 2:
        o = sorted(done, key=lambda a: st.mean(done[a])); w = o[0]; wins[d] = w
        win = f"{NAME[w].split(' [')[0]} ({st.mean(done[o[1]]) - st.mean(done[w]):.3f})"
    else: win = "pending"
    P(f"| {lab} | " + " | ".join(cells) + f" | {win} |")
tok = {k: [D[a]["tok_s"] for a in D if a.split("|")[1:2] == [k]] for k in KS}
P("\nTraining throughput (mean over all domain runs, tokens/s): " + ", ".join(f"{NAME[k].split(' [')[0]} {st.mean(v)/1e3:.1f}k" for k, v in tok.items() if v) + ".\n")

P("### 1b. Is the image gap caused by DEEPA or by ALiBi? (same hybrid, RoPE instead of ALiBi in its 2 attention layers)\n")
P("| Domain | Model | bpb full @1024 | @2048 (2x) | @4096 (4x) |"); P("|---|---|---|---|---|")
for d in ("image", "code"):
    for k in ("alibi-hybrid-h6", "rope-hybrid-h6", "rope-gpt", "alibi-pure"):
        rs = [D[f"{d}|{k}|seed{s}"] for s in range(3) if f"{d}|{k}|seed{s}" in D]
        if rs: P(f"| {d} | {NAME[k]} | {ms([r['val_bpb_full'] for r in rs])} | {ms([r['val_bpb_len2048'] for r in rs])} | {ms([r['val_bpb_len4096'] for r in rs])} |")
P("\n**Reading:** on images, swapping ALiBi for RoPE inside the *same* DEEPA architecture recovers the whole gap and "
  "beats RoPE GPT, so the image weakness belongs to ALiBi's monotonic distance penalty (the pixel above is 96 bytes back), "
  "not to the DEEPA mixer. On code the two DEEPA variants tie, and RoPE-DEEPA degrades far less beyond the training length "
  "than RoPE GPT (only 2 of 8 layers carry positional encoding). The paper model keeps ALiBi for length extrapolation.\n")

# ---------------- 2. WikiText fairness ----------------
W = J("benchmark_results/paper/wikitext103_lr_and_horizon/gpu_wikitext103_fair.json")
P("## 2. WikiText-103: is pure ALiBi's 1,500-step lead real? (Reviewer item 4)\n")
P("(a) LR sweep, seed 0, 1,500 steps, full-validation bpb:\n")
P("| LR | " + " | ".join(NAME[k] for k in ("alibi-hybrid-h6", "alibi-pure")) + " |"); P("|---|---|---|")
for lr in (0.001, 0.002, 0.003):
    P(f"| {lr:g} | " + " | ".join(f"{W[f'sweep|{k}|{lr}']['val_bpb_full']:.4f}" if f"sweep|{k}|{lr}" in W else "pending" for k in ("alibi-hybrid-h6", "alibi-pure")) + " |")
P("\n(b) Each model at its best LR, trained to 5 tokens/param (4,406 steps), seeds 0 and 1:\n")
P("| Model | best LR | bpb full | @1024 (64 KB) | @2048 | @4096 |"); P("|---|---|---|---|---|---|")
for k in ("alibi-hybrid-h6", "alibi-pure"):
    rs = [W[f"long|{k}|seed{s}"] for s in (0, 1) if f"long|{k}|seed{s}" in W]
    if rs: P(f"| {NAME[k]} | {W.get(f'best_lr|{k}', '?')} | {ms([r['val_bpb_full'] for r in rs])} | {ms([r['val_bpb'] for r in rs])} | {ms([r['val_bpb_len2048'] for r in rs])} | {ms([r['val_bpb_len4096'] for r in rs])} |")
P("\n**Reading:** both models' best LR is the paper's 2e-3 (no tuning artefact). At 1,500 steps pure ALiBi led by 0.012 bpb; "
  "trained to 5 tokens/param the hybrid leads at every evaluation length on both seeds. The Phase A WikiText finding "
  "(PHASE_A_RESULTS.md section 1) was a short-horizon effect, consistent with the 18,000-step enwik8 result.\n")

# ---------------- 3. extrapolation ----------------
E = J("benchmark_results/paper/length_extrapolation_16k/gpu_extrap_16k.json"); L = (1024, 2048, 4096, 8192, 16384)
P("## 3. Length extrapolation to 16x, 3 seeds (Reviewer item 7)\n")
P("enwik8, trained at ctx 1024 (1,500 steps), evaluated zero-shot on the same 512 KB at every length.\n")
P("| Model | " + " | ".join(f"L={l}" for l in L) + " |"); P("|---" * (len(L) + 1) + "|")
for k in ("alibi-hybrid-h6", "alibi-pure", "rope-gpt"):
    rs = [E[f"{k}|seed{s}"] for s in range(3) if "val_bpb_len16384" in E.get(f"{k}|seed{s}", {})]
    P(f"| {NAME[k]} ({len(rs)} seeds) | " + " | ".join(ms([r[f'val_bpb_len{l}'] for r in rs], 3) for l in L) + " |")
P("\n**Reading:** both ALiBi models improve monotonically to 16x the training length on every seed; RoPE GPT collapses "
  "(~3.5x worse at 16k). The hybrid has the lowest mean at every length, but beyond 2k its lead over pure ALiBi "
  "(0.006-0.009) is within seed spread (seed 2 favours pure ALiBi), so claim *stable* extrapolation for both ALiBi models, "
  "with the hybrid doing it at 1/4 of the KV cache (section 4).\n")

# ---------------- 4. serving / overhead ----------------
P("## 4. Inference, serving capacity, training cost (Reviewer items 5, 8)\n")
DF, DFF = J("benchmark_results/paper/decode_and_serving/gpu_decode_fair.json"), J("benchmark_results/paper/decode_and_serving/gpu_decode_fair_flashalibi.json")
OV, OVF = J("benchmark_results/paper/training_overhead_profile/gpu_overhead_profile.json"), J("benchmark_results/paper/training_overhead_profile/gpu_overhead_profile_flashalibi.json")
P("Decode (CUDA-graph, preallocated KV cache for every model; ms per generated token, best ALiBi implementation per cell):\n")
P("| Context | Batch | ALiBi-DEEPA (h6) | Pure ALiBi | RoPE GPT | hybrid cache MB | GPT cache MB |"); P("|---|---|---|---|---|---|---|")
def best(k, key):
    c = [x[key] for x in (DF.get(k), DFF.get(k)) if x and x.get("status") == "ok"]
    return min(c, key=lambda r: r["ms_per_step"]) if c else None
for T in (1024, 4096, 16384, 65536):
    for Bn in (1, 32):
        r = {k: best(f"{k}|{T}|B{Bn}|graph", None) if False else None for k in ()}
        row = []
        for k in ("alibi-hybrid-h6", "alibi-pure", "rope-gpt"):
            c = [x.get(f"{k}|{T}|B{Bn}|graph") for x in (DF, DFF)]; c = [x for x in c if x and x.get("status") == "ok"]
            row.append(min(c, key=lambda q: q["ms_per_step"]) if c else None)
        if not any(row): continue
        f = lambda q: f"{q['ms_per_step']:.2f}" if q else "OOM/—"
        P(f"| {T} | {Bn} | {f(row[0])} | {f(row[1])} | {f(row[2])} | {row[0]['cache_mb']:.0f} | {row[2]['cache_mb']:.0f} |" if row[0] and row[2] else
          f"| {T} | {Bn} | {f(row[0])} | {f(row[1])} | {f(row[2])} | — | — |")
P("\nServing capacity in 8 GB (max concurrent sequences, aggregate decode tokens/s):\n")
P("| Context | ALiBi-DEEPA (h6) | Pure ALiBi | RoPE GPT |"); P("|---|---|---|---|")
for T in (4096, 16384, 65536):
    cells = []
    for k in ("alibi-hybrid-h6", "alibi-pure", "rope-gpt"):
        c = [x.get(f"capacity|{k}|{T}") for x in (DF, DFF)]; c = [x for x in c if x and x.get("max_batch")]
        q = max(c, key=lambda z: z["tok_s"]) if c else None
        cells.append(f"{q['max_batch']} seq, {q['tok_s']:.0f} tok/s" if q else "—")
    P(f"| {T} | " + " | ".join(cells) + " |")
P("\nTraining step (B=1, no checkpointing) and prefill, ms (best ALiBi implementation):\n")
P("| Length | DEEPA train | RoPE GPT train | DEEPA prefill | RoPE GPT prefill | DEEPA train peak MB | GPT train peak MB |"); P("|---|---|---|---|---|---|---|")
for T in (1024, 2048, 4096, 8192, 16384):
    def g(k, w):
        c = [x.get(f"{k}|{T}|{w}") for x in (OV, OVF)]; c = [x for x in c if x and x.get("status") == "ok"]
        return min(c, key=lambda q: q["ms"]) if c else None
    h, r, hp, rp = g("alibi-hybrid-h6", "train_nockpt"), g("rope-gpt", "train_nockpt"), g("alibi-hybrid-h6", "prefill"), g("rope-gpt", "prefill")
    f = lambda q, key="ms": f"{q[key]:.1f}" if q else "—"
    P(f"| {T} | {f(h)} | {f(r)} | {f(hp)} | {f(rp)} | {f(h, 'peak_mem_mb')} | {f(r, 'peak_mem_mb')} |")
P("\n**Reading:** the hybrid's KV cache is ~1/4 of full attention at every length, giving ~4x the concurrent sequences "
  "and 3.7-4.8x the aggregate decode throughput in 8 GB, and faster single-stream decode from 4k up (2.5x at 64k). Known "
  "costs: at <=1k context, training/prefill is ~1.4x slower than RoPE GPT (many small SheafDelta kernels), domain-run "
  "throughput is 26% below pure ALiBi, and 16k training memory is ~14% higher.\n")

# ---------------- 5. LRA ----------------
A = J("benchmark_results/paper/long_range_arena/gpu_lra_subset.json")
P("## 5. Long Range Arena subset (Reviewer item 6/7)\n")
if not A: P("PENDING.\n")
else:
    P("dim 256, 4 layers, 5,000 steps, lr 1e-3, B 32, seeds 0/1; test accuracy.\n")
    P("| Task | " + " | ".join(NAME[k] for k in ("alibi-hybrid-h6", "alibi-pure", "rope-gpt")) + " |"); P("|---|---|---|---|")
    for t, lab in (("listops", "ListOps (<=2k)"), ("text", "IMDb bytes (4k)"), ("image", "sCIFAR grayscale (1k)")):
        P(f"| {lab} | " + " | ".join(ms([A[f'{t}|{k}|seed{s}']['test_acc'] for s in (0, 1) if f'{t}|{k}|seed{s}' in A], 3) if any(f'{t}|{k}|seed{s}' in A for s in (0, 1)) else "pending" for k in ("alibi-hybrid-h6", "alibi-pure", "rope-gpt")) + " |")
    P("")

# ---------------- 5b. iso wall-clock ----------------
I, C18 = J("benchmark_results/paper/enwik8_equal_wallclock/gpu_chinchilla_15m_13300.json"), J("benchmark_results/paper/enwik8_18k_convergence/gpu_chinchilla_15m_18000.json")
P("## 5b. Equal wall-clock training vs Pure ALiBi (Reviewer item 5)\n")
if I:
    P("With the folded ALiBi kernel, Pure ALiBi trains 1.353x faster than DEEPA at the paper recipe (117.0k vs 86.5k tok/s, "
      "B 16, ctx 1024), so Pure ALiBi's 18,000-step budget buys DEEPA 13,300 steps (cosine schedule fitted to 13,300). enwik8, "
      "64 KB slice (same metric as the 18k table), 3 seeds.\n")
    P("| Model | Steps | Train s (measured) | bpb @1024 | per seed | @2048 | @4096 |"); P("|---|---|---|---|---|---|---|")
    rows = [("ALiBi-DEEPA 3:1 (h6), equal wall-clock", I, "alibi-hybrid-h6", 13300), ("Pure ALiBi", C18, "alibi-pure", 18000),
            ("Pure ALiBi, equal steps (control)", I, "alibi-pure", 13300), ("ALiBi-DEEPA 3:1 (h6), 18k steps", C18, "alibi-hybrid-h6", 18000)]
    for lab, src, k, st_ in rows:
        rs = [src[f"{k}|seed{s}"] for s in range(3) if f"{k}|seed{s}" in src]
        if rs: P(f"| {lab} | {st_:,} | {st.mean(r['train_s'] for r in rs):,.0f} | {ms([r['val_bpb'] for r in rs])} | " + "/".join(f"{r['val_bpb']:.4f}" for r in rs) +
                 f" | {ms([r['val_bpb_len2048'] for r in rs])} | {ms([r['val_bpb_len4096'] for r in rs])} |")
    both = [k for k in range(3) if f"alibi-pure|seed{k}" in I and f"alibi-hybrid-h6|seed{k}" in I]
    pure13 = [I[f"alibi-pure|seed{k}"]["val_bpb"] for k in both]
    deepa13 = [I[f"alibi-hybrid-h6|seed{k}"]["val_bpb"] for k in both]
    ctl = (f" At equal steps (13,300 each) DEEPA leads Pure ALiBi on {sum(d < q for d, q in zip(deepa13, pure13))}/{len(pure13)} seeds "
           f"({st.mean(deepa13):.4f} vs {st.mean(pure13):.4f}); both models also score better at 13.3k than at 18k, consistent with "
           f"over-repetition of the 40 MB pool at 18k, so the comparison is not explained by that effect." if pure13 else
           " The Pure ALiBi 13.3k equal-steps control is pending.")
    P("\n**Reading:** in equal (folded-kernel) wall-clock, DEEPA beats Pure ALiBi on 3/3 seeds at every length, and on the measured "
      "hardware it also used 22% less time." + ctl + "\n")
else: P("PENDING.\n")

# ---------------- 5c. negative-eigenvalue control ----------------
PB = J("benchmark_results/paper/parity_negative_eigenvalue_control/gpu_parity_beta_control.json")
P("## 5c. Causal control: parity with negative eigenvalues removed (beta in (0,1))\n")
if PB:
    P("Same converged protocol as section 6 (curriculum 8->64, up to 20k steps, LR grid, 3 seeds), delta-rule step size capped at "
      "beta_max = 1 (eigenvalues in [0,1]) instead of the paper's 2 ([-1,1]). Identical parameter counts.\n")
    P("| Model | acc @64 (train length) | acc @256 (4x) | steps to solve |"); P("|---|---|---|---|")
    for k, lab in (("alibi-hybrid-h6-beta1", "ALiBi-DEEPA, beta in (0,1)"), ("sheafdelta-pure-beta1", "Pure SheafDelta, beta in (0,1)")):
        rs = [PB[f"parity|{k}|seed{s}"] for s in range(3) if f"parity|{k}|seed{s}" in PB]
        if rs: P(f"| {lab} | " + "/".join(f"{r['acc_T64']*100:.1f}" for r in rs) + " | " + "/".join(f"{r['acc_T256']*100:.1f}" for r in rs) + " | " + "/".join(str(r["steps_to_solve"]) for r in rs) + " |")
    SPp = J("benchmark_results/paper/synthetic_converged/gpu_synthetic_proper.json")
    for k, lab in (("alibi-hybrid-h6", "ALiBi-DEEPA, beta in (0,2) [paper]"), ("sheafdelta-pure", "Pure SheafDelta, beta in (0,2)")):
        rs = [SPp[f"parity|{k}|seed{s}"] for s in range(3) if f"parity|{k}|seed{s}" in SPp]
        P(f"| {lab} | " + ("/".join(f"{r['acc_T64']*100:.1f}" for r in rs) + " | " + "/".join(f"{r['acc_T256']*100:.1f}" for r in rs) + " | " + "/".join(str(r["steps_to_solve"]) for r in rs) if rs else "pending (converged suite) | | ") + " |")
    P("\n**Reading:** without negative eigenvalues both mixers still fit parity at the training length (with ~4x more steps and the "
      "curriculum) but fall to ~64% at 4x length; with beta in (0,2) the 3,000-step screening already gave 95-100% at 4x length "
      "(section 6). Negative eigenvalues are what make the learned solution length-general.\n")
else: P("PENDING.\n")

# ---------------- 6b. LR robustness ----------------
SPg = J("benchmark_results/paper/synthetic_converged/gpu_synthetic_proper.json")
if any(k.startswith("grid|") for k in SPg):
    P("## 6b. Learning-rate robustness for the seed-sensitive tasks\n")
    P("Every LR x every seed (seed 0 = sweep run, seeds 1-2 = grid). For each model the LR is chosen by the 3-seed mean accuracy "
      "at the training length (the same rule for every model). Cells: mean accuracy at train length / at 4x length (copy) "
      "and seeds solved (>=99%).\n")
    for task in ("copy_paper", "copy", "mqar64_T512"):
        T = {"copy_paper": 128, "copy": 128, "mqar64_T512": 512}[task]
        P(f"**{task}**\n"); P("| Model | LR 3e-4 | LR 1e-3 | LR 3e-3 | best LR (3-seed mean) |"); P("|---|---|---|---|---|")
        for k in ("alibi-hybrid-h6", "alibi-pure", "rope-gpt", "sheafdelta-pure"):
            cells, best = [], None
            for lr in (0.0003, 0.001, 0.003):
                rs = [SPg.get(f"sweep|{task}|{k}|{lr}")] + [SPg.get(f"grid|{task}|{k}|{lr}|seed{sd}") for sd in (1, 2)]
                rs = [r for r in rs if r]
                if len(rs) < 3: cells.append("—"); continue
                m = st.mean(r[f"acc_T{T}"] for r in rs); solved = sum(1 for r in rs if r["steps_to_solve"])
                ext = (" / " + f"{st.mean(r['acc_T512'] for r in rs)*100:.0f}" if "copy" in task else "")
                cells.append(f"{m*100:.1f}{ext}% ({solved}/3)")
                if best is None or m > best[1]: best = (lr, m, solved)
            P(f"| {NAME[k]} | " + " | ".join(cells) + f" | {f'{best[0]:g} ({best[2]}/3 solved)' if best else '—'} |")
        P("")

# ---------------- 6. synthetic ----------------
S, SP = J("benchmark_results/paper/synthetic_screening_3k/gpu_synthetic_suite.json"), J("benchmark_results/paper/synthetic_converged/gpu_synthetic_proper.json")
SK = ("alibi-hybrid-h6", "alibi-pure", "rope-gpt", "sheafdelta-pure")
P("## 6. Synthetic capability suite (mechanism evidence)\n")
if SP:
    P("Converged protocol: up to 20k steps, LR grid {3e-4,1e-3,3e-3} on seed 0, best LR x 3 seeds, early stop at >=99% "
      "(two consecutive evals). Cells: accuracy per seed at each test length; [steps to solve].\n")
    tasks = sorted({k.split("|")[1] for k in SP if k.startswith("sweep|")}, key=lambda t: ["copy_paper", "copy", "mqar8_T64", "mqar32_T512", "mqar64_T512", "parity", "s3", "s5", "selcopy"].index(t) if t in ["copy_paper", "copy", "mqar8_T64", "mqar32_T512", "mqar64_T512", "parity", "s3", "s5", "selcopy"] else 99)
    P("| Task | " + " | ".join(NAME[k] for k in SK) + " |"); P("|---" * 5 + "|")
    for t in tasks:
        cells = []
        for k in ("alibi-hybrid-h6", "alibi-pure", "rope-gpt", "sheafdelta-pure"):
            rs = [SP[f"{t}|{k}|seed{s}"] for s in range(3) if f"{t}|{k}|seed{s}" in SP]
            if not rs: cells.append("pending"); continue
            accs = [a for a in rs[0] if a.startswith("acc_")]
            cells.append("; ".join(f"{a[4:]}: " + "/".join(f"{r[a]*100:.0f}" for r in rs) for a in accs) +
                         " [" + "/".join(str(r["steps_to_solve"] or "x") for r in rs) + "]")
        P(f"| {t} | " + " | ".join(cells) + " |")
    P("")
else: P("Converged protocol: PENDING (queued; expected Sunday afternoon/evening).\n")
P("Screening run (3,000 steps, dim 128, 4 layers; NOT converged -- MQAR/S5/copy numbers are superseded by the converged "
  "protocol above): accuracy % per seed.\n")
P("| Task | " + " | ".join(NAME[k] for k in SK) + " |"); P("|---" * 5 + "|")
for t in ("mqar32", "mqar64", "s5", "parity", "copy", "selcopy"):
    cells = []
    for k in SK:
        rs = [S.get(f"{t}|{k}|seed{s}") for s in range(3)]; rs = [r for r in rs if r]
        accs = [a for a in (rs[0] if rs else {}) if a.startswith("acc_")]
        cells.append("; ".join(f"{a[4:]}: " + "/".join(f"{r[a]*100:.0f}" for r in rs) for a in accs) if rs else "—")
    P(f"| {t} | " + " | ".join(cells) + " |")

open("benchmark_results/PHASE_B_RESULTS.md", "w").write("\n".join(out) + "\n"); print("wrote benchmark_results/PHASE_B_RESULTS.md", len(out), "lines")
