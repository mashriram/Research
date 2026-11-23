"""Single source of truth for every benchmark in this repo: what it tests, which script runs it, which result files it
owns, where it appears in the paper, and whether its numbers are current.

Used by ``benchmarks/build_results_index.py`` to (re)generate ``benchmark_results/**/README.md`` and the two index
files. Add an entry here whenever a new benchmark is created.

Fields
  group   : "paper"        -> numbers used in the journal paper (DEEPA_Springer.tex) / reviewer response
            "development"  -> earlier exploration that led to the paper model (kept for provenance)
  status  : "current" | "superseded by <name>" | "negative result" | "not run"
  files   : result files (in this benchmark's folder) -- json = machine-readable results, log = stdout of the run
  run     : exact command(s), from the hsnlm/ root, to reproduce
"""

PAPER_MODEL = "ALiBi-DEEPA 3:1 (h6): pattern dddaddda, dim 384, 8 layers, 8 attn heads, 6 mixer heads, 14.44M params"

CATALOG = [
    # =========================================================== paper: language modelling quality
    dict(name="enwik8_18k_convergence", group="paper", title="Chinchilla-horizon convergence on enwik8 (18,000 steps, 3 seeds)",
         paper="Table 4 (tab:chinchilla)", status="current", scripts=["gpu_chinchilla_15m.py"],
         files=["gpu_chinchilla_15m_18000.json"],
         run=["python benchmarks/gpu_chinchilla_15m.py 18000 0,1,2"],
         notes="DEEPA beats Pure ALiBi on 3/3 seeds (0.7794 vs 0.7863 bpb, 64 KB slice). 18,000 steps = 7.4 epochs over the "
               "40 MB training pool (sampled with replacement). ALiBi here uses the dense-mask path (tok/s are pre-folding)."),
    dict(name="enwik8_equal_wallclock", group="paper", title="Equal wall-clock and equal-steps control (13,300 steps, 3 seeds)",
         paper="Table 4 (13.3k rows)", status="current", scripts=["gpu_chinchilla_15m.py"],
         files=["gpu_chinchilla_15m_13300.json", "gpu_isotime_13300.log", "gpu_isostep_pure_13300.log"],
         run=["KINDS=alibi-hybrid-h6 python benchmarks/gpu_chinchilla_15m.py 13300 0,1,2   # DEEPA, equal wall-clock",
              "KINDS=alibi-pure python benchmarks/gpu_chinchilla_15m.py 13300 0,1,2        # Pure ALiBi, equal steps"],
         notes="Folded-kernel Pure ALiBi trains 1.353x faster than DEEPA (117.0k vs 86.5k tok/s), so its 18,000-step budget "
               "buys DEEPA 13,300 steps. The Pure ALiBi 13,300-step rows are the equal-steps control for the "
               "over-repetition hypothesis (both models score better at 13.3k than at 18k)."),
    dict(name="enwik8_split_weight_decay", group="paper", title="Uniform vs split weight decay (1,500 steps, 3 seeds)",
         paper="Table 4 (1.5k rows)", status="current", scripts=["gpu_splitwd_3seed.py"], files=["gpu_splitwd_3seed.json"],
         run=["python benchmarks/gpu_splitwd_3seed.py"]),
    dict(name="mixer_ablation_v2", group="paper", title="Sheaf-Delta mixer ablations (h6, decay init, post-norm, token shift)",
         paper="Table 8 (tab:ablation)", status="current", scripts=["gpu_v2_ablation.py"], files=["gpu_v2_ablation.json"],
         run=["python benchmarks/gpu_v2_ablation.py"],
         notes="3 seeds x 8 variants, 1,500 steps, enwik8 64 KB slice. h6 (6 mixer heads, d_h 64) adopted."),
    dict(name="multiseed_compiled", group="paper", title="Multi-seed compiled comparison (hybrids vs GPT)",
         paper="Sec. 5.1 (iso-quality context)", status="current", scripts=["gpu_multiseed_compiled.py"],
         files=["gpu_multiseed_compiled.json"], run=["python benchmarks/gpu_multiseed_compiled.py"]),
    dict(name="isoquality_vs_rope", group="paper", title="Iso-quality: tokens / wall-clock to reach RoPE GPT's final loss",
         paper="Sec. 5.1 (42.4% fewer tokens, 30.3% less time)", status="current", scripts=["gpu_isoquality_compiled.py"],
         files=["gpu_isoquality_compiled.json", "gpu_isoquality_compiled.log"], run=["python benchmarks/gpu_isoquality_compiled.py"]),
    dict(name="wikitext103_3seed", group="paper", title="WikiText-103 (byte-level), 1,500 steps, 3 seeds",
         paper="Table 6 (WikiText-103 1.5k row)", status="current", scripts=["gpu_wikitext103_3seed.py", "data_prep/build_wikitext103_bytes.py"],
         files=["gpu_wikitext103_3seed.json", "gpu_wikitext103_3seed.log"],
         run=["python benchmarks/data_prep/build_wikitext103_bytes.py", "python benchmarks/gpu_wikitext103_3seed.py"],
         notes="At 1,500 steps Pure ALiBi leads by 0.012 bpb; see wikitext103_lr_and_horizon for the tuned / longer run."),
    dict(name="wikitext103_lr_and_horizon", group="paper", title="WikiText-103: LR sweep + 5 tokens/param (4,406 steps)",
         paper="Table 6 (WikiText-103 4.4k row)", status="current", scripts=["gpu_wikitext103_fair.py"],
         files=["gpu_wikitext103_fair.json", "gpu_wikitext103_fair.log"], run=["python benchmarks/gpu_wikitext103_fair.py"],
         notes="Both models' best LR is 2e-3; at 4,406 steps DEEPA leads on both seeds (1.4250 vs 1.4404 full-val bpb)."),
    dict(name="multidomain_3seed", group="paper", title="Six extra domains (code, Hindi, Chinese, DNA, images, audio), 3 seeds",
         paper="Table 6 (tab:domains), Fig. 3", status="current", scripts=["gpu_domains_3seed.py", "data_prep/build_domains.py"],
         files=["gpu_domains_3seed.json", "gpu_domains_3seed.log", "gpu_domains_ropehybrid.log", "gpu_domains_sheafpure.log"],
         run=["python -I benchmarks/data_prep/build_domains.py .",
              "python benchmarks/gpu_domains_3seed.py                                   # DEEPA, Pure ALiBi, RoPE GPT",
              "python benchmarks/gpu_domains_3seed.py --kinds=sheafdelta-pure            # Pure SheafDelta ablation",
              "python benchmarks/gpu_domains_3seed.py image code --kinds=rope-hybrid-h6  # RoPE-DEEPA diagnostic"],
         notes="All four model families plus the RoPE-DEEPA diagnostic share one JSON (keys domain|model|seed)."),
    dict(name="length_extrapolation_16k", group="paper", title="Zero-shot length extrapolation 1k -> 16k (3 seeds)",
         paper="Table 5 (tab:extrap), Fig. 2a", status="current", scripts=["gpu_extrap_16k.py"],
         files=["gpu_extrap_16k.json", "gpu_extrap_16k.log", "gpu_extrap_16k_seeds12.log"],
         run=["python benchmarks/gpu_extrap_16k.py 0 1 2"],
         notes="Every length is scored on the same 512 KB of validation text. Checkpoints saved in checkpoints/extrap16k/."),
    dict(name="scaling_2m_to_38m", group="paper", title="Parameter scaling 2M -> 38M at 5 tokens/param",
         paper="Table 5 (tab:scaling), Fig. 2b", status="current", scripts=["gpu_scaling_law.py"],
         files=["gpu_scaling_law.json", "gpu_scaling_law.log"], run=["python benchmarks/gpu_scaling_law.py"],
         notes="Seed 0 only. At 38M DEEPA used activation checkpointing (its tok/s there is not like-for-like)."),
    # =========================================================== paper: systems
    dict(name="training_overhead_profile", group="paper", title="Training step / prefill latency and memory vs length (1k-16k)",
         paper="Table 7 (tab:train_speed), Fig. 4a, Table 3", status="current", scripts=["gpu_overhead_profile.py"],
         files=["gpu_overhead_profile.json", "gpu_overhead_profile.log", "gpu_overhead_profile_flashalibi.json", "gpu_overhead_profile_flashalibi.log"],
         run=["python benchmarks/gpu_overhead_profile.py                # dense-mask ALiBi",
              "python benchmarks/gpu_overhead_profile.py --flash-alibi  # folded (FlashAttention) ALiBi"],
         notes="The paper reports the faster implementation per cell; *_flashalibi files are the folded-kernel ALiBi runs."),
    dict(name="decode_and_serving", group="paper", title="CUDA-graph decoding latency and 8 GB serving capacity",
         paper="Table 7b (tab:decode_serve), Fig. 4b", status="current", scripts=["gpu_decode_fair.py"],
         files=["gpu_decode_fair.json", "gpu_decode_fair.log", "gpu_decode_fair_flashalibi.json", "gpu_decode_fair_flashalibi.log"],
         run=["python benchmarks/gpu_decode_fair.py", "python benchmarks/gpu_decode_fair.py --flash-alibi"],
         notes="Every model uses a preallocated KV cache + CUDA graphs. DEEPA B=32 numbers use the dense-mask path, B=1 the "
               "folded kernel (best per regime). Includes forward-vs-step correctness checks."),
    # =========================================================== paper: capabilities
    dict(name="long_range_arena", group="paper", title="Long Range Arena subset (ListOps, IMDb bytes, sCIFAR), 2 seeds",
         paper="Table 9 (tab:lra)", status="current", scripts=["gpu_lra_subset.py"],
         files=["gpu_lra_subset.json", "gpu_lra_subset.log", "gpu_lra_subset_resume.log"], run=["python benchmarks/gpu_lra_subset.py"],
         notes="The first pass hit OOM at the 4k-byte IMDb task; the resume pass (gradient accumulation 4x8, same effective "
               "batch 32) completed IMDb and sCIFAR. ListOps is at chance for every model."),
    dict(name="synthetic_converged", group="paper", title="Converged synthetic suite (MQAR, parity, S3, S5, copy, selective copy)",
         paper="Table 10 (tab:synth)", status="current", scripts=["gpu_synthetic_proper.py"],
         files=["gpu_synthetic_proper.json", "gpu_synthetic_proper.log", "gpu_synthetic_grid.log"],
         run=["python benchmarks/gpu_synthetic_proper.py                                          # LR sweep on seed 0 + 3 seeds",
              "PROPER_FULLGRID=1 python benchmarks/gpu_synthetic_proper.py copy_paper copy mqar64_T512  # every LR x every seed"],
         notes="Up to 20k steps, early stop at >=99% (two consecutive evals). dim 128 / 4 layers except copy_paper (paper "
               "model). grid|... keys = LR-robustness pass so the LR can be chosen by 3-seed mean for every model."),
    dict(name="parity_negative_eigenvalue_control", group="paper", title="Causal control: parity with beta capped to (0,1)",
         paper="Table 9b (parity control)", status="current", scripts=["gpu_synthetic_proper.py"],
         files=["gpu_parity_beta_control.json", "gpu_parity_beta_control.log"],
         run=["PROPER_ARCHS=alibi-hybrid-h6-beta1,sheafdelta-pure-beta1 PROPER_OUT=benchmark_results/paper/parity_negative_eigenvalue_control/gpu_parity_beta_control.json python benchmarks/gpu_synthetic_proper.py parity"],
         notes="Same protocol as synthetic_converged; only the delta-rule step-size cap changes (no negative eigenvalues)."),
    dict(name="synthetic_screening_3k", group="paper", title="Synthetic suite screening run (3,000 steps)", paper="superseded",
         status="superseded by synthetic_converged", scripts=["gpu_synthetic_suite.py"],
         files=["gpu_synthetic_suite.json", "gpu_synthetic_suite.log"], run=["python benchmarks/gpu_synthetic_suite.py"],
         notes="Under-trained: MQAR / S5 / copy numbers here are NOT valid conclusions; kept for transparency."),
    # =========================================================== development history
    dict(name="cpu_suite", group="development", title="Original CPU benchmark suite (HSN vs GPT)", status="superseded",
         scripts=["cpu_suite.py"], files=["cpu_suite.json", "cpu_suite.log", "BENCH_CPU.md"], run=["python benchmarks/cpu_suite.py"]),
    dict(name="gpu_suite", group="development", title="GPU replica of the CPU suite", status="superseded",
         scripts=["gpu_suite.py"], files=["gpu_suite.json", "gpu_suite.log"], run=["python benchmarks/gpu_suite.py"]),
    dict(name="gpu_scale", group="development", title="Largest single-GPU LM run (pre-checkpointing)", status="superseded",
         scripts=["gpu_scale.py"], files=["gpu_scale.json", "gpu_scale.log"], run=["python benchmarks/gpu_scale.py"]),
    dict(name="gpu_scale_ckpt", group="development", title="Scaling past 14M with gradient checkpointing (7:1 and 3:1)",
         status="superseded by scaling_2m_to_38m", scripts=["gpu_scale_ckpt.py", "gpu_scale_ckpt_31.py"],
         files=["gpu_scale_ckpt.json", "gpu_scale_ckpt.log", "gpu_scale_ckpt_31.json", "gpu_scale_ckpt_31.log"],
         run=["python benchmarks/gpu_scale_ckpt.py", "python benchmarks/gpu_scale_ckpt_31.py"],
         notes="Source of the '206M params fit in 5.3 GB' memory statement."),
    dict(name="speed_vs_length", group="development", title="Training speed/memory vs context: Transformer vs HSN (eager)",
         status="superseded by training_overhead_profile", scripts=["gpu_speed.py"], files=["gpu_speed.json", "gpu_speed.log"],
         run=["python benchmarks/gpu_speed.py"]),
    dict(name="fused_kernel_crossover", group="development", title="Fused Triton delta-rule kernel speed crossover",
         status="superseded by training_overhead_profile", scripts=["gpu_crossover.py"], files=["gpu_crossover.json", "gpu_crossover.log"],
         run=["python benchmarks/gpu_crossover.py"]),
    dict(name="reconcile_speed_quality", group="development", title="Speed and quality measured together (isolated processes)",
         status="superseded", scripts=["gpu_reconcile.py", "gpu_reconcile_clean.py"],
         files=["gpu_reconcile.json", "gpu_reconcile.log", "gpu_reconcile_clean_gpt.json", "gpu_reconcile_clean_hsn-fused.json",
                "gpu_reconcile_clean_hybrid.json", "gpu_reconcile_clean_hsn.log"],
         run=["python benchmarks/gpu_reconcile.py", "python benchmarks/gpu_reconcile_clean.py <arch>"]),
    dict(name="attention_ratio", group="development", title="Hybrid attention-ratio ablation (RoPE hybrids)", status="superseded",
         scripts=["gpu_ratio.py"], files=["gpu_ratio.json", "gpu_ratio.log"], run=["python benchmarks/gpu_ratio.py"]),
    dict(name="positional_encoding", group="development", title="NoPE / ALiBi / RoPE positional-encoding comparison",
         status="current (motivates ALiBi)", scripts=["gpu_posenc.py", "gpu_posenc_recheck.py"],
         files=["gpu_posenc.json", "gpu_posenc.log", "gpu_posenc_recheck.json", "gpu_posenc_recheck.log"],
         run=["python benchmarks/gpu_posenc.py", "python benchmarks/gpu_posenc_recheck.py"],
         notes="Source of the NoPE (2.344 bpb) negative result cited in the ablation section."),
    dict(name="alibi_hybrid_ratio", group="development", title="ALiBi hybrids: 3:1 vs 4:1 dilution", status="current (negative result for 4:1)",
         scripts=["gpu_alibi_hybrid.py", "gpu_alibi_hybrid_ratio.py"], files=["gpu_alibi_hybrid.log", "gpu_alibi_hybrid_4to1.json", "gpu_alibi_hybrid_4to1.log"],
         run=["python benchmarks/gpu_alibi_hybrid.py", "python benchmarks/gpu_alibi_hybrid_ratio.py"],
         notes="gpu_alibi_hybrid.json was produced by a run with a residual bug and is archived (see _archive/)."),
    dict(name="sliding_window_alibi", group="development", title="Sliding-window ALiBi hybrid (W=512)", status="negative result",
         scripts=["gpu_swa_alibi_hybrid.py"], files=["gpu_swa_alibi_hybrid.json", "gpu_swa_alibi_hybrid.log"],
         run=["python benchmarks/gpu_swa_alibi_hybrid.py"], notes="Source of the SWA 1.061 bpb negative result."),
    dict(name="yarn_rope_scaling", group="development", title="YaRN RoPE rescaling for GPT and the RoPE hybrid", status="superseded",
         scripts=["gpu_yarn.py", "gpu_yarn_recheck.py", "gpu_yarn_hybrid.py"],
         files=["gpu_yarn.log", "gpu_yarn_recheck.json", "gpu_yarn_recheck.log", "gpu_yarn_hybrid.json"],
         run=["python benchmarks/gpu_yarn_recheck.py", "python benchmarks/gpu_yarn_hybrid.py"]),
    dict(name="lowrank_mlp", group="development", title="Low-rank MLP factorisation", status="negative result",
         scripts=["gpu_lowrank_mlp.py"], files=["gpu_lowrank_mlp.json", "gpu_lowrank_mlp.log"], run=["python benchmarks/gpu_lowrank_mlp.py"]),
    dict(name="credit_assignment", group="development", title="Backprop-free credit assignment (DAD, DFA, targets, JVP)",
         status="negative result", scripts=["gpu_credit.py"], files=["gpu_credit.json", "gpu_credit.log"], run=["python benchmarks/gpu_credit.py"]),
    dict(name="block_decode", group="development", title="Block / self-speculative decoding primitive", status="exploratory",
         scripts=["gpu_blockdecode.py"], files=["gpu_blockdecode.json"], run=["python benchmarks/gpu_blockdecode.py"]),
    dict(name="decode_early", group="development", title="Early decode benchmarks (torch.cat KV cache; per-layer cost)",
         status="superseded by decode_and_serving", scripts=["gpu_decode.py", "gpu_decode_hybrid_recheck.py"],
         files=["gpu_decode_hybrid_recheck.json"], run=["python benchmarks/gpu_decode_hybrid_recheck.py"],
         notes="The GPT baseline here grew its KV cache with torch.cat (unfair); use decode_and_serving instead."),
    dict(name="reversible_blocks", group="development", title="Reversible-block memory/speed trade-off", status="exploratory",
         scripts=["gpu_reversible.py"], files=[], run=["python benchmarks/gpu_reversible.py"]),
]

# Scripts that are not benchmarks of their own
SUPPORT = {
    "deepa_common.py": "Shared model builders (paper model, Pure ALiBi, RoPE GPT, Pure SheafDelta, RoPE-DEEPA), training loop and eval for all paper benchmarks.",
    "summarize_phaseA.py": "Rebuilds benchmark_results/PHASE_A_RESULTS.md from the Phase A JSONs.",
    "summarize_phaseB.py": "Rebuilds benchmark_results/PHASE_B_RESULTS.md (all reviewer-response tables) from the JSONs.",
    "build_results_index.py": "Regenerates every benchmark_results/**/README.md and both index files from catalog.py.",
    "catalog.py": "This catalog.",
    "run_bench.py": "Early generic benchmark runner (development).",
    "rerun_failed_seeds.py": "NOT RUN. Would rerun only DEEPA's failed synthetic seeds at another LR (one-sided); superseded by the fair PROPER_FULLGRID pass in synthetic_converged.",
    "data_prep/build_domains.py": "Downloads and builds the six 41 MB domain files (code, Hindi, Chinese, DNA, CIFAR-10 pixels, mu-law audio).",
    "data_prep/build_wikitext103_bytes.py": "Builds data/wikitext103_raw.bytes (40 MB train + 1 MB validation).",
}

ARCHIVE = {  # result files that must not be used (kept for provenance only)
    "gpu_alibi_hybrid.json.BROKEN_residualbug": "alibi_hybrid_ratio", "gpu_alibi_hybrid_4to1.json.BROKEN_residualbug": "alibi_hybrid_ratio",
    "gpu_posenc.json.BROKEN_residualbug": "positional_encoding", "gpu_posenc_recheck.json.BROKEN_residualbug": "positional_encoding",
    "gpu_swa_alibi_hybrid.json.BROKEN_residualbug": "sliding_window_alibi", "gpu_yarn.json.BROKEN_residualbug": "yarn_rope_scaling",
    "gpu_yarn_recheck.json.BROKEN_residualbug": "yarn_rope_scaling", "gpu_isoquality_compiled.json.OLD_noextrap": "isoquality_vs_rope",
}
SUMMARIES = ["PHASE_A_RESULTS.md", "PHASE_B_RESULTS.md"]


def folder(entry) -> str:
    return f"benchmark_results/{entry['group']}/{entry['name']}"


def file_map() -> dict:
    """result file name -> new relative path (from the hsnlm root)."""
    m = {}
    for e in CATALOG:
        for f in e["files"]:
            m[f] = f"{folder(e)}/{f}"
    for f in ARCHIVE:
        m[f] = f"benchmark_results/_archive/{f}"
    for f in SUMMARIES:
        m[f] = f"benchmark_results/{f}"
    return m
