# Benchmarks

All scripts run from the `hsnlm/` root (`python benchmarks/<script>.py`). Results go to `benchmark_results/<group>/<benchmark>/`; each folder has a README with the exact command and the results.

## Benchmark scripts

| script | benchmark(s) | results folder |
|---|---|---|
| [`cpu_suite.py`](cpu_suite.py) | Original CPU benchmark suite (HSN vs GPT) | [cpu_suite](../benchmark_results/development/cpu_suite/README.md) |
| [`data_prep/build_domains.py`](data_prep/build_domains.py) | Six extra domains (code, Hindi, Chinese, DNA, images, audio), 3 seeds | [multidomain_3seed](../benchmark_results/paper/multidomain_3seed/README.md) |
| [`data_prep/build_wikitext103_bytes.py`](data_prep/build_wikitext103_bytes.py) | WikiText-103 (byte-level), 1,500 steps, 3 seeds | [wikitext103_3seed](../benchmark_results/paper/wikitext103_3seed/README.md) |
| [`gpu_alibi_hybrid.py`](gpu_alibi_hybrid.py) | ALiBi hybrids: 3:1 vs 4:1 dilution | [alibi_hybrid_ratio](../benchmark_results/development/alibi_hybrid_ratio/README.md) |
| [`gpu_alibi_hybrid_ratio.py`](gpu_alibi_hybrid_ratio.py) | ALiBi hybrids: 3:1 vs 4:1 dilution | [alibi_hybrid_ratio](../benchmark_results/development/alibi_hybrid_ratio/README.md) |
| [`gpu_blockdecode.py`](gpu_blockdecode.py) | Block / self-speculative decoding primitive | [block_decode](../benchmark_results/development/block_decode/README.md) |
| [`gpu_chinchilla_15m.py`](gpu_chinchilla_15m.py) | Chinchilla-horizon convergence on enwik8 (18,000 steps, 3 seeds); Equal wall-clock and equal-steps control (13,300 steps, 3 seeds) | [enwik8_18k_convergence](../benchmark_results/paper/enwik8_18k_convergence/README.md), [enwik8_equal_wallclock](../benchmark_results/paper/enwik8_equal_wallclock/README.md) |
| [`gpu_credit.py`](gpu_credit.py) | Backprop-free credit assignment (DAD, DFA, targets, JVP) | [credit_assignment](../benchmark_results/development/credit_assignment/README.md) |
| [`gpu_crossover.py`](gpu_crossover.py) | Fused Triton delta-rule kernel speed crossover | [fused_kernel_crossover](../benchmark_results/development/fused_kernel_crossover/README.md) |
| [`gpu_decode.py`](gpu_decode.py) | Early decode benchmarks (torch.cat KV cache; per-layer cost) | [decode_early](../benchmark_results/development/decode_early/README.md) |
| [`gpu_decode_fair.py`](gpu_decode_fair.py) | CUDA-graph decoding latency and 8 GB serving capacity | [decode_and_serving](../benchmark_results/paper/decode_and_serving/README.md) |
| [`gpu_decode_hybrid_recheck.py`](gpu_decode_hybrid_recheck.py) | Early decode benchmarks (torch.cat KV cache; per-layer cost) | [decode_early](../benchmark_results/development/decode_early/README.md) |
| [`gpu_domains_3seed.py`](gpu_domains_3seed.py) | Six extra domains (code, Hindi, Chinese, DNA, images, audio), 3 seeds | [multidomain_3seed](../benchmark_results/paper/multidomain_3seed/README.md) |
| [`gpu_extrap_16k.py`](gpu_extrap_16k.py) | Zero-shot length extrapolation 1k -> 16k (3 seeds) | [length_extrapolation_16k](../benchmark_results/paper/length_extrapolation_16k/README.md) |
| [`gpu_isoquality_compiled.py`](gpu_isoquality_compiled.py) | Iso-quality: tokens / wall-clock to reach RoPE GPT's final loss | [isoquality_vs_rope](../benchmark_results/paper/isoquality_vs_rope/README.md) |
| [`gpu_lowrank_mlp.py`](gpu_lowrank_mlp.py) | Low-rank MLP factorisation | [lowrank_mlp](../benchmark_results/development/lowrank_mlp/README.md) |
| [`gpu_lra_subset.py`](gpu_lra_subset.py) | Long Range Arena subset (ListOps, IMDb bytes, sCIFAR), 2 seeds | [long_range_arena](../benchmark_results/paper/long_range_arena/README.md) |
| [`gpu_multiseed_compiled.py`](gpu_multiseed_compiled.py) | Multi-seed compiled comparison (hybrids vs GPT) | [multiseed_compiled](../benchmark_results/paper/multiseed_compiled/README.md) |
| [`gpu_overhead_profile.py`](gpu_overhead_profile.py) | Training step / prefill latency and memory vs length (1k-16k) | [training_overhead_profile](../benchmark_results/paper/training_overhead_profile/README.md) |
| [`gpu_posenc.py`](gpu_posenc.py) | NoPE / ALiBi / RoPE positional-encoding comparison | [positional_encoding](../benchmark_results/development/positional_encoding/README.md) |
| [`gpu_posenc_recheck.py`](gpu_posenc_recheck.py) | NoPE / ALiBi / RoPE positional-encoding comparison | [positional_encoding](../benchmark_results/development/positional_encoding/README.md) |
| [`gpu_ratio.py`](gpu_ratio.py) | Hybrid attention-ratio ablation (RoPE hybrids) | [attention_ratio](../benchmark_results/development/attention_ratio/README.md) |
| [`gpu_reconcile.py`](gpu_reconcile.py) | Speed and quality measured together (isolated processes) | [reconcile_speed_quality](../benchmark_results/development/reconcile_speed_quality/README.md) |
| [`gpu_reconcile_clean.py`](gpu_reconcile_clean.py) | Speed and quality measured together (isolated processes) | [reconcile_speed_quality](../benchmark_results/development/reconcile_speed_quality/README.md) |
| [`gpu_reversible.py`](gpu_reversible.py) | Reversible-block memory/speed trade-off | [reversible_blocks](../benchmark_results/development/reversible_blocks/README.md) |
| [`gpu_scale.py`](gpu_scale.py) | Largest single-GPU LM run (pre-checkpointing) | [gpu_scale](../benchmark_results/development/gpu_scale/README.md) |
| [`gpu_scale_ckpt.py`](gpu_scale_ckpt.py) | Scaling past 14M with gradient checkpointing (7:1 and 3:1) | [gpu_scale_ckpt](../benchmark_results/development/gpu_scale_ckpt/README.md) |
| [`gpu_scale_ckpt_31.py`](gpu_scale_ckpt_31.py) | Scaling past 14M with gradient checkpointing (7:1 and 3:1) | [gpu_scale_ckpt](../benchmark_results/development/gpu_scale_ckpt/README.md) |
| [`gpu_scaling_law.py`](gpu_scaling_law.py) | Parameter scaling 2M -> 38M at 5 tokens/param | [scaling_2m_to_38m](../benchmark_results/paper/scaling_2m_to_38m/README.md) |
| [`gpu_speed.py`](gpu_speed.py) | Training speed/memory vs context: Transformer vs HSN (eager) | [speed_vs_length](../benchmark_results/development/speed_vs_length/README.md) |
| [`gpu_splitwd_3seed.py`](gpu_splitwd_3seed.py) | Uniform vs split weight decay (1,500 steps, 3 seeds) | [enwik8_split_weight_decay](../benchmark_results/paper/enwik8_split_weight_decay/README.md) |
| [`gpu_suite.py`](gpu_suite.py) | GPU replica of the CPU suite | [gpu_suite](../benchmark_results/development/gpu_suite/README.md) |
| [`gpu_swa_alibi_hybrid.py`](gpu_swa_alibi_hybrid.py) | Sliding-window ALiBi hybrid (W=512) | [sliding_window_alibi](../benchmark_results/development/sliding_window_alibi/README.md) |
| [`gpu_synthetic_proper.py`](gpu_synthetic_proper.py) | Converged synthetic suite (MQAR, parity, S3, S5, copy, selective copy); Causal control: parity with beta capped to (0,1) | [synthetic_converged](../benchmark_results/paper/synthetic_converged/README.md), [parity_negative_eigenvalue_control](../benchmark_results/paper/parity_negative_eigenvalue_control/README.md) |
| [`gpu_synthetic_suite.py`](gpu_synthetic_suite.py) | Synthetic suite screening run (3,000 steps) | [synthetic_screening_3k](../benchmark_results/paper/synthetic_screening_3k/README.md) |
| [`gpu_v2_ablation.py`](gpu_v2_ablation.py) | Sheaf-Delta mixer ablations (h6, decay init, post-norm, token shift) | [mixer_ablation_v2](../benchmark_results/paper/mixer_ablation_v2/README.md) |
| [`gpu_wikitext103_3seed.py`](gpu_wikitext103_3seed.py) | WikiText-103 (byte-level), 1,500 steps, 3 seeds | [wikitext103_3seed](../benchmark_results/paper/wikitext103_3seed/README.md) |
| [`gpu_wikitext103_fair.py`](gpu_wikitext103_fair.py) | WikiText-103: LR sweep + 5 tokens/param (4,406 steps) | [wikitext103_lr_and_horizon](../benchmark_results/paper/wikitext103_lr_and_horizon/README.md) |
| [`gpu_yarn.py`](gpu_yarn.py) | YaRN RoPE rescaling for GPT and the RoPE hybrid | [yarn_rope_scaling](../benchmark_results/development/yarn_rope_scaling/README.md) |
| [`gpu_yarn_hybrid.py`](gpu_yarn_hybrid.py) | YaRN RoPE rescaling for GPT and the RoPE hybrid | [yarn_rope_scaling](../benchmark_results/development/yarn_rope_scaling/README.md) |
| [`gpu_yarn_recheck.py`](gpu_yarn_recheck.py) | YaRN RoPE rescaling for GPT and the RoPE hybrid | [yarn_rope_scaling](../benchmark_results/development/yarn_rope_scaling/README.md) |

## Support scripts

| script | purpose |
|---|---|
| [`deepa_common.py`](deepa_common.py) | Shared model builders (paper model, Pure ALiBi, RoPE GPT, Pure SheafDelta, RoPE-DEEPA), training loop and eval for all paper benchmarks. |
| [`summarize_phaseA.py`](summarize_phaseA.py) | Rebuilds benchmark_results/PHASE_A_RESULTS.md from the Phase A JSONs. |
| [`summarize_phaseB.py`](summarize_phaseB.py) | Rebuilds benchmark_results/PHASE_B_RESULTS.md (all reviewer-response tables) from the JSONs. |
| [`build_results_index.py`](build_results_index.py) | Regenerates every benchmark_results/**/README.md and both index files from catalog.py. |
| [`catalog.py`](catalog.py) | This catalog. |
| [`run_bench.py`](run_bench.py) | Early generic benchmark runner (development). |
| [`rerun_failed_seeds.py`](rerun_failed_seeds.py) | NOT RUN. Would rerun only DEEPA's failed synthetic seeds at another LR (one-sided); superseded by the fair PROPER_FULLGRID pass in synthetic_converged. |
| [`data_prep/build_domains.py`](data_prep/build_domains.py) | Downloads and builds the six 41 MB domain files (code, Hindi, Chinese, DNA, CIFAR-10 pixels, mu-law audio). |
| [`data_prep/build_wikitext103_bytes.py`](data_prep/build_wikitext103_bytes.py) | Builds data/wikitext103_raw.bytes (40 MB train + 1 MB validation). |

## GPU queues

`queues/` holds the shell queues that ran these benchmarks back to back on the single 8 GB GPU (kept as an exact record of run order). Their logs are in `benchmark_results/_queue_logs/`.

## Reproducing the paper end to end

```bash
uv venv && uv pip install -e .        # torch 2.6 + cu124, flash-linear-attention 0.6, triton 3.2 (see pyproject)
python benchmarks/data_prep/build_wikitext103_bytes.py && python -I benchmarks/data_prep/build_domains.py .
bash benchmarks/queues/queue10_phaseA.sh   # then queue12, 13, 14, 15 (each waits for the previous one)
python benchmarks/summarize_phaseB.py      # rebuild the aggregated tables
python benchmarks/build_results_index.py   # rebuild these READMEs
```