# Benchmark results

Every benchmark behind the DEEPA paper, one folder each, with its raw logs, machine-readable JSON and a README (what it tests, how to reproduce, results). Paper model: ALiBi-DEEPA 3:1 (h6): pattern dddaddda, dim 384, 8 layers, 8 attn heads, 6 mixer heads, 14.44M params.

* Aggregated paper tables: [PHASE_B_RESULTS.md](PHASE_B_RESULTS.md) (reviewer response, all new tables) and [PHASE_A_RESULTS.md](PHASE_A_RESULTS.md) (first round).
* Scripts and how they map to these folders: [../benchmarks/README.md](../benchmarks/README.md).
* `_archive/`: result files from runs with known bugs; never use them. `_queue_logs/`: orchestration logs of the GPU queues.

## Paper benchmarks (numbers used in the manuscript)

| benchmark | paper location | status |
|---|---|---|
| [Chinchilla-horizon convergence on enwik8 (18,000 steps, 3 seeds)](paper/enwik8_18k_convergence/README.md) | Table 4 (tab:chinchilla) | current |
| [Equal wall-clock and equal-steps control (13,300 steps, 3 seeds)](paper/enwik8_equal_wallclock/README.md) | Table 4 (13.3k rows) | current |
| [Uniform vs split weight decay (1,500 steps, 3 seeds)](paper/enwik8_split_weight_decay/README.md) | Table 4 (1.5k rows) | current |
| [Sheaf-Delta mixer ablations (h6, decay init, post-norm, token shift)](paper/mixer_ablation_v2/README.md) | Table 8 (tab:ablation) | current |
| [Multi-seed compiled comparison (hybrids vs GPT)](paper/multiseed_compiled/README.md) | Sec. 5.1 (iso-quality context) | current |
| [Iso-quality: tokens / wall-clock to reach RoPE GPT's final loss](paper/isoquality_vs_rope/README.md) | Sec. 5.1 (42.4% fewer tokens, 30.3% less time) | current |
| [WikiText-103 (byte-level), 1,500 steps, 3 seeds](paper/wikitext103_3seed/README.md) | Table 6 (WikiText-103 1.5k row) | current |
| [WikiText-103: LR sweep + 5 tokens/param (4,406 steps)](paper/wikitext103_lr_and_horizon/README.md) | Table 6 (WikiText-103 4.4k row) | current |
| [Six extra domains (code, Hindi, Chinese, DNA, images, audio), 3 seeds](paper/multidomain_3seed/README.md) | Table 6 (tab:domains), Fig. 3 | current |
| [Zero-shot length extrapolation 1k -> 16k (3 seeds)](paper/length_extrapolation_16k/README.md) | Table 5 (tab:extrap), Fig. 2a | current |
| [Parameter scaling 2M -> 38M at 5 tokens/param](paper/scaling_2m_to_38m/README.md) | Table 5 (tab:scaling), Fig. 2b | current |
| [Training step / prefill latency and memory vs length (1k-16k)](paper/training_overhead_profile/README.md) | Table 7 (tab:train_speed), Fig. 4a, Table 3 | current |
| [CUDA-graph decoding latency and 8 GB serving capacity](paper/decode_and_serving/README.md) | Table 7b (tab:decode_serve), Fig. 4b | current |
| [Long Range Arena subset (ListOps, IMDb bytes, sCIFAR), 2 seeds](paper/long_range_arena/README.md) | Table 9 (tab:lra) | current |
| [Converged synthetic suite (MQAR, parity, S3, S5, copy, selective copy)](paper/synthetic_converged/README.md) | Table 10 (tab:synth) | current |
| [Causal control: parity with beta capped to (0,1)](paper/parity_negative_eigenvalue_control/README.md) | Table 9b (parity control) | current |
| [Synthetic suite screening run (3,000 steps)](paper/synthetic_screening_3k/README.md) | superseded | superseded by synthetic_converged |

## Development history (how the paper model was found)

| benchmark | paper location | status |
|---|---|---|
| [Original CPU benchmark suite (HSN vs GPT)](development/cpu_suite/README.md) | - | superseded |
| [GPU replica of the CPU suite](development/gpu_suite/README.md) | - | superseded |
| [Largest single-GPU LM run (pre-checkpointing)](development/gpu_scale/README.md) | - | superseded |
| [Scaling past 14M with gradient checkpointing (7:1 and 3:1)](development/gpu_scale_ckpt/README.md) | - | superseded by scaling_2m_to_38m |
| [Training speed/memory vs context: Transformer vs HSN (eager)](development/speed_vs_length/README.md) | - | superseded by training_overhead_profile |
| [Fused Triton delta-rule kernel speed crossover](development/fused_kernel_crossover/README.md) | - | superseded by training_overhead_profile |
| [Speed and quality measured together (isolated processes)](development/reconcile_speed_quality/README.md) | - | superseded |
| [Hybrid attention-ratio ablation (RoPE hybrids)](development/attention_ratio/README.md) | - | superseded |
| [NoPE / ALiBi / RoPE positional-encoding comparison](development/positional_encoding/README.md) | - | current (motivates ALiBi) |
| [ALiBi hybrids: 3:1 vs 4:1 dilution](development/alibi_hybrid_ratio/README.md) | - | current (negative result for 4:1) |
| [Sliding-window ALiBi hybrid (W=512)](development/sliding_window_alibi/README.md) | - | negative result |
| [YaRN RoPE rescaling for GPT and the RoPE hybrid](development/yarn_rope_scaling/README.md) | - | superseded |
| [Low-rank MLP factorisation](development/lowrank_mlp/README.md) | - | negative result |
| [Backprop-free credit assignment (DAD, DFA, targets, JVP)](development/credit_assignment/README.md) | - | negative result |
| [Block / self-speculative decoding primitive](development/block_decode/README.md) | - | exploratory |
| [Early decode benchmarks (torch.cat KV cache; per-layer cost)](development/decode_early/README.md) | - | superseded by decode_and_serving |
| [Reversible-block memory/speed trade-off](development/reversible_blocks/README.md) | - | exploratory |
