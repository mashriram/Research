# hsnlm – first benchmark (CPU, 8 cores, tiny-shakespeare char-level, ~0.81M params, 800 steps x 32 x 128 tok, 1 seed)

| model | train rule | val bits/char (lower=better) | train tok/s | gen tok/s (B=16) |
|---|---|---|---|---|
| GPT (pre-LN Transformer) | backprop | 2.592 | 11.7k | 3.1k |
| **HSN** (Householder sheaf + gated recurrence) | backprop | **2.202** | 7.2k | 3.0k |
| HSN | local / DAD-style (detached blocks) | 2.372 | 7.2k | (noisy) |
| GPT | local / DAD-style | 2.618 | 14.8k | (noisy) |

Caveats: very early training (3.3 epochs), single seed, 1M-char dataset, generation speeds noisy (shared CPU),
context only 128 so O(N^2) vs O(N) not yet exercised, activation memory not yet measured.
Verified by tests/: recurrent step == parallel forward, Householder maps isometric, causality.

## Credit-assignment comparison (same HSN, 0.82M params, 800 steps, CPU, 1 seed) — experiments/credit.py
| rule | cross-block gradient chain? | val bits/char | wall time |
|---|---|---|---|
| backprop | yes | 2.202 | 536 s |
| greedy local next-token loss (DAD-style) | no | 2.372 | ~458 s |
| layer-wise teacher-target regression (oracle teacher hidden states, shared embed/head) | no | 2.316 | 589 s |
| DFA (random feedback matrices) | no | 3.164 | 463 s |
Takeaways: no alternative reduced wall-clock (each block still runs its own forward+local backward); DFA fails on this architecture;
rich per-layer targets beat the greedy label loss but still trail backprop by 0.11 bit (error compounds across the chain; not yet tried: closed-loop fine-tune / last-block KL).

## RBlockFast measurements (CPU, eager PyTorch, 1 seed) — experiments/{fixes,rbf,recall,decode_o1,fwdgrad,ngram}.py
- Reversible stack: gradients equal autograd to ~1e-7; saved activations flat 1.29 MB for L=4..32 (autograd 19->148 MB); time +0..30%.
- Forward-gradient (JVP) alignment with true gradient = sqrt(k/d): 0.002 (k=1) .. 0.095 (k=512) on a 59k-param block.
- 5-gram counting, 4 s, no gradients: 2.41 bits/char (vs 2.20 sheaf-backprop, 2.59 our Transformer).
- Associative recall (2 layers, 8 pairs, 1500 steps): attention .370 | diag-sheaf .265 | Householder-delta .466 (old variant; undertrained, 1 seed).
- Thin Householder transport == explicit reflections (tested), chunked delta rule == sequential loop (tested, atol 1e-4), step==parallel (tested).
- TRAINING SPEED (D=128, 4 layers, 8192 tok/step, N=512): Transformer 765 ms, HSN-fast 2190 ms (2.9x SLOWER), activations 378 vs 1059 MB.
  Cause: eager-mode memory-bound elementwise ops + python chunk loop; needs fused GPU kernels (fla chunk_gated_delta_rule / Triton). NOT solved here.
- DECODE per layer (D=1024, batch 1): attention 3.6 -> 56.6 ms and KV 2 -> 134 MB as context 256 -> 16k; sheaf-delta ~4-7 ms flat, state 0.33 MB.
