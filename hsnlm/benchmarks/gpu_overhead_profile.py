"""Reviewer items 5 & 8 (training speed, memory, computational overhead): measured and analytical cost of the 14.4M
models (width 384) as the sequence length grows, ctx in {1024, 2048, 4096, 8192, 16384}.
Measured, per (kind, ctx), B=1, bf16 autocast, torch.compile, AdamW:
  train  : full training step (fwd + bwd + clip + AdamW) with per-block activation checkpointing -> ms/step, tok/s,
           peak VRAM.  Also the same step WITHOUT checkpointing where it fits (OOM recorded otherwise).
  prefill: no-grad forward over the whole sequence -> ms, tok/s, peak VRAM.
  3 warm-up steps, then the median of 10 timed steps (cuda-synchronized).
Analytical, per kind: parameter count/bytes, training-state bytes (fp32 params+grads+2 Adam moments = 16 B/param),
and the per-sequence inference cache at each ctx in bf16: attention layer = K and V, 2*T*D values; SheafDelta layer =
matrix memory H*dh*dh + dyadic ring buffer 16*D values, independent of T.
Note: the model architecture is held fixed at the paper's ctx=1024 config at every length (no ctx-dependent params).
ALiBi at T > 4096 uses the exact query-chunked path (deepa_common.py)."""
import sys, os, json, time, statistics, torch, torch.nn.functional as F; sys.path.insert(0, ".")
from benchmarks.deepa_common import build, log, VOCAB
assert torch.cuda.is_available()
torch.backends.cudnn.benchmark = True
KINDS = ("alibi-hybrid-h6", "alibi-pure", "rope-gpt")
CTXS = (1024, 2048, 4096, 8192, 16384)
WARM, TIMED = 3, 10
import benchmarks.deepa_common as C
FLASH = "--flash-alibi" in sys.argv  # ALiBi layers use the exact fused-kernel path (deepa_common.alibi_attend_flash)
if FLASH: C.ALIBI_IMPL = "flash"; KINDS = ("alibi-hybrid-h6", "alibi-pure")
OUT = "benchmark_results/paper/training_overhead_profile/gpu_overhead_profile_flashalibi.json" if FLASH else "benchmark_results/paper/training_overhead_profile/gpu_overhead_profile.json"; os.makedirs("benchmark_results", exist_ok=True)
R = json.load(open(OUT)) if os.path.exists(OUT) else {}
def save(): json.dump(R, open(OUT, "w"), indent=1)

def timed(fn):
    for _ in range(WARM): fn()
    torch.cuda.synchronize(); ts = []
    for _ in range(TIMED):
        t0 = time.perf_counter(); fn(); torch.cuda.synchronize(); ts.append(time.perf_counter() - t0)
    return statistics.median(ts)

def measure(kind, T, mode):
    torch._dynamo.reset(); torch.cuda.empty_cache(); torch.manual_seed(0)
    m0 = build(kind, ckpt=(mode == "train_ckpt")); m = torch.compile(m0)
    g = torch.Generator().manual_seed(0)
    x = torch.randint(0, VOCAB, (1, T), generator=g).cuda(); y = torch.randint(0, VOCAB, (1, T), generator=g).cuda()
    try:
        if mode == "prefill":
            m0.eval()
            def fn():
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16): m(x)
        else:
            opt = torch.optim.AdamW(m0.parameters(), lr=1e-4, betas=(0.9, 0.95), weight_decay=0.05)
            def fn():
                opt.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = F.cross_entropy(m(x).reshape(-1, VOCAB), y.reshape(-1))
                loss.backward(); torch.nn.utils.clip_grad_norm_(m0.parameters(), 1.0); opt.step()
        fn(); torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        s = timed(fn)
        r = dict(ms=s * 1e3, tok_s=T / s, peak_mem_mb=torch.cuda.max_memory_allocated() / 1e6, status="ok")
    except torch.OutOfMemoryError:
        r = dict(status="OOM")
    except Exception as e:  # record, don't abort the sweep
        r = dict(status=f"ERROR {type(e).__name__}: {str(e)[:300]}")
    del m, m0; torch.cuda.empty_cache(); return r

def analytic(kind, m):
    c = m.c; D = c.dim; pat = c.pattern or ("a" * c.layers)
    n_attn, n_d = pat.count("a"), pat.count("d"); H = c.mixer_heads or c.heads; dh = D // H
    d_state = H * dh * dh + 16 * D
    a = dict(params=m.n_params(), param_bytes_bf16=2 * m.n_params(), param_bytes_fp32=4 * m.n_params(),
             train_state_bytes_fp32_adam=16 * m.n_params(), attn_layers=n_attn, sheafdelta_layers=n_d,
             sheafdelta_state_values_per_layer=d_state if n_d else 0, cache_bytes_bf16={})
    for T in CTXS + (32768, 65536):
        a["cache_bytes_bf16"][str(T)] = 2 * (n_attn * 2 * T * D + n_d * d_state)
    return a

for kind in KINDS:
    if f"analytic|{kind}" not in R: R[f"analytic|{kind}"] = analytic(kind, build(kind)); save()
    for T in CTXS:
        for mode in ("train_ckpt", "train_nockpt", "prefill"):
            k = f"{kind}|{T}|{mode}"
            if k in R: continue
            R[k] = measure(kind, T, mode); save(); log(k, R[k])
log("ALL DONE")
