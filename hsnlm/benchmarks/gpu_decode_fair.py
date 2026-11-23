"""B1 -- fair autoregressive-decode benchmark (Reviewer items 5, 8). Replaces the torch.cat-grown KV cache of
gpu_decode.py / gpu_decode_hybrid_recheck.py (which copies the whole cache every step and so penalises attention)
with what a real inference engine does:
  * attention layers (RoPE and ALiBi) use a PREALLOCATED bf16 KV cache [B,h,Tmax,dh]; one in-place write per step
  * SheafDelta layers carry their fixed state (matrix memory S [B,H,dh,dh] + 16-step dyadic buffer), updated in place
  * the whole one-token step (embed -> 8 blocks -> head -> greedy argmax) is captured as ONE CUDA graph and replayed,
    for all three models alike, so Python/kernel-launch overhead does not decide the comparison
  * models in bf16 (inference dtype); the 14.4M paper configs from deepa_common.build()
Measured at each (context T, batch B): ms per decode step (median of 100 graph replays), aggregate tok/s = B/step,
peak VRAM. Also: "serving capacity" = the largest batch (concurrent sequences) that fits in this 8 GB GPU at T, and its
throughput. The cache is filled with random values up to position T-1 (timing does not depend on cache contents);
correctness of the decode path is checked separately: step-by-step logits vs. the parallel forward() on real text."""
import sys, os, json, time, statistics, math, torch, torch.nn.functional as F; sys.path.insert(0, ".")
from hsnlm import ByteData
from hsnlm.models import Attn, HBlock, rope
from hsnlm.fastmixer import hh_coef
from benchmarks.deepa_common import build, AlibiAttn, alibi_slopes, alibi_aug_qk, PAD_DH, log
assert torch.cuda.is_available()
KINDS = ("alibi-hybrid-h6", "alibi-pure") if "--flash-alibi" in sys.argv else ("alibi-hybrid-h6", "alibi-pure", "rope-gpt")
CTXS = (1024, 4096, 16384, 65536)
BATCHES = (1, 8, 32, 128)
FLASH = "--flash-alibi" in sys.argv  # exact ALiBi folded into q.k (deepa_common.alibi_aug_qk): no mask -> fused SDPA kernel
OUT = "benchmark_results/paper/decode_and_serving/gpu_decode_fair_flashalibi.json" if FLASH else "benchmark_results/paper/decode_and_serving/gpu_decode_fair.json"; os.makedirs("benchmark_results", exist_ok=True)
R = json.load(open(OUT)) if os.path.exists(OUT) else {}
def save(): json.dump(R, open(OUT, "w"), indent=1)

def sheaf_step(mx, x, S, buf, idx):
    """FusedSheafDeltaMixer.step() (default branch: no token_shift/postnorm), with the dyadic gather index passed in as a
    device tensor so the step is CUDA-graph capturable. Returns (out, S_new, buf_new). Checked equal to mx.step() below."""
    B = x.shape[0]; q, k, v, b, a = mx._proj(x)
    Sk = torch.einsum("bhvk,bhk->bhv", S, k)
    S = a[..., None, None] * (S - b[..., None, None] * Sk[..., None] * k[..., None, :]) + b[..., None, None] * v[..., None] * k[..., None, :]
    o_raw = torch.einsum("bhvk,bhk->bhv", S, q); vf = v.reshape(B, mx.D); V, G = mx._VG()
    g = torch.sigmoid(mx.edge(x)).view(B, mx.K, mx.H)
    u = buf.index_select(1, idx)
    c = hh_coef(torch.einsum("bkd,kmd->bkm", u, V), G)
    tr = (u - 2 * torch.einsum("bkm,kmd->bkd", c, V)).view(B, mx.K, mx.H, mx.dh)
    acc = (g[..., None] * tr).sum(1) / mx.K ** 0.5
    buf = torch.cat([vf[:, None], buf[:, :-1]], 1)
    y = mx.onorm(o_raw * mx.dh ** -0.5).reshape(B, mx.D) + acc.reshape(B, mx.D)
    return mx.out(y * F.silu(mx.g(x))), S, buf

class Decoder:
    def __init__(s, m, B, Tmax, dtype=torch.bfloat16, fill_random=True):
        s.m, s.B, s.Tmax = m, B, Tmax; dev = "cuda"; D = m.c.dim
        s.tok = torch.zeros(B, dtype=torch.long, device=dev); s.st = []; s.bias = {}
        for blk in m.blocks:
            if isinstance(blk, HBlock):
                mx = blk.mix
                S = torch.zeros(B, mx.H, mx.dh, mx.dh, device=dev, dtype=dtype); buf = torch.zeros(B, mx.offsets[-1], D, device=dev, dtype=dtype)
                if fill_random: S.normal_(0, 0.02); buf.normal_(0, 0.02)
                s.st.append(("d", S, buf, torch.tensor([d - 1 for d in mx.offsets], device=dev)))
            else:
                h = blk.mix.h; dh = D // h; P = PAD_DH if (FLASH and isinstance(blk.mix, AlibiAttn)) else dh
                K = torch.empty(B, h, Tmax, P, device=dev, dtype=dtype); V = torch.empty_like(K)
                if fill_random: K.normal_(); V.normal_()
                else: K.zero_(); V.zero_()
                s.st.append(("a", K, V, None))
        s.state_bytes = sum(t.numel() * t.element_size() for e in s.st for t in e[1:3])
    def step(s, t):  # decode the token at position t (cache holds positions < t)
        m = s.m; h = m.embed(s.tok)
        for blk, (kind, A, Bf, idx) in zip(m.blocks, s.st):
            x = blk.n1(h)
            if kind == "d":
                y, S2, b2 = sheaf_step(blk.mix, x, A, Bf, idx); A.copy_(S2); Bf.copy_(b2)
            else:
                at = blk.mix; Bn, D = x.shape; H = at.h; dh = D // H
                q, k, v = at.qkv(x).view(Bn, 3, H, 1, dh).unbind(1)
                if isinstance(at, Attn) and at.rope:
                    pos = torch.full((1,), t, device=x.device); q, k = rope(q, pos).to(x.dtype), rope(k, pos).to(x.dtype)  # rope() upcasts to fp32
                if FLASH and isinstance(at, AlibiAttn):
                    if "pos" not in s.bias: s.bias["pos"] = torch.zeros(1, dtype=torch.long, device=x.device)
                    s.bias["pos"].fill_(t)
                    q2, k2 = alibi_aug_qk(q, k, s.bias["pos"]); v2 = torch.cat([v, v.new_zeros(*v.shape[:-1], PAD_DH - dh)], -1)
                    A[:, :, t:t + 1] = k2.to(A.dtype); Bf[:, :, t:t + 1] = v2.to(Bf.dtype)
                    y = F.scaled_dot_product_attention(q2, A[:, :, :t + 1], Bf[:, :, :t + 1], scale=dh ** -0.5)[..., :dh]
                    y = at.o(y.reshape(Bn, D)); h = h + y; h = h + blk.mlp(blk.n2(h)); continue
                A[:, :, t:t + 1] = k.to(A.dtype); Bf[:, :, t:t + 1] = v.to(Bf.dtype)
                Kc, Vc = A[:, :, :t + 1], Bf[:, :, :t + 1]
                if isinstance(at, AlibiAttn):
                    key = (t, H, q.dtype)  # built once (during warm-up, before graph capture), reused by every layer/replay
                    if key not in s.bias:
                        s.bias[key] = (alibi_slopes(H, x.device)[:, None, None] * (torch.arange(t + 1, device=x.device) - t).float()[None, None]).to(q.dtype)[None]
                    y = F.scaled_dot_product_attention(q, Kc, Vc, attn_mask=s.bias[key])
                else:
                    y = F.scaled_dot_product_attention(q, Kc, Vc)
                y = at.o(y.reshape(Bn, D))
            h = h + y; h = h + blk.mlp(blk.n2(h))
        logits = m.head(h); s.tok.copy_(logits.argmax(-1)); return logits

@torch.no_grad()
def check_correctness(kind, data, n=128):
    """fp32: logits from step-by-step decoding == logits from the parallel forward() (max abs diff), on real enwik8 text."""
    torch.manual_seed(0); m = build(kind).eval().float()
    x = data.val[:n][None].cuda(); ref = m(x)[0]
    dec = Decoder(m, 1, n, dtype=torch.float32, fill_random=False); out = []
    for t in range(n):
        dec.tok.copy_(x[:, t]); out.append(dec.step(t)[0])
    # also check sheaf_step == the library step() on one random input
    blk = next(b for b in m.blocks if isinstance(b, HBlock)) if kind == "alibi-hybrid-h6" else None
    lib = None
    if blk is not None:
        mx = blk.mix; xx = torch.randn(2, m.c.dim, device="cuda"); S = torch.randn(2, mx.H, mx.dh, mx.dh, device="cuda"); bf = torch.randn(2, mx.offsets[-1], mx.D, device="cuda")
        o1, (S1, b1) = mx.step(xx, (S, bf)); o2, S2, b2 = sheaf_step(mx, xx, S, bf, torch.tensor([d - 1 for d in mx.offsets], device="cuda"))
        lib = max((o1 - o2).abs().max().item(), (S1 - S2).abs().max().item(), (b1 - b2).abs().max().item())
    return dict(step_vs_forward_maxabs=(torch.stack(out) - ref).abs().max().item(), logit_scale=ref.abs().mean().item(), sheaf_step_vs_library_maxabs=lib)

@torch.no_grad()
def bench(kind, T, B, graph=True, reps=100):
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); torch.manual_seed(0)
    m = build(kind).eval().to(torch.bfloat16)
    try:
        dec = Decoder(m, B, T); t = T - 1
        if graph:
            s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(3): dec.step(t)
            torch.cuda.current_stream().wait_stream(s)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g): dec.step(t)
            run = g.replay
        else: run = lambda: dec.step(t)
        for _ in range(10): run()
        torch.cuda.synchronize(); ts = []
        for _ in range(reps):
            t0 = time.perf_counter(); run(); torch.cuda.synchronize(); ts.append(time.perf_counter() - t0)
        ms = statistics.median(ts) * 1e3
        r = dict(status="ok", ms_per_step=ms, tok_s=B / ms * 1e3, peak_mem_mb=torch.cuda.max_memory_allocated() / 1e6,
                 cache_mb=dec.state_bytes / 1e6, weights_mb=sum(p.numel() * p.element_size() for p in m.parameters()) / 1e6)
        del dec
    except torch.OutOfMemoryError: r = dict(status="OOM")
    del m; torch.cuda.empty_cache(); return r

data = ByteData("data/enwik8", ctx=1024, n_train=1_000_000, n_val=1_000_000)
for kind in KINDS:
    if f"check|{kind}" not in R: R[f"check|{kind}"] = check_correctness(kind, data); save(); log("check", kind, R[f"check|{kind}"])
for kind in KINDS:
    for T in CTXS:
        k = f"{kind}|{T}|B1|eager"
        if k not in R: R[k] = bench(kind, T, 1, graph=False); save(); log(k, R[k])
        for B in BATCHES:
            k = f"{kind}|{T}|B{B}|graph"
            if k not in R: R[k] = bench(kind, T, B); save(); log(k, R[k])
# serving capacity: max concurrent sequences in 8 GB at context T (doubling search), with throughput at that batch
for kind in KINDS:
    for T in (4096, 16384, 65536):
        k = f"capacity|{kind}|{T}"
        if k in R: continue
        best = None; B = 1; hi = None
        while B <= 4096:  # doubling to bracket, then binary search for the exact largest batch that fits
            r = bench(kind, T, B, reps=30)
            if r["status"] != "ok": hi = B; break
            best = dict(max_batch=B, **r); B *= 2
        lo = best["max_batch"] if best else 0
        while hi is not None and hi - lo > max(1, lo // 32):
            mid = (lo + hi) // 2; r = bench(kind, T, mid, reps=30)
            if r["status"] == "ok": best = dict(max_batch=mid, **r); lo = mid
            else: hi = mid
        R[k] = best or dict(max_batch=0); save(); log(k, R[k])
log("ALL DONE")
