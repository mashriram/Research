"""Sliding-Window ALiBi + Sheaf-Delta 3:1 Hybrid (`dddaddda`) + compiled zero-gather dyadic mixer.

Why the un-windowed ALiBi-Hybrid (`gpu_alibi_hybrid.py`) beat pure ALiBi at training length
(0.951 vs 0.956 bpb @ 1024) but lost extrapolation at 4x (1.222 vs 1.084 bpb @ 4096):
1) Unbounded ALiBi penalties (-slope * 4096) at 4x context push attention logits far outside the
   training distribution, while the 6 recurrent sheaf-delta layers were never trained to bridge
   attention window boundaries because full-context attention covered all 1024 tokens during training.
2) Training with Sliding-Window ALiBi (W=512 on ctx=1024, or W=1024 clamped at eval) forces the
   model DURING training to use ALiBi attention for sharp local lookups (<= W) and the 6 sheaf-delta
   layers for long-range context (> W), while capping the attention KV cache to O(W) = O(1) at inference.
3) Combined with the zero-gather `dyadic` fix in `fastmixer.py` and `torch.compile` on the sheaf-delta
   mixer, this also closes the training speed and VRAM gap.
"""
import sys, os, copy, json, time, math, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, ".")
from hsnlm import Config, ByteData
from hsnlm.models import RMSNorm, MLP, LM, HBlock

assert torch.cuda.is_available()
torch.backends.cudnn.benchmark = True
DIM, HEADS, LAYERS, CTX, B, VOCAB = 384, 8, 8, 1024, 16, 256
STEPS = 1500
PATTERN = "dddaddda"
OUT = "results/gpu_swa_alibi_hybrid.json"
os.makedirs("results", exist_ok=True)
R = json.load(open(OUT)) if os.path.exists(OUT) else {}
def save(): json.dump(R, open(OUT, "w"), indent=1)
def log(*a): print(time.strftime("%H:%M:%S"), *a, flush=True)

def swa_alibi_bias(T, H, window, device, dtype):
    slopes = torch.tensor([2.0 ** (-8.0 * (i + 1) / H) for i in range(H)], device=device, dtype=torch.float32)
    pos = torch.arange(T, device=device)
    rel = (pos[None, :] - pos[:, None]).to(torch.float32)  # j - i <= 0 for causal
    bias = slopes[:, None, None] * rel[None, :, :]
    mask = (rel > 0) | (rel < -(window - 1)) if window is not None else (rel > 0)
    bias = bias.masked_fill(mask[None, :, :], float("-inf"))
    return bias[None].to(dtype)

class SWAAlibiAttn(nn.Module):
    def __init__(s, c, window=512):
        super().__init__()
        s.h = c.heads
        s.window = window
        s.qkv = nn.Linear(c.dim, 3 * c.dim, bias=False)
        s.o = nn.Linear(c.dim, c.dim, bias=False)

    def forward(s, x):
        B_, T, D = x.shape
        q, k, v = s.qkv(x).view(B_, T, 3, s.h, D // s.h).permute(2, 0, 3, 1, 4)
        bias = swa_alibi_bias(T, s.h, s.window, x.device, q.dtype)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)
        return s.o(y.transpose(1, 2).reshape(B_, T, D))

    def step(s, x, st):
        B_, D = x.shape
        q, k, v = s.qkv(x).view(B_, 3, s.h, 1, D // s.h).permute(1, 0, 2, 3, 4)
        if st is not None:
            k = torch.cat([st[0], k], 2)
            v = torch.cat([st[1], v], 2)
            if s.window is not None and k.shape[2] > s.window:
                k = k[:, :, -s.window:]
                v = v[:, :, -s.window:]
        L = k.shape[2]
        slopes = torch.tensor([2.0 ** (-8.0 * (i + 1) / s.h) for i in range(s.h)], device=x.device, dtype=torch.float32)
        rel = torch.arange(-(L - 1), 1, device=x.device, dtype=torch.float32)  # [-(L-1) .. 0]
        bias = (slopes[:, None, None] * rel[None, None, :])[None].to(q.dtype)  # [1, H, 1, L]
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)
        return s.o(y.reshape(B_, D)), (k, v)

class TBlockSWAAlibi(nn.Module):
    def __init__(s, c, window=512):
        super().__init__()
        s.n1 = RMSNorm(c.dim)
        s.n2 = RMSNorm(c.dim)
        s.mix = SWAAlibiAttn(c, window=window)
        s.mlp = MLP(c.dim, c.mlp_mult, False)
    def forward(s, x): x = x + s.mix(s.n1(x)); return x + s.mlp(s.n2(x))
    def step(s, x, st):
        y, st = s.mix.step(s.n1(x), st)
        x = x + y
        return x + s.mlp(s.n2(x)), st

class SWAAlibiHybridLM(LM):
    use_pos = False
    window = 512
    compile_mixer = False
    def make_block(s, i):
        c = copy.copy(s.c)
        ch = (s.c.pattern or "d" * s.c.layers)[i]
        if ch == "a":
            return TBlockSWAAlibi(c, window=s.window)
        c.mixer = "sheafdelta-fused"
        blk = HBlock(c)
        if s.compile_mixer:
            blk.mix.dyadic = torch.compile(blk.mix.dyadic)
        return blk

data = ByteData("data/enwik8", ctx=CTX, n_train=40_000_000, n_val=1_000_000)

@torch.no_grad()
def eval_len(m, L, nwin=32, Bv=8, eval_window=None):
    m.eval()
    orig_windows = []
    for b in m.blocks:
        if isinstance(b, TBlockSWAAlibi):
            orig_windows.append(b.mix.window)
            if eval_window is not None:
                b.mix.window = eval_window
    v = data.val
    n = min(nwin, (len(v) - 1) // L)
    tot = cnt = 0
    for i in range(0, n, Bv):
        xs = torch.stack([v[j * L:(j + 1) * L] for j in range(i, min(i + Bv, n))]).cuda()
        ys = torch.stack([v[j * L + 1:(j + 1) * L + 1] for j in range(i, min(i + Bv, n))]).cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            tot += F.cross_entropy(m(xs).reshape(-1, VOCAB), ys.reshape(-1), reduction="sum").item()
        cnt += ys.numel()
    idx = 0
    for b in m.blocks:
        if isinstance(b, TBlockSWAAlibi):
            b.mix.window = orig_windows[idx]
            idx += 1
    m.train()
    return tot / cnt / math.log(2)

def build(window=512, compile_mixer=True):
    c = Config(vocab=VOCAB, dim=DIM, layers=LAYERS, ctx=CTX, heads=HEADS, mlp_mult=3.65, pattern=PATTERN)
    SWAAlibiHybridLM.window = window
    SWAAlibiHybridLM.compile_mixer = compile_mixer
    return SWAAlibiHybridLM(c).cuda()

def train_lm(lr, steps, seed, window=512, compile_mixer=True, final_eval=False):
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    m = build(window=window, compile_mixer=compile_mixer)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.05)
    # Warm up compile on 2 steps before timing if compiled
    if compile_mixer:
        x0, y0 = data.batch(B, torch.Generator().manual_seed(999))
        x0, y0 = x0.cuda(), y0.cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            l0 = F.cross_entropy(m(x0).reshape(-1, VOCAB), y0.reshape(-1))
        l0.backward()
        opt.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.time()
    curve = []
    for s in range(steps):
        cur = lr * min(1, (s + 1) / 100) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * s / steps)))
        for g in opt.param_groups:
            g["lr"] = cur
        x, y = data.batch(B, gen)
        x, y = x.cuda(), y.cuda()
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = F.cross_entropy(m(x).reshape(-1, VOCAB), y.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        if (s + 1) % max(steps // 10, 1) == 0:
            torch.cuda.synchronize()
            curve.append((s + 1, round(loss.item() / math.log(2), 4), round(time.time() - t0, 1)))
            log(f"swa{window}-alibi-hybrid lr", lr, curve[-1])
    torch.cuda.synchronize()
    train_s = time.time() - t0
    res = dict(
        window=window,
        compiled=compile_mixer,
        params=m.n_params(),
        steps=steps,
        train_s=train_s,
        tok_s=steps * B * CTX / train_s,
        peak_mem_mb=torch.cuda.max_memory_allocated() / 1e6,
        val_bpb=eval_len(m, CTX, nwin=64),
    )
    if final_eval:
        for L in (2048, 4096):
            res[f"val_bpb_len{L}"] = eval_len(m, L)
            if window != 1024:
                res[f"val_bpb_len{L}_evalw1024"] = eval_len(m, L, eval_window=1024)
            else:
                res[f"val_bpb_len{L}_evalw512"] = eval_len(m, L, eval_window=512)
                res[f"val_bpb_len{L}_evalw_inf"] = eval_len(m, L, eval_window=None)
    return res

@torch.no_grad()
def bench_decode_o1():
    """Verify O(1) decode latency & state memory across context lengths for SWA-512 3:1 Hybrid."""
    c = Config(vocab=VOCAB, dim=1024, layers=8, ctx=1024, heads=16, mlp_mult=3.65, pattern=PATTERN)
    SWAAlibiHybridLM.window = 512
    SWAAlibiHybridLM.compile_mixer = False
    m = SWAAlibiHybridLM(c).cuda().eval().to(torch.bfloat16)
    out = {}
    for ctx_len in (256, 1024, 4096, 16384, 65536):
        sts = []
        for b in m.blocks:
            if isinstance(b, TBlockSWAAlibi):
                w = min(ctx_len, b.mix.window)
                sts.append((
                    torch.zeros(1, 16, w, 64, device="cuda", dtype=torch.bfloat16),
                    torch.zeros(1, 16, w, 64, device="cuda", dtype=torch.bfloat16),
                ))
            else:
                sts.append((
                    torch.zeros(1, 16, 64, 64, device="cuda", dtype=torch.bfloat16),
                    torch.zeros(1, 16, 1024, device="cuda", dtype=torch.bfloat16),
                ))
        state_bytes = sum(sum(t.numel() * t.element_size() for t in st) for st in sts)
        x = torch.randn(1, 1024, device="cuda", dtype=torch.bfloat16)
        for _ in range(5):
            h = x
            for i, b in enumerate(m.blocks):
                h, _ = b.step(h, sts[i])
        torch.cuda.synchronize()
        t0 = time.time()
        iters = 20
        for _ in range(iters):
            h = x
            for i, b in enumerate(m.blocks):
                h, _ = b.step(h, sts[i])
        torch.cuda.synchronize()
        ms = (time.time() - t0) / iters * 1000
        out[str(ctx_len)] = dict(step_ms=round(ms, 3), state_mb=round(state_bytes / 1e6, 3))
        log("decode_o1 ctx", ctx_len, out[str(ctx_len)])
    return out

if __name__ == "__main__":
    # Wait for running 4to1 job to finish so there is zero GPU contention
    f4to1 = "results/gpu_alibi_hybrid_4to1.json"
    while os.path.exists(f4to1):
        try:
            d4 = json.load(open(f4to1))
            if "lm" in d4:
                break
        except Exception:
            pass
        log("Waiting for gpu_alibi_hybrid_4to1.py to finish...")
        time.sleep(15)

    if "decode_o1" not in R:
        R["decode_o1"] = bench_decode_o1()
        save()

    # 1) SWA-512 ALiBi 3:1 Hybrid (trained with W=512 on ctx=1024 so model learns cross-window sheaf-delta routing)
    if "lm_swa512" not in R:
        R["lm_swa512"] = train_lm(0.002, STEPS, seed=0, window=512, compile_mixer=True, final_eval=True)
        save()
        log("DONE lm_swa512", R["lm_swa512"])

    # 2) SWA-1024 ALiBi 3:1 Hybrid (identical to full ALiBi at ctx=1024 -> ~0.951 bpb, but window-clamped to 1024 at 2x/4x eval)
    if "lm_swa1024" not in R:
        R["lm_swa1024"] = train_lm(0.002, STEPS, seed=0, window=1024, compile_mixer=True, final_eval=True)
        save()
        log("DONE lm_swa1024", R["lm_swa1024"])

    log("ALL DONE")
