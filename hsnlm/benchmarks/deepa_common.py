"""Shared model/eval/train code for the Phase-A reviewer-response benchmarks
(gpu_wikitext103_3seed.py, gpu_extrap_16k.py, gpu_scaling_law.py, gpu_overhead_profile.py).

The three architectures are exactly the ones used in gpu_splitwd_3seed.py / gpu_multiseed_compiled.py /
gpu_isoquality_compiled.py (same classes, same parameter names, same uniform-WD AdamW recipe):
  alibi-hybrid-h6 : 8 layers "dddaddda", d = HBlock(SheafDeltaMixer, fused fla kernel, mixer_heads=6, SwiGLU MLP
                    mult 3.65), a = ALiBi attention block (GELU MLP mult 3.65)
  alibi-pure      : 8 ALiBi attention blocks, GELU MLP mult 4.0
  rope-gpt        : hsnlm.GPT (RoPE attention, no learned positions), GELU MLP mult 4.0

One change vs. those scripts: AlibiAttn handles T > DENSE_MAX (4096) by processing queries in row blocks, with the
ALiBi bias for each block built on the fly and keys/values sliced to the causal prefix. Rows of causal attention
are independent, so this is mathematically identical to the dense-mask path (verified numerically in
check_alibi_chunked()); it exists only because the dense fp32 [H,T,T] bias is 8.6 GB at T=16384, more than the
8 GB card. For T <= 4096 the original dense-mask code path is used unchanged. FlexAttention would be the natural
fix, but torch 2.6's flex_attention rejects non-power-of-2 head dims (ours is 384/8 = 48)."""
import copy, math, time, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from hsnlm import Config, GPT
from hsnlm.models import RMSNorm, MLP, LM, HBlock, TBlock

VOCAB = 256
DENSE_MAX = 4096
QCHUNK = 1024
def log(*a): print(time.strftime("%H:%M:%S"), *a, flush=True)

_SLOPES = {}
def alibi_slopes(H, device):  # memoized: building it from a Python list is a host->device copy, illegal inside CUDA-graph capture
    key = (H, str(torch.device(device)))
    if key not in _SLOPES: _SLOPES[key] = _slopes(H, device)
    return _SLOPES[key]

def _slopes(H, device):
    return torch.tensor([2.0 ** (-8.0 * (i + 1) / H) for i in range(H)], device=device, dtype=torch.float32)

def alibi_bias(T, H, device, dtype):  # verbatim from gpu_splitwd_3seed.py
    slopes = alibi_slopes(H, device)
    pos = torch.arange(T, device=device)
    rel = (pos[None, :] - pos[:, None]).to(torch.float32)
    bias = slopes[:, None, None] * rel[None, :, :]
    causal = torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1)
    bias = bias.masked_fill(causal, float("-inf"))
    return bias[None].to(dtype)

def _alibi_rows(q, k, v, i0):  # q [B,H,tq,dh] = query rows i0..i0+tq; k,v = causal prefix [B,H,i0+tq,dh]
    H, tq, tk = q.shape[1], q.shape[2], k.shape[2]
    qi = torch.arange(i0, i0 + tq, device=q.device); ki = torch.arange(tk, device=q.device)
    rel = (ki[None, :] - qi[:, None]).to(torch.float32)
    bias = alibi_slopes(H, q.device)[:, None, None] * rel[None]
    bias = bias.masked_fill(rel[None] > 0, float("-inf"))
    return F.scaled_dot_product_attention(q, k, v, attn_mask=bias[None].to(q.dtype))

ALIBI_IMPL = "dense"  # "dense" (original: explicit [H,T,T] bias mask) | "flash" (exact ALiBi folded into q.k; see below)
def pad_dh(dh): return -(-(dh + 2) // 8) * 8  # dh + 2 position dims, rounded up to a multiple of 8 (FlashAttention granularity)
PAD_DH = pad_dh(48)  # = 56, the paper models' attention head dim

def alibi_aug_qk(q, k, pos_k, pad=None):
    pad = pad or pad_dh(q.shape[-1])
    """Exact ALiBi without a bias tensor. Score = s*q.k + m_h*(j - i); the -m_h*i term is constant across each query row so
    softmax ignores it, leaving s*q.k + m_h*j. Append j = 256*hi + lo (hi, lo < 256 are exact integers in bf16) to every key
    and [256*m_h/s, m_h/s] to every query: then s*q'.k' = s*q.k + m_h*j. Zero-pad the head dim to `pad` (FlashAttention
    needs q,k,v of one head dim). With is_causal=True this runs on the fused FlashAttention / mem-efficient kernel: O(T)
    memory, no [H,T,T] tensor. Only approximation: the two query constants are rounded to bf16 (rel. err <= 2^-9, i.e. each
    head's slope is off by <0.2%)."""
    B, H, Tq, dh = q.shape; s = dh ** -0.5; m = alibi_slopes(H, q.device)
    hi, lo = (pos_k // 256).to(torch.float32), (pos_k % 256).to(torch.float32)
    kx = torch.stack([hi, lo], -1)[None, None].expand(B, H, -1, 2).to(k.dtype)
    qx = torch.stack([256 * m / s, m / s], -1)[None, :, None, :].expand(B, H, Tq, 2).to(q.dtype)
    z = lambda t: t.new_zeros(*t.shape[:-1], pad - dh - 2)
    return torch.cat([q, qx, z(q)], -1), torch.cat([k, kx, z(k)], -1)

def alibi_attend_flash(q, k, v):
    T, dh = q.shape[2], q.shape[3]
    q2, k2 = alibi_aug_qk(q, k, torch.arange(T, device=q.device))
    v2 = torch.cat([v, v.new_zeros(*v.shape[:-1], pad_dh(dh) - dh)], -1)
    return F.scaled_dot_product_attention(q2, k2, v2, is_causal=True, scale=dh ** -0.5)[..., :dh]

def alibi_attend(q, k, v):
    T = q.shape[2]
    if ALIBI_IMPL == "flash": return alibi_attend_flash(q, k, v)
    if T <= DENSE_MAX:
        return F.scaled_dot_product_attention(q, k, v, attn_mask=alibi_bias(T, q.shape[1], q.device, q.dtype))
    outs = []
    for i in range(0, T, QCHUNK):
        j = min(i + QCHUNK, T); args = (q[:, :, i:j], k[:, :, :j], v[:, :, :j], i)
        outs.append(checkpoint(_alibi_rows, *args, use_reentrant=False) if torch.is_grad_enabled() else _alibi_rows(*args))
    return torch.cat(outs, 2)

class AlibiAttn(nn.Module):
    def __init__(s, c):
        super().__init__(); s.h = c.heads; s.qkv = nn.Linear(c.dim, 3 * c.dim, bias=False); s.o = nn.Linear(c.dim, c.dim, bias=False)
    def forward(s, x):
        B_, T, D = x.shape
        q, k, v = s.qkv(x).view(B_, T, 3, s.h, D // s.h).permute(2, 0, 3, 1, 4)
        y = alibi_attend(q, k, v)
        return s.o(y.transpose(1, 2).reshape(B_, T, D))

class TBlockAlibi(nn.Module):
    def __init__(s, c):
        super().__init__(); s.n1 = RMSNorm(c.dim); s.n2 = RMSNorm(c.dim); s.mix = AlibiAttn(c); s.mlp = MLP(c.dim, c.mlp_mult, False)
    def forward(s, x): x = x + s.mix(s.n1(x)); return x + s.mlp(s.n2(x))

class CkptMixin:  # activation checkpointing per block during training (same as gpu_scale_ckpt.py)
    ckpt = False
    def forward(s, x):
        h = s.embed_in(x)
        for b in s.blocks:
            h = checkpoint(b, h, use_reentrant=False) if (s.ckpt and s.training) else b(h)
        return s.head(h)

class AlibiPureLM(CkptMixin, LM):
    use_pos = False
    def make_block(s, i): return TBlockAlibi(s.c)

class AlibiHybridLM(CkptMixin, LM):
    use_pos = False
    def make_block(s, i):
        c = copy.copy(s.c); ch = (s.c.pattern or "d" * s.c.layers)[i]
        if ch == "a": return TBlockAlibi(c)
        c.mixer = "sheafdelta-fused"; return HBlock(c)

class RopeHybridLM(AlibiHybridLM):  # diagnostic: identical DEEPA hybrid, but its attention layers use RoPE instead of ALiBi
    def make_block(s, i):
        c = copy.copy(s.c); ch = (s.c.pattern or "d" * s.c.layers)[i]
        if ch == "a": c.rope = True; return TBlock(c)
        c.mixer = "sheafdelta-fused"; return HBlock(c)

class RopeGPT(CkptMixin, GPT): pass

def spread_pattern(n_layers, ratio=4):  # 1 attention layer per `ratio` layers, attention last in each group ("dddaddda" for 8)
    return "".join("a" if (i + 1) % ratio == 0 else "d" for i in range(n_layers))

def build(kind, dim=384, layers=8, heads=8, mixer_heads=6, ctx=1024, ckpt=False, mlp_hybrid=3.65, mlp_pure=4.0, vocab=VOCAB):
    if kind.endswith("-beta1"):  # control ablation: delta-rule step size restricted to (0,1) -> no negative eigenvalues
        m = build(kind[:-6], dim, layers, heads, mixer_heads, ctx, ckpt, mlp_hybrid, mlp_pure, vocab)
        for mod in m.modules():
            if hasattr(mod, "beta_max"): mod.beta_max = 1.0
        return m
    if kind == "alibi-pure":
        m = AlibiPureLM(Config(vocab=vocab, dim=dim, layers=layers, ctx=ctx, heads=heads, mlp_mult=mlp_pure))
    elif kind == "rope-gpt":
        m = RopeGPT(Config(vocab=vocab, dim=dim, layers=layers, ctx=ctx, heads=heads, mlp_mult=mlp_pure))
    elif kind == "alibi-hybrid-h6":
        m = AlibiHybridLM(Config(vocab=vocab, dim=dim, layers=layers, ctx=ctx, heads=heads, mixer_heads=mixer_heads,
                                 mlp_mult=mlp_hybrid, pattern=spread_pattern(layers)))
    elif kind == "rope-hybrid-h6":  # diagnostic only (not the paper model): ALiBi -> RoPE in the 2 attention layers
        m = RopeHybridLM(Config(vocab=vocab, dim=dim, layers=layers, ctx=ctx, heads=heads, mixer_heads=mixer_heads,
                                mlp_mult=mlp_hybrid, pattern=spread_pattern(layers)))
    elif kind == "sheafdelta-pure":  # every layer SheafDelta (no attention at all): isolates the mixer
        m = AlibiHybridLM(Config(vocab=vocab, dim=dim, layers=layers, ctx=ctx, heads=heads, mixer_heads=mixer_heads,
                                 mlp_mult=mlp_hybrid, pattern="d" * layers))
    else: raise ValueError(kind)
    m.ckpt = ckpt
    return m.cuda()

@torch.no_grad()
def eval_len(m, val, L, nwin, Bv=8):
    """bits/byte over the first `nwin` non-overlapping length-L windows of `val` (repo convention)."""
    m.eval(); n = min(nwin, (len(val) - 1) // L); tot = cnt = 0
    for i in range(0, n, Bv):
        xs = torch.stack([val[j * L:(j + 1) * L] for j in range(i, min(i + Bv, n))]).cuda()
        ys = torch.stack([val[j * L + 1:(j + 1) * L + 1] for j in range(i, min(i + Bv, n))]).cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            tot += F.cross_entropy(m(xs).reshape(-1, VOCAB).float(), ys.reshape(-1), reduction="sum").item()
        cnt += ys.numel()
    m.train(); return tot / cnt / math.log(2), n

def train_lm(m0, data, steps, seed, lr=0.002, B=16, ctx=1024, wd=0.05, compile=True, tag="", log_every=300):
    """Exact recipe of gpu_splitwd_3seed.py / gpu_multiseed_compiled.py: AdamW(0.9,0.95) uniform wd, 100-step warmup,
    cosine to 0.1*lr, grad clip 1.0, bf16 autocast, torch.compile, one warm-up fwd/bwd excluded from timing."""
    gen = torch.Generator().manual_seed(seed)
    m = torch.compile(m0) if compile else m0
    opt = torch.optim.AdamW(m0.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=wd)
    x0, y0 = data.batch(B, torch.Generator().manual_seed(999)); x0, y0 = x0.cuda(), y0.cuda()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        l0 = F.cross_entropy(m(x0).reshape(-1, VOCAB), y0.reshape(-1))
    l0.backward(); opt.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); t0 = time.time(); curve = []
    for s in range(steps):
        cur = lr * min(1, (s + 1) / 100) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * s / steps)))
        for g in opt.param_groups: g["lr"] = cur
        x, y = data.batch(B, gen); x, y = x.cuda(), y.cuda(); opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = F.cross_entropy(m(x).reshape(-1, VOCAB), y.reshape(-1))
        loss.backward(); torch.nn.utils.clip_grad_norm_(m0.parameters(), 1.0); opt.step()
        if (s + 1) % log_every == 0 or s + 1 == steps:
            torch.cuda.synchronize(); curve.append((s + 1, round(loss.item() / math.log(2), 4), round(time.time() - t0, 1)))
            log(tag, curve[-1])
    torch.cuda.synchronize(); train_s = time.time() - t0
    return dict(params=m0.n_params(), steps=steps, lr=lr, batch=B, ctx=ctx, tokens=steps * B * ctx, train_s=train_s,
                tok_s=steps * B * ctx / train_s, peak_mem_mb=torch.cuda.max_memory_allocated() / 1e6, curve=curve)

def check_alibi_chunked():
    """Max |dense - chunked| on random inputs at T = 2048 (2 chunks) and 4096 (4 chunks)."""
    out = {}
    for T in (2048, 4096):
        g = torch.Generator(device="cuda").manual_seed(0)
        q, k, v = [torch.randn(2, 8, T, 48, device="cuda", dtype=torch.bfloat16, generator=g) for _ in range(3)]
        dense = F.scaled_dot_product_attention(q, k, v, attn_mask=alibi_bias(T, 8, q.device, q.dtype))
        with torch.no_grad():
            ch = torch.cat([_alibi_rows(q[:, :, i:i + QCHUNK], k[:, :, :i + QCHUNK], v[:, :, :i + QCHUNK], i) for i in range(0, T, QCHUNK)], 2)
        out[T] = (dense.float() - ch.float()).abs().max().item()
    return out
