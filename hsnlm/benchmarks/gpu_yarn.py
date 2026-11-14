"""Fair-baseline check: both findings reports flagged the GPT baseline as 'plain RoPE, no NoPE/YaRN/
ALiBi' -- i.e. the worst-case Transformer, not the best-known one. This retrains the same 14M-param
GPT (width 384, 8 layers, ctx 1024, same recipe as gpu_scale.py) and evaluates length extrapolation
both with plain RoPE (replicating gpu_scale.py's number as a sanity check) and with YaRN-scaled RoPE
(NTK-by-parts frequency interpolation + attention temperature scaling), applied ONLY at eval time for
contexts beyond the 1024 training length -- no retraining needed for YaRN itself, exactly how it's
used in practice to extend an already-trained model's context. Compares both against the hybrid-fused
3:1 number already measured (1.084 bpb @ 4096, gpu_ratio.json/gpu_scale.json)."""
import sys, os, copy, json, time, math, torch, torch.nn as nn, torch.nn.functional as F; sys.path.insert(0, ".")
from hsnlm import Config, ByteData
from hsnlm.models import RMSNorm, MLP, LM
assert torch.cuda.is_available()
torch.backends.cudnn.benchmark = True
DIM, HEADS, LAYERS, CTX, B, VOCAB = 384, 8, 8, 1024, 16, 256
STEPS = 1500
OUT = "results/gpu_yarn.json"; os.makedirs("results", exist_ok=True)
R = json.load(open(OUT)) if os.path.exists(OUT) else {}
def save(): json.dump(R, open(OUT, "w"), indent=1)
def log(*a): print(time.strftime("%H:%M:%S"), *a, flush=True)

def rope_plain(x, pos, base=10000.0):
    dh = x.shape[-1]; inv = 1.0 / (base ** (torch.arange(0, dh, 2, device=x.device).float() / dh))
    ang = pos[:, None].float() * inv[None]; cos, sin = ang.cos(), ang.sin(); x1, x2 = x[..., :dh // 2], x[..., dh // 2:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1)

def rope_yarn(x, pos, orig_ctx, base=10000.0, beta_fast=32, beta_slow=1):
    """NTK-by-parts frequency interpolation (Peng et al. 2023) + YaRN attention-temperature scaling.
    No-op (reduces to plain RoPE) when max position is within orig_ctx."""
    dh = x.shape[-1]; max_pos = pos.max().item() + 1
    scale = max(1.0, max_pos / orig_ctx)
    dim_idx = torch.arange(0, dh, 2, device=x.device).float()
    inv_freq = 1.0 / (base ** (dim_idx / dh))
    if scale > 1.0:
        low = max(0.0, dh * math.log(orig_ctx / (beta_fast * 2 * math.pi)) / (2 * math.log(base)))
        high = min(dh / 2 - 1, dh * math.log(orig_ctx / (beta_slow * 2 * math.pi)) / (2 * math.log(base)))
        ramp = ((dim_idx / 2 - low) / max(high - low, 1e-3)).clamp(0, 1)  # 0 = high-freq (keep), 1 = low-freq (interpolate)
        inv_freq = inv_freq * (1 - ramp) + (inv_freq / scale) * ramp
        mscale = 0.1 * math.log(scale) + 1.0
    else:
        mscale = 1.0
    ang = pos[:, None].float() * inv_freq[None]; cos, sin = ang.cos(), ang.sin(); x1, x2 = x[..., :dh // 2], x[..., dh // 2:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1) * mscale

class AttnY(nn.Module):
    """Same as hsnlm.models.Attn, but the rope function used at forward() is swappable (plain for
    training-length context, YaRN beyond it) without touching trained weights."""
    def __init__(s, c):
        super().__init__(); s.h = c.heads; s.qkv = nn.Linear(c.dim, 3 * c.dim, bias=False); s.o = nn.Linear(c.dim, c.dim, bias=False)
    def forward(s, x, rope_fn):
        B_, T, D = x.shape
        q, k, v = s.qkv(x).view(B_, T, 3, s.h, D // s.h).permute(2, 0, 3, 1, 4)
        pos = torch.arange(T, device=x.device); q, k = rope_fn(q, pos), rope_fn(k, pos)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return s.o(y.transpose(1, 2).reshape(B_, T, D))

class TBlockY(nn.Module):
    def __init__(s, c):
        super().__init__(); s.n1 = RMSNorm(c.dim); s.n2 = RMSNorm(c.dim); s.mix = AttnY(c); s.mlp = MLP(c.dim, c.mlp_mult, False)
    def forward(s, x, rope_fn): x = x + s.mix(s.n1(x), rope_fn); return x + s.mlp(s.n2(x))

class GPTY(LM):
    use_pos = False
    def make_block(s, i): return TBlockY(s.c)
    def forward(s, x, rope_fn=rope_plain):
        h = s.embed_in(x)
        for b in s.blocks: h = b(h, rope_fn)
        return s.head(h)

data = ByteData("data/enwik8", ctx=CTX, n_train=40_000_000, n_val=1_000_000)

@torch.no_grad()
def eval_len(m, L, rope_fn, nwin=32, Bv=8):
    m.eval(); v = data.val; n = min(nwin, (len(v) - 1) // L); tot = cnt = 0
    for i in range(0, n, Bv):
        xs = torch.stack([v[j * L:(j + 1) * L] for j in range(i, min(i + Bv, n))]).cuda()
        ys = torch.stack([v[j * L + 1:(j + 1) * L + 1] for j in range(i, min(i + Bv, n))]).cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            tot += F.cross_entropy(m(xs, rope_fn).reshape(-1, VOCAB), ys.reshape(-1), reduction="sum").item()
        cnt += ys.numel()
    m.train(); return tot / cnt / math.log(2)

def train_lm(lr, steps, seed):
    torch.manual_seed(seed); gen = torch.Generator().manual_seed(seed)
    m = GPTY(Config(vocab=VOCAB, dim=DIM, layers=LAYERS, ctx=CTX, heads=HEADS, mlp_mult=4.0)).cuda()
    opt = torch.optim.AdamW(m.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.05)
    torch.cuda.synchronize(); t0 = time.time()
    for s in range(steps):
        cur = lr * min(1, (s + 1) / 100) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * s / steps)))
        for g in opt.param_groups: g["lr"] = cur
        x, y = data.batch(B, gen); x, y = x.cuda(), y.cuda(); opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = F.cross_entropy(m(x).reshape(-1, VOCAB), y.reshape(-1))
        loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        if (s + 1) % max(steps // 10, 1) == 0:
            torch.cuda.synchronize(); log("gpt-yarn lr", lr, s + 1, round(loss.item() / math.log(2), 4), round(time.time() - t0, 1))
    train_s = time.time() - t0
    res = dict(params=m.n_params(), steps=steps, train_s=train_s, val_bpb=eval_len(m, CTX, rope_plain))
    for L in (2048, 4096):
        yarn_fn = lambda x, pos, L=L: rope_yarn(x, pos, orig_ctx=CTX)
        res[f"plain_len{L}"] = eval_len(m, L, rope_plain)
        res[f"yarn_len{L}"] = eval_len(m, L, yarn_fn)
    return res

for lr in (1e-3, 2e-3):
    k = f"sweep|{lr}"
    if k not in R:
        torch.manual_seed(100); gen = torch.Generator().manual_seed(100)
        m = GPTY(Config(vocab=VOCAB, dim=DIM, layers=LAYERS, ctx=CTX, heads=HEADS, mlp_mult=4.0)).cuda()
        opt = torch.optim.AdamW(m.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.05)
        for s in range(100):
            cur = lr * min(1, (s + 1) / 100) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * s / 100)))
            for g in opt.param_groups: g["lr"] = cur
            x, y = data.batch(B, gen); x, y = x.cuda(), y.cuda(); opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = F.cross_entropy(m(x).reshape(-1, VOCAB), y.reshape(-1))
            loss.backward(); opt.step()
        R[k] = eval_len(m, CTX, rope_plain); save(); log("sweep", k, R[k])
best_lr = min((1e-3, 2e-3), key=lambda lr: R[f"sweep|{lr}"]); R["best_lr"] = best_lr; save()
if "lm" not in R: R["lm"] = train_lm(best_lr, STEPS, seed=0); save(); log("DONE", R["lm"])
log("ALL DONE")
