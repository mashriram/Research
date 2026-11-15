"""Multi-seed check on the closest race in the whole report: RoPE-hybrid 3:1 (gpu_ratio.py, seed 0:
1.0136/0.9950/1.0835), pure ALiBi (gpu_posenc_recheck.py, seed 0: 1.0300/1.0062/1.0872), and ALiBi-hybrid
3:1 (gpu_posenc_recheck.py, seed 0: 1.0237/1.0048/1.0875) are separated by only 0.01-0.016 bpb -- on the
order of single-seed noise. This reruns all three at seeds 0, 1, 2 (seed 0 redone fresh here too, under
torch.compile, rather than reusing the uncompiled seed-0 numbers, so the whole multi-seed set is
internally consistent) with full eval (nwin=64 @1024, nwin=32 @2048/@4096) and no missing cells."""
import sys, os, copy, json, time, math, torch, torch.nn as nn, torch.nn.functional as F; sys.path.insert(0, ".")
from hsnlm import Config, ByteData
from hsnlm.models import RMSNorm, MLP, LM, HBlock, TBlock
assert torch.cuda.is_available()
torch.backends.cudnn.benchmark = True
DIM, HEADS, LAYERS, CTX, B, VOCAB = 384, 8, 8, 1024, 16, 256
LR = 0.002
STEPS = 1500
PATTERN = "dddaddda"
SEEDS = (0, 1, 2)
OUT = "results/gpu_multiseed_compiled.json"; os.makedirs("results", exist_ok=True)
R = json.load(open(OUT)) if os.path.exists(OUT) else {}
def save(): json.dump(R, open(OUT, "w"), indent=1)
def log(*a): print(time.strftime("%H:%M:%S"), *a, flush=True)

def alibi_bias(T, H, device, dtype):
    slopes = torch.tensor([2.0 ** (-8.0 * (i + 1) / H) for i in range(H)], device=device, dtype=torch.float32)
    pos = torch.arange(T, device=device)
    rel = (pos[None, :] - pos[:, None]).to(torch.float32)
    bias = slopes[:, None, None] * rel[None, :, :]
    causal = torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1)
    bias = bias.masked_fill(causal, float("-inf"))
    return bias[None].to(dtype)

class AlibiAttn(nn.Module):
    def __init__(s, c):
        super().__init__(); s.h = c.heads; s.qkv = nn.Linear(c.dim, 3 * c.dim, bias=False); s.o = nn.Linear(c.dim, c.dim, bias=False)
    def forward(s, x):
        B_, T, D = x.shape
        q, k, v = s.qkv(x).view(B_, T, 3, s.h, D // s.h).permute(2, 0, 3, 1, 4)
        bias = alibi_bias(T, s.h, x.device, q.dtype)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)
        return s.o(y.transpose(1, 2).reshape(B_, T, D))

class TBlockAlibi(nn.Module):
    def __init__(s, c):
        super().__init__(); s.n1 = RMSNorm(c.dim); s.n2 = RMSNorm(c.dim); s.mix = AlibiAttn(c); s.mlp = MLP(c.dim, c.mlp_mult, False)
    def forward(s, x): x = x + s.mix(s.n1(x)); return x + s.mlp(s.n2(x))

class AlibiPureLM(LM):
    use_pos = False
    def make_block(s, i): return TBlockAlibi(s.c)

class AlibiHybridLM(LM):
    use_pos = False
    def make_block(s, i):
        c = copy.copy(s.c); ch = (s.c.pattern or "d" * s.c.layers)[i]
        if ch == "a": return TBlockAlibi(c)
        c.mixer = "sheafdelta-fused"; return HBlock(c)

class RopeHybridLM(LM):
    use_pos = False
    def make_block(s, i):
        c = copy.copy(s.c); ch = (s.c.pattern or "d" * s.c.layers)[i]
        if ch == "a": return TBlock(c)
        c.mixer = "sheafdelta-fused"; return HBlock(c)

def build(kind):
    if kind == "alibi-pure":
        return AlibiPureLM(Config(vocab=VOCAB, dim=DIM, layers=LAYERS, ctx=CTX, heads=HEADS, mlp_mult=4.0)).cuda()
    if kind == "alibi-hybrid":
        return AlibiHybridLM(Config(vocab=VOCAB, dim=DIM, layers=LAYERS, ctx=CTX, heads=HEADS, mlp_mult=3.65, pattern=PATTERN)).cuda()
    return RopeHybridLM(Config(vocab=VOCAB, dim=DIM, layers=LAYERS, ctx=CTX, heads=HEADS, mlp_mult=3.65, pattern=PATTERN)).cuda()

data = ByteData("data/enwik8", ctx=CTX, n_train=40_000_000, n_val=1_000_000)

@torch.no_grad()
def eval_len(m, L, nwin, Bv=8):
    m.eval(); v = data.val; n = min(nwin, (len(v) - 1) // L); tot = cnt = 0
    for i in range(0, n, Bv):
        xs = torch.stack([v[j * L:(j + 1) * L] for j in range(i, min(i + Bv, n))]).cuda()
        ys = torch.stack([v[j * L + 1:(j + 1) * L + 1] for j in range(i, min(i + Bv, n))]).cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            tot += F.cross_entropy(m(xs).reshape(-1, VOCAB), ys.reshape(-1), reduction="sum").item()
        cnt += ys.numel()
    m.train(); return tot / cnt / math.log(2)

def train_lm(kind, lr, steps, seed):
    torch.manual_seed(seed); gen = torch.Generator().manual_seed(seed); m0 = build(kind)
    m = torch.compile(m0)
    opt = torch.optim.AdamW(m0.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.05)
    x0, y0 = data.batch(B, torch.Generator().manual_seed(999)); x0, y0 = x0.cuda(), y0.cuda()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        l0 = F.cross_entropy(m(x0).reshape(-1, VOCAB), y0.reshape(-1))
    l0.backward(); opt.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); t0 = time.time()
    for s in range(steps):
        cur = lr * min(1, (s + 1) / 100) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * s / steps)))
        for g in opt.param_groups: g["lr"] = cur
        x, y = data.batch(B, gen); x, y = x.cuda(), y.cuda(); opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = F.cross_entropy(m(x).reshape(-1, VOCAB), y.reshape(-1))
        loss.backward(); torch.nn.utils.clip_grad_norm_(m0.parameters(), 1.0); opt.step()
        if (s + 1) % 300 == 0:
            torch.cuda.synchronize(); log(kind, "seed", seed, s + 1, round(loss.item() / math.log(2), 4), round(time.time() - t0, 1))
    torch.cuda.synchronize(); train_s = time.time() - t0
    res = dict(kind=kind, seed=seed, params=m0.n_params(), steps=steps, train_s=train_s, tok_s=steps * B * CTX / train_s,
               peak_mem_mb=torch.cuda.max_memory_allocated() / 1e6,
               val_bpb=eval_len(m0, CTX, nwin=64))
    for L in (2048, 4096): res[f"val_bpb_len{L}"] = eval_len(m0, L, nwin=32)
    return res

for kind in ("rope-hybrid", "alibi-pure", "alibi-hybrid"):
    for seed in SEEDS:
        k = f"{kind}|seed{seed}"
        if k not in R:
            R[k] = train_lm(kind, LR, STEPS, seed); save(); log("DONE", k, {a: b for a, b in R[k].items() if a != "curve"})
log("ALL DONE")
