"""Tests one legitimate, separable idea extracted from the pasted 'GOAT LoSparse' code: does a
FIXED-rank low-rank factorization of the MLP's up/down projections save real parameters without
hurting quality, at the scale we already have a dense baseline for? Deliberately drops the pasted
code's two unvalidated/buggy pieces: (1) adaptive rank growth, which reassigns nn.Parameter objects
post-hoc -- the already-constructed AdamW optimizer keeps references to the OLD tensors, so expanded
rank columns silently never receive a gradient update; (2) the sparse "specialist" path, which adds
its own magnitude-mask machinery under torch.compile with a data-dependent topk -> constant
recompilation. This isolates the one piece worth measuring: plain bottleneck factorization, fixed
rank, ordinary backprop, no exotic machinery. Same recipe as gpu_scale.py (14M-param hybrid-fused
config, enwik8, bf16, 1500 steps) for direct comparison against the dense result (1.021 bpb, 14.64M
params, 760.2s, 4737.7 MB, from results/gpu_scale.json)."""
import sys, os, copy, json, time, math, torch, torch.nn as nn, torch.nn.functional as F; sys.path.insert(0, ".")
from hsnlm import Config, ByteData
from hsnlm.models import LM, HBlock, TBlock, RMSNorm, make_mixer
assert torch.cuda.is_available()
torch.backends.cudnn.benchmark = True
DIM, HEADS, LAYERS, CTX, B, VOCAB = 384, 8, 8, 1024, 16, 256
PATTERN = "ddddddda"  # the 7:1 winner from gpu_scale.py
STEPS = 1500
OUT = "results/gpu_lowrank_mlp.json"; os.makedirs("results", exist_ok=True)
R = json.load(open(OUT)) if os.path.exists(OUT) else {}
def save(): json.dump(R, open(OUT, "w"), indent=1)
def log(*a): print(time.strftime("%H:%M:%S"), *a, flush=True)

class LowRankLinear(nn.Module):
    """y = (x @ B^T) @ A^T, a plain bottleneck factorization -- no sparse path, no adaptive growth."""
    def __init__(s, in_f, out_f, rank):
        super().__init__(); s.B = nn.Linear(in_f, rank, bias=False); s.A = nn.Linear(rank, out_f, bias=False)
    def forward(s, x): return s.A(s.B(x))

class LowRankMLP(nn.Module):
    def __init__(s, d, mult, gated, rank):
        super().__init__(); s.gated = gated
        h = int(d * mult * (2 / 3 if gated else 1))
        s.up = LowRankLinear(d, h * (2 if gated else 1), rank); s.down = LowRankLinear(h, d, rank)
    def forward(s, x):
        u = s.up(x)
        if s.gated: a, b = u.chunk(2, -1); return s.down(F.silu(a) * b)
        return s.down(F.gelu(u))

class HBlockLR(nn.Module):
    def __init__(s, c, rank):
        super().__init__(); s.n1 = RMSNorm(c.dim); s.n2 = RMSNorm(c.dim); s.mix = make_mixer(c); s.mlp = LowRankMLP(c.dim, c.mlp_mult, True, rank)
    def forward(s, x): x = x + s.mix(s.n1(x)); return x + s.mlp(s.n2(x))

class TBlockLR(nn.Module):
    def __init__(s, c, rank):
        super().__init__(); s.n1 = RMSNorm(c.dim); s.n2 = RMSNorm(c.dim); s.mix = TBlock(c).mix; s.mlp = LowRankMLP(c.dim, c.mlp_mult, False, rank)
    def forward(s, x): x = x + s.mix(s.n1(x)); return x + s.mlp(s.n2(x))

class HybridFusedLowRankLM(LM):
    use_pos = False
    rank = None
    def make_block(s, i):
        c = copy.copy(s.c); ch = (s.c.pattern or "d" * s.c.layers)[i]
        if ch == "a": return TBlockLR(c, s.rank)
        c.mixer = "sheafdelta-fused"; return HBlockLR(c, s.rank)

def build(rank):
    c = Config(vocab=VOCAB, dim=DIM, layers=LAYERS, ctx=CTX, heads=HEADS, mlp_mult=3.65, pattern=PATTERN)
    m = HybridFusedLowRankLM.__new__(HybridFusedLowRankLM); m.rank = rank
    nn.Module.__init__(m); m.c = c
    m.embed = nn.Embedding(c.vocab, c.dim); m.pos = None
    m.blocks = nn.ModuleList([m.make_block(i) for i in range(c.layers)])
    m.norm_f = RMSNorm(c.dim); m.apply(m._init)
    return m.cuda()

data = ByteData("data/enwik8", ctx=CTX, n_train=40_000_000, n_val=1_000_000)

@torch.no_grad()
def eval_len(m, L, nwin=64, Bv=8):
    m.eval(); v = data.val; n = min(nwin, (len(v) - 1) // L); tot = cnt = 0
    for i in range(0, n, Bv):
        xs = torch.stack([v[j * L:(j + 1) * L] for j in range(i, min(i + Bv, n))]).cuda()
        ys = torch.stack([v[j * L + 1:(j + 1) * L + 1] for j in range(i, min(i + Bv, n))]).cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            tot += F.cross_entropy(m(xs).reshape(-1, VOCAB), ys.reshape(-1), reduction="sum").item()
        cnt += ys.numel()
    m.train(); return tot / cnt / math.log(2)

def train_lm(rank, lr, seed=0):
    torch.manual_seed(seed); gen = torch.Generator().manual_seed(seed); m = build(rank)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.05)
    torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); t0 = time.time(); curve = []
    for s in range(STEPS):
        cur = lr * min(1, (s + 1) / 100) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * s / STEPS)))
        for g in opt.param_groups: g["lr"] = cur
        x, y = data.batch(B, gen); x, y = x.cuda(), y.cuda(); opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = F.cross_entropy(m(x).reshape(-1, VOCAB), y.reshape(-1))
        loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        if (s + 1) % 150 == 0:
            torch.cuda.synchronize(); curve.append((s + 1, round(loss.item() / math.log(2), 4), round(time.time() - t0, 1)))
            log(f"rank={rank}", curve[-1])
    torch.cuda.synchronize(); train_s = time.time() - t0
    res = dict(params=m.n_params(), steps=STEPS, train_s=train_s, tok_s=STEPS * B * CTX / train_s,
               peak_mem_mb=torch.cuda.max_memory_allocated() / 1e6, val_bpb=eval_len(m, CTX))
    for L in (2048, 4096): res[f"val_bpb_len{L}"] = eval_len(m, L, nwin=32)
    return res

for rank in (64, 128, 192):
    for lr in (0.002,):  # reuse gpu_scale.py's already-picked lr; rank shouldn't shift the optimum much
        k = f"rank{rank}|{lr}"
        if k not in R: R[k] = train_lm(rank, lr); save(); log("DONE", k, {a: b for a, b in R[k].items() if a != "curve"})
log("ALL DONE")
