"""Direct answer to 'how do we beat ALiBi': ALiBi's quality edge (0.956 bpb, gpu_posenc.py) over the
RoPE-based 3:1 hybrid (1.014 bpb, gpu_ratio.py) might come from its attention mechanism specifically.
This builds the SAME 3:1 sheaf-delta:attention hybrid, but with the attention layers using ALiBi
instead of RoPE -- if ALiBi's quality boost transfers, this should beat both pure ALiBi (extrapolation
is ALiBi's weak point, 1.192 at 4x) and the RoPE-hybrid (quality is RoPE-hybrid's weak point, 1.014 at
train length), potentially dominating both. Same 14M-param/width-384/8-layer/ctx-1024 recipe, same
enwik8 data. Also includes a second-seed check on the three key configs (ALiBi, RoPE-hybrid, ALiBi-
hybrid) to make sure the ranking isn't within noise before investing further."""
import sys, os, copy, json, time, math, torch, torch.nn as nn, torch.nn.functional as F; sys.path.insert(0, ".")
from hsnlm import Config, ByteData
from hsnlm.models import RMSNorm, MLP, LM, HBlock
assert torch.cuda.is_available()
torch.backends.cudnn.benchmark = True
DIM, HEADS, CTX, B, VOCAB = 384, 8, 1024, 16, 256
STEPS = 1500
RATIO_NAME = sys.argv[1]  # e.g. "4to1" or "5to1"
LAYERS = int(sys.argv[2])
N_ATTN = int(sys.argv[3])

def spread_pattern(n_attn, n_layers):
    pat = ["d"] * n_layers
    for i in range(n_attn):
        pos = round((i + 1) * n_layers / n_attn) - 1
        pat[pos] = "a"
    return "".join(pat)
PATTERN = spread_pattern(N_ATTN, LAYERS)
OUT = f"benchmark_results/development/alibi_hybrid_ratio/gpu_alibi_hybrid_{RATIO_NAME}.json"; os.makedirs("benchmark_results", exist_ok=True)
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

class AlibiHybridLM(LM):
    use_pos = False
    def make_block(s, i):
        c = copy.copy(s.c); ch = (s.c.pattern or "d" * s.c.layers)[i]
        if ch == "a": return TBlockAlibi(c)
        c.mixer = "sheafdelta-fused"; return HBlock(c)

data = ByteData("data/enwik8", ctx=CTX, n_train=40_000_000, n_val=1_000_000)

@torch.no_grad()
def eval_len(m, L, nwin=32, Bv=8):
    m.eval(); v = data.val; n = min(nwin, (len(v) - 1) // L); tot = cnt = 0
    for i in range(0, n, Bv):
        xs = torch.stack([v[j * L:(j + 1) * L] for j in range(i, min(i + Bv, n))]).cuda()
        ys = torch.stack([v[j * L + 1:(j + 1) * L + 1] for j in range(i, min(i + Bv, n))]).cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            tot += F.cross_entropy(m(xs).reshape(-1, VOCAB), ys.reshape(-1), reduction="sum").item()
        cnt += ys.numel()
    m.train(); return tot / cnt / math.log(2)

def build():
    c = Config(vocab=VOCAB, dim=DIM, layers=LAYERS, ctx=CTX, heads=HEADS, mlp_mult=3.65, pattern=PATTERN)
    return AlibiHybridLM(c).cuda()

def train_lm(lr, steps, seed, final_eval=False):
    torch.manual_seed(seed); gen = torch.Generator().manual_seed(seed); m = build()
    opt = torch.optim.AdamW(m.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.05)
    torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); t0 = time.time(); curve = []
    for s in range(steps):
        cur = lr * min(1, (s + 1) / 100) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * s / steps)))
        for g in opt.param_groups: g["lr"] = cur
        x, y = data.batch(B, gen); x, y = x.cuda(), y.cuda(); opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = F.cross_entropy(m(x).reshape(-1, VOCAB), y.reshape(-1))
        loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        if (s + 1) % max(steps // 10, 1) == 0:
            torch.cuda.synchronize(); curve.append((s + 1, round(loss.item() / math.log(2), 4), round(time.time() - t0, 1)))
            log("alibi-hybrid lr", lr, curve[-1])
    train_s = time.time() - t0
    res = dict(params=m.n_params(), steps=steps, train_s=train_s, tok_s=steps * B * CTX / train_s,
               peak_mem_mb=torch.cuda.max_memory_allocated() / 1e6, val_bpb=eval_len(m, CTX, nwin=64))
    if final_eval:
        for L in (2048, 4096): res[f"val_bpb_len{L}"] = eval_len(m, L)
    return res

for lr in (1e-3, 2e-3):
    k = f"sweep|{lr}"
    if k not in R: R[k] = train_lm(lr, 100, seed=100); save(); log("sweep", k, R[k]["val_bpb"])
best_lr = min((1e-3, 2e-3), key=lambda lr: R.get(f"sweep|{lr}", {"val_bpb": 9})["val_bpb"]); R["best_lr"] = best_lr; save()
if "lm" not in R: R["lm"] = train_lm(best_lr, STEPS, seed=0, final_eval=True); save(); log("DONE", {a: b for a, b in R["lm"].items() if a != "curve"})
log("ALL DONE")
