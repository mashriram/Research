"""Checks whether NoPE or ALiBi -- the two context-extension techniques flagged as untested in every
caveat so far (they need retraining from scratch with the position-encoding choice baked in, unlike
YaRN which is a post-hoc fix, see gpu_yarn.py) -- beat plain RoPE, YaRN, or the hybrid 3:1 architecture
on length extrapolation. Same 14M-param/width-384/8-layer/ctx-1024 recipe as gpu_scale.py, real enwik8
training from scratch for each variant. Compares against: plain RoPE (1.103/2.111/3.547, gpu_scale.py),
YaRN (1.130/1.370/1.472, gpu_yarn.py), hybrid 3:1 (1.014/0.995/1.084, gpu_ratio.py)."""
import sys, os, json, time, math, torch, torch.nn as nn, torch.nn.functional as F; sys.path.insert(0, ".")
from hsnlm import Config, ByteData
from hsnlm.models import RMSNorm, MLP, LM
assert torch.cuda.is_available()
torch.backends.cudnn.benchmark = True
DIM, HEADS, LAYERS, CTX, B, VOCAB = 384, 8, 8, 1024, 16, 256
STEPS = 1500
OUT = "benchmark_results/development/positional_encoding/gpu_posenc.json"; os.makedirs("benchmark_results", exist_ok=True)
R = json.load(open(OUT)) if os.path.exists(OUT) else {}
def save(): json.dump(R, open(OUT, "w"), indent=1)
def log(*a): print(time.strftime("%H:%M:%S"), *a, flush=True)

def alibi_bias(T, H, device, dtype):
    slopes = torch.tensor([2.0 ** (-8.0 * (i + 1) / H) for i in range(H)], device=device, dtype=torch.float32)
    pos = torch.arange(T, device=device)
    rel = (pos[None, :] - pos[:, None]).to(torch.float32)  # rel[i,j] = j-i; causal: j<=i so rel<=0
    bias = slopes[:, None, None] * rel[None, :, :]  # [H,T,T], more negative = further in the past
    causal = torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1)
    bias = bias.masked_fill(causal, float("-inf"))
    return bias[None].to(dtype)  # [1,H,T,T]

class AttnPE(nn.Module):
    """mode: 'rope' | 'nope' | 'alibi'."""
    def __init__(s, c, mode):
        super().__init__(); s.h = c.heads; s.mode = mode; s.qkv = nn.Linear(c.dim, 3 * c.dim, bias=False); s.o = nn.Linear(c.dim, c.dim, bias=False)
    def forward(s, x):
        B_, T, D = x.shape
        q, k, v = s.qkv(x).view(B_, T, 3, s.h, D // s.h).permute(2, 0, 3, 1, 4)
        if s.mode == "rope":
            pos = torch.arange(T, device=x.device); dh = D // s.h
            inv = 1.0 / (10000.0 ** (torch.arange(0, dh, 2, device=x.device).float() / dh))
            ang = pos[:, None].float() * inv[None]; cos, sin = ang.cos(), ang.sin()
            def rope(t): x1, x2 = t[..., :dh // 2], t[..., dh // 2:]; return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1)
            q, k = rope(q), rope(k)
        if s.mode == "alibi":
            bias = alibi_bias(T, s.h, x.device, q.dtype)
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)
        else:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return s.o(y.transpose(1, 2).reshape(B_, T, D))

class TBlockPE(nn.Module):
    def __init__(s, c, mode):
        super().__init__(); s.n1 = RMSNorm(c.dim); s.n2 = RMSNorm(c.dim); s.mix = AttnPE(c, mode); s.mlp = MLP(c.dim, c.mlp_mult, False)
    def forward(s, x): x = x + s.mix(s.n1(x)); return x + s.mlp(s.n2(x))

class GPTPE(LM):
    use_pos = False
    mode = "rope"
    def make_block(s, i): return TBlockPE(s.c, s.mode)

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

def build(mode):
    m = GPTPE(Config(vocab=VOCAB, dim=DIM, layers=LAYERS, ctx=CTX, heads=HEADS, mlp_mult=4.0)); m.mode = mode
    for b in m.blocks: b.mix.mode = mode
    return m.cuda()

def train_lm(mode, lr, steps, seed, final_eval=False):
    torch.manual_seed(seed); gen = torch.Generator().manual_seed(seed); m = build(mode)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.05)
    torch.cuda.synchronize(); t0 = time.time(); curve = []
    for s in range(steps):
        cur = lr * min(1, (s + 1) / 100) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * s / steps)))
        for g in opt.param_groups: g["lr"] = cur
        x, y = data.batch(B, gen); x, y = x.cuda(), y.cuda(); opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = F.cross_entropy(m(x).reshape(-1, VOCAB), y.reshape(-1))
        loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        if (s + 1) % max(steps // 10, 1) == 0:
            torch.cuda.synchronize(); curve.append((s + 1, round(loss.item() / math.log(2), 4), round(time.time() - t0, 1)))
            log(mode, "lr", lr, curve[-1])
    train_s = time.time() - t0
    res = dict(params=m.n_params(), steps=steps, train_s=train_s, tok_s=steps * B * CTX / train_s, val_bpb=eval_len(m, CTX, nwin=64))
    if final_eval:
        for L in (2048, 4096): res[f"val_bpb_len{L}"] = eval_len(m, L, nwin=32)
    return res

for mode in ("nope",):
    for lr in (1e-3, 2e-3):
        k = f"sweep|{mode}|{lr}"
        if k not in R: R[k] = train_lm(mode, lr, 100, seed=100); save(); log("sweep", k, R[k]["val_bpb"])
best = {m: min((1e-3, 2e-3), key=lambda lr: R.get(f"sweep|{m}|{lr}", {"val_bpb": 9})["val_bpb"]) for m in ("nope", "alibi")}
log("best lr", best); R["best_lr"] = best; save()
for mode in ("nope",):
    k = f"lm|{mode}"
    if k not in R: R[k] = train_lm(mode, best[mode], STEPS, seed=0, final_eval=True); save(); log("DONE", k, {a: b for a, b in R[k].items() if a != "curve"})
log("ALL DONE")
