"""Fair iso-quality (time-to-target-loss) comparison: does the RoPE 3:1 hybrid reach GPT's final
quality in less wall-clock time once BOTH are compiled the same way (torch.compile(model), applied
uniformly -- not compiling only the hybrid's mixer while leaving GPT uncompiled, which is not a fair
comparison). Same recipe as gpu_scale.py/gpu_ratio.py (14M params, width 384, 8 layers, ctx 1024,
lr=0.002, enwik8), real training curves saved for both so the iso-quality crossing point is measured,
not guessed from an uncompiled baseline's curve."""
import sys, os, copy, json, time, math, torch, torch.nn.functional as F; sys.path.insert(0, ".")
from hsnlm import GPT, Config, ByteData
from hsnlm.models import LM, HBlock, TBlock
assert torch.cuda.is_available()
torch.backends.cudnn.benchmark = True
DIM, HEADS, LAYERS, CTX, B, VOCAB = 384, 8, 8, 1024, 16, 256
LR = 0.002
STEPS = 1500
PATTERN = "dddaddda"
OUT = "results/gpu_isoquality_compiled.json"; os.makedirs("results", exist_ok=True)
R = json.load(open(OUT)) if os.path.exists(OUT) else {}
def save(): json.dump(R, open(OUT, "w"), indent=1)
def log(*a): print(time.strftime("%H:%M:%S"), *a, flush=True)

class HybridFusedLM(LM):
    use_pos = False
    def make_block(s, i):
        c = copy.copy(s.c); ch = (s.c.pattern or "d" * s.c.layers)[i]
        if ch == "a": return TBlock(c)
        c.mixer = "sheafdelta-fused"; return HBlock(c)

def build(arch):
    if arch == "gpt": return GPT(Config(vocab=VOCAB, dim=DIM, layers=LAYERS, ctx=CTX, heads=HEADS, mlp_mult=4.0)).cuda()
    return HybridFusedLM(Config(vocab=VOCAB, dim=DIM, layers=LAYERS, ctx=CTX, heads=HEADS, mlp_mult=3.65, pattern=PATTERN)).cuda()

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

def train_lm(arch, lr, steps, seed, compiled):
    torch.manual_seed(seed); gen = torch.Generator().manual_seed(seed); m0 = build(arch)
    m = torch.compile(m0) if compiled else m0
    opt = torch.optim.AdamW(m0.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.05)
    if compiled:
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
        if (s + 1) % 50 == 0:
            torch.cuda.synchronize(); curve.append((s + 1, round(loss.item() / math.log(2), 4), round(time.time() - t0, 2)))
            if (s + 1) % 300 == 0: log(arch, "compiled" if compiled else "uncompiled", curve[-1])
    train_s = time.time() - t0
    res = dict(params=m0.n_params(), steps=steps, train_s=train_s, tok_s=steps * B * CTX / train_s, curve=curve,
               peak_mem_mb=torch.cuda.max_memory_allocated() / 1e6, val_bpb=eval_len(m0, CTX))
    return res

for arch in ("gpt", "hybrid"):
    k = f"{arch}|compiled"
    if k not in R: R[k] = train_lm(arch, LR, STEPS, seed=0, compiled=True); save(); log("DONE", k, {a: b for a, b in R[k].items() if a != "curve"})
log("ALL DONE")
