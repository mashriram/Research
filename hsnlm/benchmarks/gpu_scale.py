"""Larger-than-CPU-suite LM quality/speed run on enwik8, scaled to what one 8GB GPU can hold comfortably
(dim 384, 8 layers, ctx 1024) -- the GPU analogue of cpu_suite's LM stage, but ~15x the parameters and
4x the context. Compares GPT vs HSN-fused (pure fla-fused sheaf-delta) vs Hybrid-fused (7 sheaf-delta + 1
attention). bf16 autocast. Resumable: results/gpu_scale.json. usage: gpu_scale.py [smoke]"""
import sys, os, copy, json, time, math, torch, torch.nn.functional as F; sys.path.insert(0, ".")
from hsnlm import GPT, Config, ByteData
from hsnlm.models import LM, HBlock, TBlock
assert torch.cuda.is_available()
torch.backends.cudnn.benchmark = True
SMOKE = len(sys.argv) > 1 and sys.argv[1] == "smoke"
DIM, HEADS, LAYERS, CTX, B, VOCAB = 384, 8, 8, 1024, 16, 256
OUT = "results/gpu_scale_smoke.json" if SMOKE else "results/gpu_scale.json"; os.makedirs("results", exist_ok=True)
R = json.load(open(OUT)) if os.path.exists(OUT) else {}
def save(): json.dump(R, open(OUT, "w"), indent=1)
def log(*a): print(time.strftime("%H:%M:%S"), *a, flush=True)
S = (lambda n: 3) if SMOKE else (lambda n: n)

class HybridFusedLM(LM):
    use_pos = False
    def make_block(s, i):
        c = copy.copy(s.c); ch = (s.c.pattern or "d" * s.c.layers)[i]
        if ch == "a": return TBlock(c)
        c.mixer = "sheafdelta-fused"; return HBlock(c)

def build(arch):
    if arch == "gpt": return GPT(Config(vocab=VOCAB, dim=DIM, layers=LAYERS, ctx=CTX, heads=HEADS, mlp_mult=4.0)).cuda()
    if arch == "hsn-fused": c = Config(vocab=VOCAB, dim=DIM, layers=LAYERS, ctx=CTX, heads=HEADS, mlp_mult=3.4, mixer="sheafdelta-fused"); from hsnlm import HSNLM; return HSNLM(c).cuda()
    c = Config(vocab=VOCAB, dim=DIM, layers=LAYERS, ctx=CTX, heads=HEADS, mlp_mult=3.65, pattern="ddddddda"); return HybridFusedLM(c).cuda()
ARCHS = ["gpt", "hsn-fused", "hybrid-fused"]
data = ByteData("data/enwik8", ctx=CTX, n_train=2_000_000 if SMOKE else 40_000_000, n_val=1_000_000)

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

def train_lm(arch, lr, steps, seed, final_eval=False):
    torch.manual_seed(seed); gen = torch.Generator().manual_seed(seed); m = build(arch)
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
            log(arch, "lr", lr, "seed", seed, curve[-1])
    torch.cuda.synchronize(); train_s = time.time() - t0
    res = dict(params=m.n_params(), steps=steps, train_s=train_s, tok_s=steps * B * CTX / train_s, curve=curve,
               peak_mem_mb=torch.cuda.max_memory_allocated() / 1e6, val_bpb=eval_len(m, CTX))
    if final_eval:
        for L in (2048, 4096): res[f"val_bpb_len{L}"] = eval_len(m, L, nwin=32)
    return res

for arch in ARCHS:
    for lr in (1e-3, 2e-3):
        k = f"sweep|{arch}|{lr}"
        if k not in R: R[k] = train_lm(arch, lr, S(100), seed=100); save(); log("sweep", k, R[k]["val_bpb"])
best = {a: min((1e-3, 2e-3), key=lambda lr: R.get(f"sweep|{a}|{lr}", {"val_bpb": 9})["val_bpb"]) for a in ARCHS}; log("best lr", best); R["best_lr"] = best; save()
for arch in ARCHS:
    k = f"lm|{arch}"
    if k not in R: R[k] = train_lm(arch, best[arch], S(1500), seed=0, final_eval=True); save(); log("DONE", k, {a: b for a, b in R[k].items() if a != "curve"})
log("ALL DONE")
