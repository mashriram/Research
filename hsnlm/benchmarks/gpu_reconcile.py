"""Reconciles the two halves of this report that were never measured together: §3's speed crossover
(narrow width, long context, bf16 -- sheaf-delta fused beats GPT there) and §5a's quality ranking
(3:1 hybrid ratio beats GPT -- but measured at width 384/ctx 1024, where GPT is still faster). This
trains REAL models (not just speed-timing with random data) at width=128/8 layers/ctx=16384/batch=1 --
the exact width and a context length inside the confirmed bf16 speed-crossover zone (§3.1: ctx 16384,
bf16, fused ~190-210k tok/s vs GPT's 132.9k tok/s) -- and asks: does the 3:1 hybrid's quality edge
survive here, in a regime where it is ALSO the faster architecture? Real enwik8 data, 1 seed.
Cheap: batch=1 x ctx=16384 is only 16,384 tokens/step, so even thousands of steps finish in minutes."""
import sys, os, copy, json, time, math, torch, torch.nn.functional as F; sys.path.insert(0, ".")
from hsnlm import GPT, Config, ByteData
from hsnlm.models import LM, HBlock, TBlock
assert torch.cuda.is_available()
torch.backends.cudnn.benchmark = True
DIM, HEADS, LAYERS, CTX, B, VOCAB = 128, 4, 8, 16384, 1, 256
STEPS = 3000  # 3000 * 16384 = 49.2M tokens/arch -- a real token budget, not a speed-only probe
OUT = "benchmark_results/development/reconcile_speed_quality/gpu_reconcile.json"; os.makedirs("benchmark_results", exist_ok=True)
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
    if arch == "hsn-fused": from hsnlm import HSNLM; return HSNLM(Config(vocab=VOCAB, dim=DIM, layers=LAYERS, ctx=CTX, heads=HEADS, mlp_mult=3.4, mixer="sheafdelta-fused")).cuda()
    pat = "dddaddda"  # the §5a winner (ratio 2/8 = 3:1), same pattern, now at width 128 instead of 384
    return HybridFusedLM(Config(vocab=VOCAB, dim=DIM, layers=LAYERS, ctx=CTX, heads=HEADS, mlp_mult=3.65, pattern=pat)).cuda()
ARCHS = ["gpt", "hsn-fused", "hybrid-fused"]
data = ByteData("data/enwik8", ctx=CTX, n_train=60_000_000, n_val=2_000_000)

@torch.no_grad()
def eval_len(m, L, nwin=16, Bv=1):
    m.eval(); v = data.val; n = min(nwin, (len(v) - 1) // L); tot = cnt = 0
    for i in range(0, n, Bv):
        xs = torch.stack([v[j * L:(j + 1) * L] for j in range(i, min(i + Bv, n))]).cuda()
        ys = torch.stack([v[j * L + 1:(j + 1) * L + 1] for j in range(i, min(i + Bv, n))]).cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            tot += F.cross_entropy(m(xs).reshape(-1, VOCAB), ys.reshape(-1), reduction="sum").item()
        cnt += ys.numel()
    m.train(); return tot / cnt / math.log(2)

def train_lm(arch, lr, steps, seed):
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
            log(arch, curve[-1])
    torch.cuda.synchronize(); train_s = time.time() - t0
    return dict(params=m.n_params(), steps=steps, train_s=train_s, tok_s=steps * B * CTX / train_s,
                peak_mem_mb=torch.cuda.max_memory_allocated() / 1e6, val_bpb=eval_len(m, CTX))

for arch in ARCHS:
    for lr in (1e-3, 2e-3):
        k = f"sweep|{arch}|{lr}"
        if k not in R: R[k] = train_lm(arch, lr, 300, seed=100); save(); log("sweep", k, R[k]["val_bpb"])
best = {a: min((1e-3, 2e-3), key=lambda lr: R.get(f"sweep|{a}|{lr}", {"val_bpb": 9})["val_bpb"]) for a in ARCHS}; log("best lr", best); R["best_lr"] = best; save()
for arch in ARCHS:
    k = f"lm|{arch}"
    if k not in R: R[k] = train_lm(arch, best[arch], STEPS, seed=0); save(); log("DONE", k, R[k])
log("ALL DONE")
