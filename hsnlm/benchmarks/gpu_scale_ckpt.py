"""Push past the 14M-param VRAM ceiling hit in gpu_scale.py, via gradient checkpointing (trade compute
for activation memory) applied uniformly to all three architectures -- more general than the existing
reversible-block mechanism (hsnlm/reversible.py), which only supports a homogeneous sheaf-mixer stack,
not GPT or the hybrid pattern. Each block's forward is recomputed during backward instead of its
activations being stored, so peak memory becomes closer to O(one block) instead of O(depth x block).
Same recipe/data as gpu_scale.py; width/layers raised to whatever this 8GB card can hold with
checkpointing on. usage: gpu_scale_ckpt.py probe   -- just print peak memory for one step at WIDTH/LAYERS
       gpu_scale_ckpt.py            -- full training run (sweep -> 1500-step main stage)"""
import sys, os, copy, json, time, math, torch, torch.nn.functional as F; sys.path.insert(0, ".")
from torch.utils.checkpoint import checkpoint
from hsnlm import GPT, Config, ByteData
from hsnlm.models import LM, HBlock, TBlock
assert torch.cuda.is_available()
torch.backends.cudnn.benchmark = True
PROBE = len(sys.argv) > 1 and sys.argv[1] == "probe"
WIDTH, HEADSN, LAYERSN, CTX, B, VOCAB = 1024, 16, 16, 1024, 16, 256  # ~2.3x gpu_scale.py's param count
OUT = "results/gpu_scale_ckpt.json"; os.makedirs("results", exist_ok=True)
R = json.load(open(OUT)) if os.path.exists(OUT) else {}
def save(): json.dump(R, open(OUT, "w"), indent=1)
def log(*a): print(time.strftime("%H:%M:%S"), *a, flush=True)

class CkptMixin:
    def forward(s, x):
        h = s.embed_in(x)
        for b in s.blocks:
            h = checkpoint(b, h, use_reentrant=False) if s.training else b(h)
        return s.head(h)

class GPTCkpt(CkptMixin, GPT): pass

class HybridFusedLM(CkptMixin, LM):
    use_pos = False
    def make_block(s, i):
        c = copy.copy(s.c); ch = (s.c.pattern or "d" * s.c.layers)[i]
        if ch == "a": return TBlock(c)
        c.mixer = "sheafdelta-fused"; return HBlock(c)

def build(arch):
    if arch == "gpt": return GPTCkpt(Config(vocab=VOCAB, dim=WIDTH, layers=LAYERSN, ctx=CTX, heads=HEADSN, mlp_mult=4.0)).cuda()
    if arch == "hsn-fused": c = Config(vocab=VOCAB, dim=WIDTH, layers=LAYERSN, ctx=CTX, heads=HEADSN, mlp_mult=3.4, mixer="sheafdelta-fused"); return HybridFusedLM(c).cuda()
    pat = ("d" * (LAYERSN - max(1, LAYERSN // 8))) + "a" * max(1, LAYERSN // 8)
    c = Config(vocab=VOCAB, dim=WIDTH, layers=LAYERSN, ctx=CTX, heads=HEADSN, mlp_mult=3.65, pattern=pat); return HybridFusedLM(c).cuda()
ARCHS = ["gpt", "hsn-fused", "hybrid-fused"]

if PROBE:
    for arch in ARCHS:
        try:
            m = build(arch); x = torch.randint(0, VOCAB, (B, CTX), device="cuda"); y = torch.randint(0, VOCAB, (B, CTX), device="cuda")
            torch.cuda.reset_peak_memory_stats()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = F.cross_entropy(m(x).reshape(-1, VOCAB), y.reshape(-1))
            loss.backward(); torch.cuda.synchronize()
            print(arch, "params", m.n_params(), "peak_mem_mb", round(torch.cuda.max_memory_allocated() / 1e6, 1))
        except torch.cuda.OutOfMemoryError:
            print(arch, "OOM at width", WIDTH, "layers", LAYERSN, "batch", B)
        if "m" in dir(): del m
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    sys.exit(0)

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
    res = dict(params=m.n_params(), steps=steps, train_s=train_s, tok_s=steps * B * CTX / train_s,
               peak_mem_mb=torch.cuda.max_memory_allocated() / 1e6, val_bpb=eval_len(m, CTX))
    if final_eval:
        for L in (2048, 4096): res[f"val_bpb_len{L}"] = eval_len(m, L, nwin=32)
    return res

SWEEP_STEPS, MAIN_STEPS = 40, 300  # cut from gpu_scale.py's 100/1500: at ~200M params + checkpointing
                                    # recompute, 1500 steps/arch would take hours; this is a directional
                                    # check at a much bigger scale, not a fully converged comparison.
for arch in ARCHS:
    for lr in (1e-3, 2e-3):
        k = f"sweep|{arch}|{lr}"
        if k not in R: R[k] = train_lm(arch, lr, SWEEP_STEPS, seed=100); save(); log("sweep", k, R[k]["val_bpb"])
best = {a: min((1e-3, 2e-3), key=lambda lr: R.get(f"sweep|{a}|{lr}", {"val_bpb": 9})["val_bpb"]) for a in ARCHS}; log("best lr", best); R["best_lr"] = best; save()
for arch in ARCHS:
    k = f"lm|{arch}"
    if k not in R: R[k] = train_lm(arch, best[arch], MAIN_STEPS, seed=0, final_eval=True); save(); log("DONE", k, {a: b for a, b in R[k].items() if a != "curve"})
log("ALL DONE")
