"""Follow-up to gpu_scale_ckpt.py: that run used a 7:1-equivalent attention ratio (baked in before the
G5 ratio ablation, gpu_ratio.py, found 3:1 to be the actual Pareto-best point at the 14M/width-384 scale).
This re-runs ONLY the hybrid-fused architecture at the same ~200M/width-1024/16-layer/checkpointed scale,
with an evenly-spread 3:1-equivalent ratio (4 attention layers out of 16), to check whether that ratio
still wins at ~14x the parameter count. Writes to its own results file so it doesn't clobber
gpu_scale_ckpt.json; gpt and hsn-fused endpoints are unaffected by this ratio choice and are reused
from gpu_scale_ckpt.json rather than re-run."""
import sys, os, copy, json, time, math, torch, torch.nn.functional as F; sys.path.insert(0, ".")
from torch.utils.checkpoint import checkpoint
from hsnlm import Config, ByteData
from hsnlm.models import LM, HBlock, TBlock
assert torch.cuda.is_available()
torch.backends.cudnn.benchmark = True
WIDTH, HEADSN, LAYERSN, CTX, B, VOCAB = 1024, 16, 16, 1024, 16, 256
SWEEP_STEPS, MAIN_STEPS = 40, 300
OUT = "benchmark_results/development/gpu_scale_ckpt/gpu_scale_ckpt_31.json"; os.makedirs("benchmark_results", exist_ok=True)
R = json.load(open(OUT)) if os.path.exists(OUT) else {}
def save(): json.dump(R, open(OUT, "w"), indent=1)
def log(*a): print(time.strftime("%H:%M:%S"), *a, flush=True)

def spread_pattern(n_attn, n_layers):
    pat = ["d"] * n_layers
    for i in range(n_attn):
        pos = round((i + 1) * n_layers / n_attn) - 1
        pat[pos] = "a"
    return "".join(pat)
PATTERN = spread_pattern(max(1, round(LAYERSN / 4)), LAYERSN)  # 1 attn per 4 layers ~ the ratio-2-of-8 winner

class CkptMixin:
    def forward(s, x):
        h = s.embed_in(x)
        for b in s.blocks:
            h = checkpoint(b, h, use_reentrant=False) if s.training else b(h)
        return s.head(h)

class HybridFusedLM(CkptMixin, LM):
    use_pos = False
    def make_block(s, i):
        c = copy.copy(s.c); ch = (s.c.pattern or "d" * s.c.layers)[i]
        if ch == "a": return TBlock(c)
        c.mixer = "sheafdelta-fused"; return HBlock(c)

def build():
    c = Config(vocab=VOCAB, dim=WIDTH, layers=LAYERSN, ctx=CTX, heads=HEADSN, mlp_mult=3.65, pattern=PATTERN)
    return HybridFusedLM(c).cuda()

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
            log("hybrid-31", "lr", lr, "seed", seed, curve[-1])
    torch.cuda.synchronize(); train_s = time.time() - t0
    res = dict(pattern=PATTERN, params=m.n_params(), steps=steps, train_s=train_s, tok_s=steps * B * CTX / train_s,
               peak_mem_mb=torch.cuda.max_memory_allocated() / 1e6, val_bpb=eval_len(m, CTX))
    if final_eval:
        for L in (2048, 4096): res[f"val_bpb_len{L}"] = eval_len(m, L, nwin=32)
    return res

log("pattern", PATTERN)
for lr in (1e-3, 2e-3):
    k = f"sweep|{lr}"
    if k not in R: R[k] = train_lm(lr, SWEEP_STEPS, seed=100); save(); log("sweep", k, R[k]["val_bpb"])
best_lr = min((1e-3, 2e-3), key=lambda lr: R.get(f"sweep|{lr}", {"val_bpb": 9})["val_bpb"]); R["best_lr"] = best_lr; save()
if "lm" not in R: R["lm"] = train_lm(best_lr, MAIN_STEPS, seed=0, final_eval=True); save(); log("DONE", {a: b for a, b in R["lm"].items() if a != "curve"})
log("ALL DONE")
