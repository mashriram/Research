"""G5 from findings_report.md's GPU plan: hybrid attention-ratio ablation. Same scale/recipe as
gpu_scale.py (14M params, width 384, 8 layers, ctx 1024, bf16, enwik8) so results are directly
comparable to the already-measured endpoints: 0 attention layers = hsn-fused (1.199 bpb, gpu_scale.py),
8 attention layers = gpt (1.103 bpb, gpu_scale.py), 1 attention layer = hybrid-fused 7:1 (1.021 bpb,
gpu_scale.py). This script fills in the missing ratios: 2, 3, 4 attention layers out of 8.
Attention layers are spread evenly across depth rather than clustered at the end."""
import sys, os, copy, json, time, math, torch, torch.nn.functional as F; sys.path.insert(0, ".")
from hsnlm import Config, ByteData
from hsnlm.models import LM, HBlock, TBlock
assert torch.cuda.is_available()
torch.backends.cudnn.benchmark = True
DIM, HEADS, LAYERS, CTX, B, VOCAB = 384, 8, 8, 1024, 16, 256
LR = 0.002  # reuse the value gpu_scale.py's sweep already picked for gpt/hsn-fused/hybrid-fused
STEPS = 1500
OUT = "results/gpu_ratio.json"; os.makedirs("results", exist_ok=True)
R = json.load(open(OUT)) if os.path.exists(OUT) else {}
def save(): json.dump(R, open(OUT, "w"), indent=1)
def log(*a): print(time.strftime("%H:%M:%S"), *a, flush=True)

def spread_pattern(n_attn, n_layers=LAYERS):
    """Evenly space n_attn 'a' positions among n_layers slots (rest 'd'), attention layers last in each span."""
    pat = ["d"] * n_layers
    for i in range(n_attn):
        pos = round((i + 1) * n_layers / n_attn) - 1
        pat[pos] = "a"
    return "".join(pat)

class HybridFusedLM(LM):
    use_pos = False
    def make_block(s, i):
        c = copy.copy(s.c); ch = (s.c.pattern or "d" * s.c.layers)[i]
        if ch == "a": return TBlock(c)
        c.mixer = "sheafdelta-fused"; return HBlock(c)

def build(n_attn):
    pat = spread_pattern(n_attn)
    c = Config(vocab=VOCAB, dim=DIM, layers=LAYERS, ctx=CTX, heads=HEADS, mlp_mult=3.65, pattern=pat)
    return HybridFusedLM(c).cuda(), pat

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

def train_lm(n_attn, seed=0):
    torch.manual_seed(seed); gen = torch.Generator().manual_seed(seed); m, pat = build(n_attn)
    opt = torch.optim.AdamW(m.parameters(), lr=LR, betas=(0.9, 0.95), weight_decay=0.05)
    torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); t0 = time.time(); curve = []
    for s in range(STEPS):
        cur = LR * min(1, (s + 1) / 100) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * s / STEPS)))
        for g in opt.param_groups: g["lr"] = cur
        x, y = data.batch(B, gen); x, y = x.cuda(), y.cuda(); opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = F.cross_entropy(m(x).reshape(-1, VOCAB), y.reshape(-1))
        loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        if (s + 1) % 150 == 0:
            torch.cuda.synchronize(); curve.append((s + 1, round(loss.item() / math.log(2), 4), round(time.time() - t0, 1)))
            log(f"attn={n_attn} pat={pat}", curve[-1])
    torch.cuda.synchronize(); train_s = time.time() - t0
    res = dict(pattern=pat, params=m.n_params(), steps=STEPS, train_s=train_s, tok_s=STEPS * B * CTX / train_s,
               peak_mem_mb=torch.cuda.max_memory_allocated() / 1e6, val_bpb=eval_len(m, CTX))
    for L in (2048, 4096): res[f"val_bpb_len{L}"] = eval_len(m, L, nwin=32)
    return res

for n_attn in (2, 3, 4):
    k = f"ratio|{n_attn}"
    if k not in R: R[k] = train_lm(n_attn); save(); log("DONE", k, {a: b for a, b in R[k].items() if a != "curve"})
log("ALL DONE")
