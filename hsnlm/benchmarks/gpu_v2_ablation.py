"""DEEPA v2 bonus ablation (NOT part of the locked v1 result set -- v1 is untouched by this,
every new knob is an opt-in Config field defaulting to exact v1 behavior, verified bit-exact
step()==forward() for both the eager and fused mixer before this script was trusted, see the
conversation for the sanity-check transcript). Tests the three genuinely zero-/near-zero-risk
items from improvements-scope.md, one at a time plus combined, against the v1 winner (ALiBi-
hybrid 3:1, gpu_multiseed_compiled.py's "baseline" entry is reused rather than rerun):
  h6        -- SheafDeltaMixer uses mixer_heads=6 (dh=64) instead of attention's heads=8 (dh=48);
               decoupled via Config.mixer_heads so the 2 ALiBi attention layers are NOT changed
               (a confound in the original proposal, which assumed a single shared `heads` field).
  decay     -- multi-scale decay-bias init (linspace(-1.0, 4.5, H)) instead of flat 3.0.
  postnorm  -- SheafDeltaMixer's onorm applied after summing delta+dyadic instead of before (uncompensated,
               later found to be a regression caused by chunk_gated_delta_rule's implicit dh**-0.5 readout
               scale -- see DEEPA_V2_ABLATION.md).
  combined  -- all three together.
Round 2 (layered on top of h6, the only round-1 win, per the corrected proposals):
  h6_postnorm_scaled -- postnorm with the dh**-0.5 scale compensated first (Config.postnorm_scaled).
  h6_token_shift     -- causal token-shift on q,k before F.normalize, learnable per-head alpha, with a
                        1-token raw-q/k state added to .step() (Config.token_shift).
  h6_both            -- both together.
Same 14.5M-param/width-384/8-layer/ctx-1024/dddaddda recipe as gpu_multiseed_compiled.py, 3
seeds, compiled, full eval (nwin=64 @1024, nwin=32 @2048/@4096), so directly comparable to the
"alibi-hybrid" row there."""
import sys, os, copy, json, time, math, torch, torch.nn as nn, torch.nn.functional as F; sys.path.insert(0, ".")
from hsnlm import Config, ByteData
from hsnlm.models import RMSNorm, MLP, LM, HBlock
assert torch.cuda.is_available()
torch.backends.cudnn.benchmark = True
DIM, HEADS, LAYERS, CTX, B, VOCAB = 384, 8, 8, 1024, 16, 256
LR = 0.002
STEPS = 1500
PATTERN = "dddaddda"
SEEDS = (0, 1, 2)
VARIANTS = {
    "h6":                dict(mixer_heads=6),
    "decay":             dict(decay_init="multiscale"),
    "postnorm":          dict(postnorm=True),
    "combined":          dict(mixer_heads=6, decay_init="multiscale", postnorm=True),
    # round 2, layered on top of h6 (the only round-1 winner), per the corrected/new proposals:
    "h6_postnorm_scaled": dict(mixer_heads=6, postnorm_scaled=True),  # fixes postnorm's dh**-0.5 amplitude mismatch
    "h6_token_shift":     dict(mixer_heads=6, token_shift=True),      # causal shift on q,k before F.normalize (item 4)
    "h6_both":            dict(mixer_heads=6, postnorm_scaled=True, token_shift=True),
}
OUT = "results/gpu_v2_ablation.json"; os.makedirs("results", exist_ok=True)
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

def build(variant):
    kw = VARIANTS[variant]
    c = Config(vocab=VOCAB, dim=DIM, layers=LAYERS, ctx=CTX, heads=HEADS, mlp_mult=3.65, pattern=PATTERN, **kw)
    return AlibiHybridLM(c).cuda()

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

def train_lm(variant, lr, steps, seed):
    torch.manual_seed(seed); gen = torch.Generator().manual_seed(seed); m0 = build(variant)
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
            torch.cuda.synchronize(); log(variant, "seed", seed, s + 1, round(loss.item() / math.log(2), 4), round(time.time() - t0, 1))
    torch.cuda.synchronize(); train_s = time.time() - t0
    res = dict(variant=variant, seed=seed, params=m0.n_params(), steps=steps, train_s=train_s, tok_s=steps * B * CTX / train_s,
               peak_mem_mb=torch.cuda.max_memory_allocated() / 1e6,
               val_bpb=eval_len(m0, CTX, nwin=64))
    for L in (2048, 4096): res[f"val_bpb_len{L}"] = eval_len(m0, L, nwin=32)
    return res

ORDER = sys.argv[1:] if len(sys.argv) > 1 else list(VARIANTS)
for variant in ORDER:
    for seed in SEEDS:
        k = f"{variant}|seed{seed}"
        if k not in R:
            R[k] = train_lm(variant, LR, STEPS, seed); save(); log("DONE", k, {a: b for a, b in R[k].items() if a != "curve"})
log("ALL DONE")
