"""Full CPU benchmark suite. Resumable: results/cpu_suite.json holds finished runs; rerun skips them.
Stages: sweep (lr per arch) -> lm seed0 -> synthetic (MQAR, parity) -> lm seed1.   usage: cpu_suite.py [smoke]"""
import sys, os, json, time, math, torch, torch.nn.functional as F; sys.path.insert(0, ".")
from hsnlm import GPT, HybridLM, Config, ByteData
torch.set_num_threads(8)
SMOKE = len(sys.argv) > 1 and sys.argv[1] == "smoke"
OUT = "results/cpu_suite_smoke.json" if SMOKE else "results/cpu_suite.json"; os.makedirs("results", exist_ok=True)
R = json.load(open(OUT)) if os.path.exists(OUT) else {}
def save(): json.dump(R, open(OUT, "w"), indent=1)
def log(*a): print(time.strftime("%H:%M:%S"), *a, flush=True)
S = (lambda n: 3) if SMOKE else (lambda n: n)

def build(arch, vocab, dim=128, layers=4, ctx=256, heads=4):
    if arch == "gpt": return GPT(Config(vocab=vocab, dim=dim, layers=layers, ctx=ctx, heads=heads, mlp_mult=4.0))
    if arch == "sheaf-delta": return HybridLM(Config(vocab=vocab, dim=dim, layers=layers, ctx=ctx, heads=heads, mlp_mult=3.4, pattern="d" * layers))
    if arch == "hybrid": return HybridLM(Config(vocab=vocab, dim=dim, layers=layers, ctx=ctx, heads=heads, mlp_mult=3.65, pattern="ddad" if layers == 4 else "da"))
ARCHS = ["gpt", "sheaf-delta", "hybrid"]
data = ByteData("data/enwik8", ctx=256, n_train=2_000_000 if SMOKE else 8_000_000, n_val=300_000 if SMOKE else 1_000_000)

@torch.no_grad()
def eval_len(m, L, nwin=128, B=8):
    m.eval(); v = data.val; n = min(nwin, (len(v) - 1) // L); tot = cnt = 0
    for i in range(0, n, B):
        xs = torch.stack([v[j * L:(j + 1) * L] for j in range(i, min(i + B, n))]); ys = torch.stack([v[j * L + 1:(j + 1) * L + 1] for j in range(i, min(i + B, n))])
        tot += F.cross_entropy(m(xs).reshape(-1, 256), ys.reshape(-1), reduction="sum").item(); cnt += ys.numel()
    m.train(); return tot / cnt / math.log(2)

def train_lm(arch, lr, steps, seed, B=16, final_eval=False):
    torch.manual_seed(seed); gen = torch.Generator().manual_seed(seed); m = build(arch, 256)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.05); t0 = time.time(); curve = []
    for s in range(steps):
        cur = lr * min(1, (s + 1) / 50) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * s / steps)))
        for g in opt.param_groups: g["lr"] = cur
        x, y = data.batch(B, gen); opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(m(x).reshape(-1, 256), y.reshape(-1)); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        if (s + 1) % max(steps // 4, 1) == 0: curve.append((s + 1, round(loss.item() / math.log(2), 4), round(time.time() - t0))); log(arch, "lr", lr, "seed", seed, curve[-1])
    train_s = time.time() - t0; res = dict(params=m.n_params(), steps=steps, train_s=train_s, tok_s=steps * B * 256 / train_s, curve=curve, val_bpb=eval_len(m, 256))
    if final_eval:
        for L in (512, 1024, 2048): res[f"val_bpb_len{L}"] = eval_len(m, L)
    return res

# ---------- stage 1: LR sweep ----------
for arch in ARCHS:
    for lr in (1e-3, 2e-3, 4e-3):
        k = f"sweep|{arch}|{lr}"
        if k not in R: R[k] = train_lm(arch, lr, S(200), seed=100); save()
best = {a: min((1e-3, 2e-3, 4e-3), key=lambda lr: R.get(f"sweep|{a}|{lr}", {"val_bpb": 9})["val_bpb"]) for a in ARCHS}; log("best lr", best); R["best_lr"] = best; save()
def lm_stage(seed):
    for arch in ARCHS:
        k = f"lm|{arch}|seed{seed}"
        if k not in R: R[k] = train_lm(arch, best[arch], S(2500), seed, final_eval=True); save(); log("DONE", k, {a: b for a, b in R[k].items() if a != "curve"})
lm_stage(0); log("STAGE lm seed0 done")

# ---------- stage 3: synthetic capability suite ----------
def mqar_batch(B, NP, NK=64):
    keys = torch.stack([torch.randperm(NK)[:NP] for _ in range(B)]); vals = torch.randint(NK, 2 * NK, (B, NP)); perm = torch.stack([torch.randperm(NP) for _ in range(B)])
    qk, qv = keys.gather(1, perm), vals.gather(1, perm)
    seq = torch.cat([torch.stack([keys, vals], 2).reshape(B, 2 * NP), torch.stack([qk, qv], 2).reshape(B, 2 * NP)], 1)
    mk = torch.zeros(B, 4 * NP - 1, dtype=torch.bool); mk[:, 2 * NP:][:, 0::2] = True; return seq[:, :-1], seq[:, 1:], mk
def run_synth(task, arch, steps, **kw):
    torch.manual_seed(0)
    if task == "mqar": NP = kw["NP"]; m = build(arch, 128, dim=64, layers=2, ctx=4 * NP); bf = lambda B: mqar_batch(B, NP)
    else: m = build(arch, 2, dim=64, layers=2, ctx=32); bf = None
    opt = torch.optim.AdamW(m.parameters(), 2e-3 if task == "mqar" else 3e-3, weight_decay=0.0); t0 = time.time()
    for s in range(steps):
        for g in opt.param_groups: g["lr"] = (2e-3 if task == "mqar" else 3e-3) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * s / steps)))
        if task == "mqar": x, y, mk = bf(32); lg = m(x); loss = F.cross_entropy(lg[mk], y[mk])
        else: x = torch.randint(0, 2, (64, 32)); y = x.cumsum(1) % 2; loss = F.cross_entropy(m(x).reshape(-1, 2), y.reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        if task == "mqar": x, y, mk = bf(512); return dict(acc=(m(x).argmax(-1)[mk] == y[mk]).float().mean().item(), train_s=time.time() - t0)
        out = {}
        for L in (32, 64, 128):
            x = torch.randint(0, 2, (256, L)); out[f"acc_len{L}"] = (m(x).argmax(-1) == x.cumsum(1) % 2).float().mean().item()
        out["train_s"] = time.time() - t0; return out
for NP in (4, 8, 16):
    for arch in ARCHS:
        k = f"mqar|{arch}|pairs{NP}"
        if k not in R: R[k] = run_synth("mqar", arch, S(2500), NP=NP); save(); log("DONE", k, R[k])
for arch in ARCHS:
    k = f"parity|{arch}"
    if k not in R: R[k] = run_synth("parity", arch, S(2500)); save(); log("DONE", k, R[k])
log("STAGE synthetic done")
lm_stage(1); log("STAGE lm seed1 done  ALL DONE")
