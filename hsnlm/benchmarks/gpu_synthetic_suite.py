"""B2 -- synthetic capability suite: tests WHAT each architecture can do, independent of data/tokenisation.
Small models (dim 128, 4 layers, attn heads 4, mixer heads 2 -> mixer head dim 64), vocab 256, compiled-off eager, bf16.
Architectures: alibi-hybrid-h6 ("ddda": 3 SheafDelta + 1 ALiBi), alibi-pure, rope-gpt, sheafdelta-pure (no attention).
ALiBi layers use the exact fused-kernel path (deepa_common.ALIBI_IMPL="flash").
Tasks (loss/accuracy only on target positions):
  mqar{n}   multi-query associative recall (Zoology, Arora et al. 2023): n key-value pairs, then queries scattered among noise;
            predict each queried key's value. T=512. n in {32, 64}. Tests in-context recall / memory capacity.
  s5        S5 permutation word problem: input = sequence of permutations of 5 items, target = running composition.
            NC1-complete; provably beyond fixed-depth attention and diagonal SSMs, solvable by delta rules whose
            transition can have negative eigenvalues (Grazzi et al. 2024) -- our beta in [0,2]. Train T=64, test 64 & 256.
  parity    running parity of a bit string. Train T=64, test 64 & 256 (length generalisation).
  copy      induction/copy: random string of L tokens, separator, repeat it; predict the repeat. Train L=128, test L=128 & 512.
  selcopy   selective copying (Mamba, Gu & Dao 2023): 16 data tokens at random positions in T=256 of noise; after a marker,
            output them in order. Tests content-aware filtering.
Protocol: per (task, arch) LR picked from {1e-3, 3e-3} on seed 0, then 3 seeds (0,1,2) at that LR, 3,000 steps, B=64,
AdamW(0.9,0.95) wd 0.05 (wd 0.0 on nothing -- same uniform recipe as the paper), warmup 100 + cosine. Fresh random batches every
step (infinite data), fixed held-out eval batches (seeded) at each test length: 2,048 sequences."""
import sys, os, json, math, time, torch, torch.nn.functional as F; sys.path.insert(0, ".")
import benchmarks.deepa_common as C
from benchmarks.deepa_common import build, log
C.ALIBI_IMPL = "flash"
assert torch.cuda.is_available()
DEV = "cuda"
ARCHS = ("alibi-hybrid-h6", "alibi-pure", "rope-gpt", "sheafdelta-pure")
CFG = dict(dim=128, layers=4, heads=4, mixer_heads=2)
STEPS, B, SEEDS = int(os.environ.get("SYN_STEPS", 3000)), 64, (0, 1, 2)
LRS = tuple(float(a) for a in os.environ.get("SYN_LRS", "1e-3,3e-3").split(","))
OUT = os.environ.get("SYN_OUT", "benchmark_results/paper/synthetic_screening_3k/gpu_synthetic_suite.json"); os.makedirs("benchmark_results", exist_ok=True)
R = json.load(open(OUT)) if os.path.exists(OUT) else {}
def save(): json.dump(R, open(OUT, "w"), indent=1)

# ---------------- task generators: return x [B,T] long, y [B,T] long (-100 = ignored) ----------------
def gen_mqar(Bn, g, n, T=512):
    K, V0, NOISE = 64, 64, 128  # keys 0..63, values 64..127, noise 128..255
    keys = torch.argsort(torch.rand(Bn, K, generator=g, device=DEV), 1)[:, :n]
    vals = torch.randint(V0, V0 + 64, (Bn, n), generator=g, device=DEV)
    x = torch.randint(NOISE, 256, (Bn, T), generator=g, device=DEV); y = torch.full((Bn, T), -100, device=DEV)
    x[:, 0:2 * n:2] = keys; x[:, 1:2 * n:2] = vals
    nq = n  # one query per key, at random distinct even slots in the query region
    slots = 2 * n + 2 * torch.argsort(torch.rand(Bn, (T - 2 * n) // 2, generator=g, device=DEV), 1)[:, :nq]
    qi = torch.argsort(torch.rand(Bn, n, generator=g, device=DEV), 1)[:, :nq]
    ar = torch.arange(Bn, device=DEV)[:, None]
    x[ar, slots] = keys.gather(1, qi); y[ar, slots] = vals.gather(1, qi)   # predict value from the query key position
    return x, y

_perms = None
def s5_tables():
    global _perms
    if _perms is None:
        import itertools
        P = list(itertools.permutations(range(5))); idx = {p: i for i, p in enumerate(P)}
        mul = torch.tensor([[idx[tuple(a[b[i]] for i in range(5))] for b in P] for a in P], device=DEV)
        _perms = mul
    return _perms
def gen_s5(Bn, g, T=64):
    mul = s5_tables(); x = torch.randint(0, 120, (Bn, T), generator=g, device=DEV); y = torch.empty_like(x)
    acc = x[:, 0].clone(); y[:, 0] = acc
    for t in range(1, T): acc = mul[acc, x[:, t]]; y[:, t] = acc
    return x, y
def gen_parity(Bn, g, T=64):
    x = torch.randint(0, 2, (Bn, T), generator=g, device=DEV); return x, x.cumsum(1) % 2
def gen_copy(Bn, g, L=128):
    s = torch.randint(0, 254, (Bn, L), generator=g, device=DEV); sep = torch.full((Bn, 1), 255, device=DEV)
    x = torch.cat([s, sep, s], 1)[:, :-1]; y = torch.full_like(x, -100); y[:, L:] = s   # at sep and after, predict next copy token
    return x, y
def gen_selcopy(Bn, g, T=256, n=16):
    NOISE, MARK = 0, 1  # data tokens 2..255
    x = torch.full((Bn, T + n), NOISE, device=DEV); y = torch.full_like(x, -100)
    pos = torch.sort(torch.argsort(torch.rand(Bn, T, generator=g, device=DEV), 1)[:, :n], 1).values
    d = torch.randint(2, 256, (Bn, n), generator=g, device=DEV); ar = torch.arange(Bn, device=DEV)[:, None]
    x[ar, pos] = d; x[:, T:] = MARK; y[:, T:] = d   # at the i-th marker, output the i-th data token
    return x, y

TASKS = {  # name: (generator, train kwargs, {eval name: kwargs})
    "mqar32": (gen_mqar, dict(n=32), {"T512": dict(n=32)}),
    "mqar64": (gen_mqar, dict(n=64), {"T512": dict(n=64)}),
    "s5": (gen_s5, dict(T=64), {"T64": dict(T=64), "T256": dict(T=256)}),
    "parity": (gen_parity, dict(T=64), {"T64": dict(T=64), "T256": dict(T=256)}),
    "copy": (gen_copy, dict(L=128), {"L128": dict(L=128), "L512": dict(L=512)}),
    "selcopy": (gen_selcopy, dict(), {"T256": dict()}),
}

@torch.no_grad()
def evaluate(m, gen, kw, n=2048, bs=128):
    m.eval(); g = torch.Generator(device=DEV).manual_seed(12345); ok = tot = 0
    for _ in range(n // bs):
        x, y = gen(bs, g, **kw)
        with torch.autocast("cuda", dtype=torch.bfloat16): p = m(x).argmax(-1)
        msk = y != -100; ok += (p[msk] == y[msk]).sum().item(); tot += msk.sum().item()
    m.train(); return ok / tot

def run(task, arch, lr, seed, steps=STEPS):
    gen, kw, evals = TASKS[task]
    torch.manual_seed(seed); m = build(arch, **CFG); g = torch.Generator(device=DEV).manual_seed(seed)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.05); t0 = time.time()
    for s in range(steps):
        cur = lr * min(1, (s + 1) / 100) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * s / steps)))
        for pg in opt.param_groups: pg["lr"] = cur
        x, y = gen(B, g, **kw); opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = F.cross_entropy(m(x).reshape(-1, 256).float(), y.reshape(-1), ignore_index=-100)
        loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
    res = dict(task=task, arch=arch, lr=lr, seed=seed, params=m.n_params(), final_loss=loss.item(), train_s=time.time() - t0)
    for en, ekw in evals.items(): res[f"acc_{en}"] = evaluate(m, gen, ekw)
    del m, opt; torch.cuda.empty_cache(); return res

ONLY = [a for a in sys.argv[1:] if a in TASKS] or list(TASKS)
for task in ONLY:
    for arch in ARCHS:
        for lr in LRS:
            k = f"sweep|{task}|{arch}|{lr}"
            if k not in R: R[k] = run(task, arch, lr, 0); save(); log(k, {a: b for a, b in R[k].items() if a.startswith("acc") or a == "final_loss"})
        first = list(TASKS[task][2])[0]
        best = max(LRS, key=lambda l: (R[f"sweep|{task}|{arch}|{l}"][f"acc_{first}"], -R[f"sweep|{task}|{arch}|{l}"]["final_loss"]))
        for seed in SEEDS:
            k = f"{task}|{arch}|seed{seed}"
            if k in R: continue
            R[k] = R[f"sweep|{task}|{arch}|{best}"] if seed == 0 else run(task, arch, best, seed)  # seed-0 sweep run IS the seed-0 run
            save(); log("DONE", k, {a: round(b, 4) for a, b in R[k].items() if a.startswith("acc")})
log("ALL DONE")
