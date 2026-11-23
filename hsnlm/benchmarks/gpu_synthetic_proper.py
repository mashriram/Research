"""B2-proper -- synthetic capability suite, run to convergence (replaces the 3,000-step screening in gpu_synthetic_suite.py,
whose MQAR / S5 numbers were under-trained: Zoology-style MQAR needs O(10^5) steps, and S5 is normally learned with a
length curriculum).

Protocol (identical for every architecture):
  * models: dim 128, 4 layers, attn heads 4, mixer heads 2 (same as the screening suite), vocab 256, bf16, torch.compile.
  * budget: up to MAX_STEPS (default 20,000) steps, B=64, AdamW(0.9,0.95) wd 0.05, warmup 200 + cosine over MAX_STEPS.
  * LR grid {3e-4, 1e-3, 3e-3} on seed 0; the best LR (final acc, then fewest steps-to-solve) is rerun on seeds 1 and 2.
  * eval every 500 steps on a fixed held-out set at the training length (1,024 sequences); "solved" = acc >= 0.99;
    runs stop early after two consecutive solved evals (so cost tracks difficulty). We log the whole accuracy curve and
    steps_to_solve (sample efficiency), then a final held-out eval (2,048 sequences) at every test length.
  * state-tracking tasks use a length curriculum 8 -> 16 -> 32 -> 64 (advance when the current length is solved);
    a run that never reaches length 64 is reported with the stage it reached.
Tasks:
  mqar8_T64    sanity / easy recall: 8 pairs in 64 tokens.          mqar32_T512, mqar64_T512: Zoology-scale recall.
  parity       Z2 running parity, train 64, test 64 / 256.
  s3           S3 (6 elements, non-abelian, solvable) running product, train 64, test 64 / 256.
  s5           S5 (120 elements, non-solvable, NC1-complete) running product, train 64, test 64 / 256.
  copy_paper   copy again, at the paper model size (dim 384, 8 layers, 2 attention layers in the hybrid).
  copy         copy a random 128-token string after a separator, test 128 / 512.  selcopy: selective copying, 16 of 256.
Usage: python benchmarks/gpu_synthetic_proper.py [task ...]      env: PROPER_MAX_STEPS, PROPER_SMOKE=1"""
import sys, os, json, math, time, itertools, torch, torch.nn.functional as F; sys.path.insert(0, ".")
import benchmarks.deepa_common as C
from benchmarks.deepa_common import build, log
C.ALIBI_IMPL = "flash"
torch._dynamo.config.cache_size_limit = 64  # curriculum lengths x train/eval modes x test lengths exceed the default 8 (speed only)
assert torch.cuda.is_available(); DEV = "cuda"
SMOKE = os.environ.get("PROPER_SMOKE") == "1"
ARCHS = tuple(os.environ.get("PROPER_ARCHS", "alibi-hybrid-h6,alibi-pure,rope-gpt,sheafdelta-pure").split(","))
CFG = dict(dim=128, layers=4, heads=4, mixer_heads=2)
MAX_STEPS = int(os.environ.get("PROPER_MAX_STEPS", 60 if SMOKE else 20000))
EVAL_EVERY = 20 if SMOKE else 500
B, LRS, SEEDS, SOLVED = 64, (3e-4, 1e-3, 3e-3), (0, 1, 2), 0.99
OUT = "benchmark_results/paper/synthetic_converged/gpu_synthetic_proper_smoke.json" if SMOKE else os.environ.get("PROPER_OUT", "benchmark_results/paper/synthetic_converged/gpu_synthetic_proper.json")
os.makedirs("benchmark_results", exist_ok=True)
R = json.load(open(OUT)) if os.path.exists(OUT) else {}
def save(): json.dump(R, open(OUT, "w"), indent=1)

# ---------------- generators: x [B,T] long, y [B,T] long (-100 = ignored) ----------------
def gen_mqar(Bn, g, n, T):  # keys 0..63, values 64..127, noise 128..255; query = key token, target = its value
    keys = torch.argsort(torch.rand(Bn, 64, generator=g, device=DEV), 1)[:, :n]
    vals = torch.randint(64, 128, (Bn, n), generator=g, device=DEV)
    x = torch.randint(128, 256, (Bn, T), generator=g, device=DEV); y = torch.full((Bn, T), -100, device=DEV)
    x[:, 0:2 * n:2] = keys; x[:, 1:2 * n:2] = vals
    slots = 2 * n + 2 * torch.argsort(torch.rand(Bn, (T - 2 * n) // 2, generator=g, device=DEV), 1)[:, :n]
    qi = torch.argsort(torch.rand(Bn, n, generator=g, device=DEV), 1)
    ar = torch.arange(Bn, device=DEV)[:, None]
    x[ar, slots] = keys.gather(1, qi); y[ar, slots] = vals.gather(1, qi)
    return x, y

def group_table(perms):
    idx = {p: i for i, p in enumerate(perms)}
    return torch.tensor([[idx[tuple(a[b[i]] for i in range(len(a)))] for b in perms] for a in perms], device=DEV)
_TAB = {}
def gen_group(Bn, g, T, name):  # running product of random group elements; every position is a target
    if name not in _TAB: _TAB[name] = group_table(list(itertools.permutations(range(3 if name == "s3" else 5))))
    mul = _TAB[name]; x = torch.randint(0, mul.shape[0], (Bn, T), generator=g, device=DEV); y = torch.empty_like(x)
    acc = x[:, 0].clone(); y[:, 0] = acc
    for t in range(1, T): acc = mul[acc, x[:, t]]; y[:, t] = acc
    return x, y
def gen_copy(Bn, g, T):  # T = string length L; random string, separator, repeat; predict the repeat
    s_ = torch.randint(0, 254, (Bn, T), generator=g, device=DEV); sep = torch.full((Bn, 1), 255, device=DEV)
    x = torch.cat([s_, sep, s_], 1)[:, :-1]; y = torch.full_like(x, -100); y[:, T:] = s_; return x, y
def gen_selcopy(Bn, g, T, n=16):  # 16 data tokens at random positions in T noise tokens; after markers, emit them in order
    x = torch.zeros((Bn, T + n), dtype=torch.long, device=DEV); y = torch.full_like(x, -100)
    pos = torch.sort(torch.argsort(torch.rand(Bn, T, generator=g, device=DEV), 1)[:, :n], 1).values
    d = torch.randint(2, 256, (Bn, n), generator=g, device=DEV); ar = torch.arange(Bn, device=DEV)[:, None]
    x[ar, pos] = d; x[:, T:] = 1; y[:, T:] = d; return x, y
def gen_parity(Bn, g, T):
    x = torch.randint(0, 2, (Bn, T), generator=g, device=DEV); return x, x.cumsum(1) % 2

# name: (generator(Bn, g, T), curriculum lengths (last = train length), test lengths)
TASKS = {
    "mqar8_T64": (lambda Bn, g, T: gen_mqar(Bn, g, 8, T), [64], [64]),
    "mqar32_T512": (lambda Bn, g, T: gen_mqar(Bn, g, 32, T), [512], [512]),
    "mqar64_T512": (lambda Bn, g, T: gen_mqar(Bn, g, 64, T), [512], [512]),
    "parity": (gen_parity, [8, 16, 32, 64], [64, 256]),
    "s3": (lambda Bn, g, T: gen_group(Bn, g, T, "s3"), [8, 16, 32, 64], [64, 256]),
    "s5": (lambda Bn, g, T: gen_group(Bn, g, T, "s5"), [8, 16, 32, 64], [64, 256]),
    "copy": (gen_copy, [128], [128, 512]),
    "selcopy": (gen_selcopy, [256], [256]),
    "copy_paper": (gen_copy, [128], [128, 512]),  # same copy task on the PAPER model size (dim 384, 8 layers, "dddaddda")
}
TASK_CFG = {"copy_paper": dict(dim=384, layers=8, heads=8, mixer_heads=6)}

@torch.no_grad()
def evaluate(fwd, gen, T, n, seed=12345, bs=128):
    g = torch.Generator(device=DEV).manual_seed(seed + T); ok = tot = 0
    for _ in range(max(n // bs, 1)):
        x, y = gen(bs, g, T)
        with torch.autocast("cuda", dtype=torch.bfloat16): p = fwd(x).argmax(-1)
        k = y != -100; ok += (p[k] == y[k]).sum().item(); tot += k.sum().item()
    return ok / tot

def run(task, arch, lr, seed):
    gen, stages, tests = TASKS[task]
    torch.manual_seed(seed); m = build(arch, **TASK_CFG.get(task, CFG)); fwd = torch.compile(m, dynamic=False)
    g = torch.Generator(device=DEV).manual_seed(seed)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.05)
    st, curve, solved_at, streak, t0 = 0, [], None, 0, time.time()
    stage_steps = {}
    for s in range(MAX_STEPS):
        cur = lr * min(1, (s + 1) / 200) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * s / MAX_STEPS)))
        for pg in opt.param_groups: pg["lr"] = cur
        x, y = gen(B, g, stages[st]); opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = F.cross_entropy(fwd(x).reshape(-1, 256).float(), y.reshape(-1), ignore_index=-100)
        loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        if (s + 1) % EVAL_EVERY == 0:
            m.eval(); acc = evaluate(fwd, gen, stages[st], 1024); m.train()
            curve.append((s + 1, stages[st], round(loss.item(), 4), round(acc, 4)))
            if acc >= SOLVED:
                if st < len(stages) - 1: stage_steps[stages[st]] = s + 1; st += 1; streak = 0
                else:
                    if solved_at is None: solved_at = s + 1
                    streak += 1
                    if streak >= 2: break
            else: streak = 0
    m.eval()
    res = dict(task=task, arch=arch, lr=lr, seed=seed, params=m.n_params(), steps_run=s + 1, train_s=time.time() - t0,
               final_loss=loss.item(), reached_len=stages[st], stage_solved_at=stage_steps, steps_to_solve=solved_at, curve=curve)
    for T in tests: res[f"acc_T{T}"] = evaluate(fwd, gen, T, 2048, seed=999)
    del m, fwd, opt; torch.cuda.empty_cache(); torch._dynamo.reset(); return res

def score(r):  # higher is better: final train-length acc, then fewer steps to solve
    T = TASKS[r["task"]][1][-1]
    return (r[f"acc_T{T}"] if r["reached_len"] == T else 0.0, -(r["steps_to_solve"] or 10 ** 9))

ONLY = [a for a in sys.argv[1:] if a in TASKS] or ["copy_paper", "copy"] + [t for t in TASKS if t not in ("copy_paper", "copy")]
if os.environ.get("PROPER_FULLGRID") == "1":
    # LR-robustness pass: every LR x seeds 1,2 for every arch (seed 0 = the existing sweep run), so the LR can be chosen by the
    # 3-SEED MEAN instead of seed 0 alone. Same rule for every arch. Skipped: arch/task pairs where no LR solved seed 0
    # (more seeds cannot change a 0/3 outcome). Keys: grid|task|arch|lr|seed{s}.
    for task in ONLY:
        for arch in ARCHS:
            sw = [R.get(f"sweep|{task}|{arch}|{lr}") for lr in LRS]
            if not any(r and r["steps_to_solve"] for r in sw): log("skip (unsolved at every LR on seed 0)", task, arch); continue
            for lr in LRS:
                for seed in (1, 2):
                    k = f"grid|{task}|{arch}|{lr}|seed{seed}"
                    if k in R: continue
                    if lr == R.get(f"best_lr|{task}|{arch}") and f"{task}|{arch}|seed{seed}" in R: R[k] = R[f"{task}|{arch}|seed{seed}"]
                    else: R[k] = run(task, arch, lr, seed)
                    save(); log("GRID", k, {a: b for a, b in R[k].items() if a.startswith("acc") or a == "steps_to_solve"})
    log("ALL DONE"); sys.exit(0)
R["_meta"] = dict(max_steps=MAX_STEPS, eval_every=EVAL_EVERY, batch=B, lrs=LRS, seeds=SEEDS, solved=SOLVED, cfg=CFG)
for task in ONLY:
    for arch in ARCHS:
        for lr in (LRS[:1] if SMOKE else LRS):
            k = f"sweep|{task}|{arch}|{lr}"
            if k not in R:
                R[k] = run(task, arch, lr, 0); save()
                log(k, {a: b for a, b in R[k].items() if a.startswith("acc") or a in ("steps_to_solve", "reached_len", "steps_run")})
        if SMOKE: continue
        best = max(LRS, key=lambda l: score(R[f"sweep|{task}|{arch}|{l}"])); R[f"best_lr|{task}|{arch}"] = best
        for seed in SEEDS:
            k = f"{task}|{arch}|seed{seed}"
            if k in R: continue
            R[k] = R[f"sweep|{task}|{arch}|{best}"] if seed == 0 else run(task, arch, best, seed); save()
            log("DONE", k, {a: b for a, b in R[k].items() if a.startswith("acc") or a in ("steps_to_solve", "reached_len")})
log("ALL DONE")
