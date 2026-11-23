"""Reviewer item 6 (scalability): alibi-hybrid-h6 (3:1 SheafDelta:ALiBi) vs pure ALiBi at four sizes, each trained at a
matched 5 tokens/parameter (5 x the hybrid's param count; both models at a scale get the identical step count, so
they are compute/token-matched to within the hybrid's +1.2-1.4% params), seed 0, torch.compile.
   scale  dim layers attn_heads mixer_heads  pattern
   ~2M    192   4      4          3          ddda
   ~6M    256   8      4          4          dddaddda
   ~14M   384   8      8          6          dddaddda      (the paper's model)
   ~38M   512  12      8          8          dddadddaddda
Mixer head dim is 64 at every scale. 5 tok/param needs up to ~190M training bytes, which would be ~4.7 epochs of the
usual 40M-byte slice; to avoid a repeated-data confound the training set here is ALL of enwik8 except the standard
validation slice: bytes [0,40M) + [41M,100M) = 99M bytes (<= 1.95 epochs at the largest scale). Validation is the
SAME bytes [40M,41M) used everywhere else in the paper.
LR: 2e-3 (the tuned value at 14M) for the three smaller scales. At ~38M, each architecture picks its own LR from
{1e-3, 2e-3} by a 1,000-step sweep run (seed 100), so neither model is handicapped by a shared untuned LR.
Activation checkpointing is turned on automatically only if a B=16 step OOMs (recorded per run)."""
import sys, os, json, torch; sys.path.insert(0, ".")
from benchmarks.deepa_common import build, train_lm, eval_len, log
assert torch.cuda.is_available()
torch.backends.cudnn.benchmark = True
KINDS = ("alibi-hybrid-h6", "alibi-pure")
SCALES = {"2M": (192, 4, 4, 3), "6M": (256, 8, 4, 4), "14M": (384, 8, 8, 6), "38M": (512, 12, 8, 8)}
ONLY = sys.argv[1:] or list(SCALES)
TOK_PER_PARAM, B, CTX = 5, 16, 1024
OUT = "benchmark_results/paper/scaling_2m_to_38m/gpu_scaling_law.json"; os.makedirs("benchmark_results", exist_ok=True)
R = json.load(open(OUT)) if os.path.exists(OUT) else {}
def save(): json.dump(R, open(OUT, "w"), indent=1)

class SplitData:
    def __init__(s, path):
        raw = open(path, "rb").read(); assert len(raw) == 100_000_000
        ids = torch.frombuffer(bytearray(raw), dtype=torch.uint8).long()
        s.train = torch.cat([ids[:40_000_000], ids[41_000_000:]]); s.val = ids[40_000_000:41_000_000]; s.ctx = CTX
    def batch(s, Bn, gen=None):
        ix = torch.randint(len(s.train) - s.ctx - 1, (Bn,), generator=gen)
        return torch.stack([s.train[i:i + s.ctx] for i in ix]), torch.stack([s.train[i + 1:i + s.ctx + 1] for i in ix])
data = SplitData("data/enwik8")

def needs_ckpt(kind, cfg):
    torch._dynamo.reset(); torch.cuda.empty_cache(); m = build(kind, *cfg)
    try:
        x, y = data.batch(B, torch.Generator().manual_seed(1)); x, y = x.cuda(), y.cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = torch.nn.functional.cross_entropy(m(x).reshape(-1, 256), y.reshape(-1))
        loss.backward(); torch.cuda.synchronize(); ok = torch.cuda.max_memory_allocated() < 7.0e9
    except torch.OutOfMemoryError: ok = False
    del m; torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); return not ok

def run(kind, cfg, steps, lr, seed, tag, full_eval):
    ck = needs_ckpt(kind, cfg); torch._dynamo.reset()
    torch.manual_seed(seed); m0 = build(kind, *cfg, ckpt=ck)
    res = dict(kind=kind, seed=seed, grad_ckpt=ck, **train_lm(m0, data, steps, seed, lr=lr, B=B, ctx=CTX, tag=tag, log_every=max(steps // 10, 1)))
    res["val_bpb"], _ = eval_len(m0, data.val, 1024, nwin=64)
    if full_eval:
        for L in (2048, 4096): res[f"val_bpb_len{L}"], _ = eval_len(m0, data.val, L, nwin=32)
        res["val_bpb_full"], res["full_windows"] = eval_len(m0, data.val, 1024, nwin=10 ** 9)
    del m0; torch.cuda.empty_cache(); torch._dynamo.reset(); return res

R["_meta"] = dict(tok_per_param=TOK_PER_PARAM, batch=B, ctx=CTX, train_bytes=99_000_000, val="enwik8 [40M,41M)", scales=SCALES)
for name in ONLY:
    cfg = SCALES[name]
    n_hyb = build("alibi-hybrid-h6", *cfg).n_params(); steps = round(TOK_PER_PARAM * n_hyb / (B * CTX)); torch.cuda.empty_cache()
    for kind in KINDS:
        if name == "38M":
            for lr in (1e-3, 2e-3):
                sk = f"sweep|{name}|{kind}|{lr}"
                if sk not in R: R[sk] = run(kind, cfg, 1000, lr, 100, sk, False); save(); log("sweep", sk, R[sk]["val_bpb"])
            lr = min((1e-3, 2e-3), key=lambda l: R[f"sweep|{name}|{kind}|{l}"]["val_bpb"])
        else: lr = 2e-3
        k = f"{name}|{kind}"
        if k in R: continue
        R[k] = dict(scale=name, cfg=cfg, **run(kind, cfg, steps, lr, 0, k, True)); R[k]["tok_per_param"] = R[k]["tokens"] / R[k]["params"]
        save(); log("DONE", k, {a: b for a, b in R[k].items() if a != "curve"})
log("ALL DONE")
