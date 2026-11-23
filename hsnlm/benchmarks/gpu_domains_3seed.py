"""B3 -- multi-modal / multi-domain byte-level LM: the paper's 3-seed, 1,500-step compiled comparison repeated on six
non-enwik8 domains (code, Hindi, Chinese, DNA, images, audio; data/domain_<name>.bytes, built by
benchmarks/data_prep/build_domains.py). Identical models/recipe to gpu_wikitext103_3seed.py, except ALiBi attention uses the
exact fused-kernel path (deepa_common.ALIBI_IMPL="flash"; validated to 4e-5 bpb of the dense-mask path) for BOTH ALiBi models.
For image bytes, bits/byte == bits/dim (the standard image-likelihood metric); for 8-bit mu-law audio it is bits/sample.
Eval: val_bpb = 64 windows @1024 (repo convention), val_bpb_full = all 976 windows @1024, plus @2048/@4096 (32 windows).
Usage: python benchmarks/gpu_domains_3seed.py [domain ...] [--kinds=a,b]   (default: all six domains, the 3 paper models)"""
import sys, os, time
while os.path.exists("benchmark_results/HOLD_GPU"): time.sleep(30)  # manual pause gate: rm benchmark_results/HOLD_GPU to resume the queue
import json, torch; sys.path.insert(0, ".")
from hsnlm import ByteData
import benchmarks.deepa_common as C
from benchmarks.deepa_common import build, train_lm, eval_len, log
C.ALIBI_IMPL = "flash"
assert torch.cuda.is_available()
torch.backends.cudnn.benchmark = True
KINDS = ("alibi-hybrid-h6", "alibi-pure", "rope-gpt")
KINDS = tuple(a.split("=", 1)[1].split(",") for a in sys.argv[1:] if a.startswith("--kinds="))[0] if any(a.startswith("--kinds=") for a in sys.argv[1:]) else KINDS  # e.g. --kinds=sheafdelta-pure
DOMAINS = [a for a in sys.argv[1:] if not a.startswith("-")] or ["code", "hindi", "zh", "dna", "image", "audio"]
OUT = "benchmark_results/paper/multidomain_3seed/gpu_domains_3seed.json"; os.makedirs("benchmark_results", exist_ok=True)
R = json.load(open(OUT)) if os.path.exists(OUT) else {}
def save(): json.dump(R, open(OUT, "w"), indent=1)
R["_meta"] = dict(steps=1500, lr=0.002, batch=16, ctx=1024, wd=0.05, compiled=True, alibi_impl="flash (exact)", seeds=[0, 1, 2])
for dom in DOMAINS:
    path = f"data/domain_{dom}.bytes"
    if not os.path.exists(path): log("missing", path); continue
    data = ByteData(path, ctx=1024, n_train=40_000_000, n_val=1_000_000)
    for seed in (0, 1, 2):
        for kind in KINDS:
            k = f"{dom}|{kind}|seed{seed}"
            if k in R: continue
            torch.manual_seed(seed); m0 = build(kind)
            res = dict(domain=dom, kind=kind, seed=seed, **train_lm(m0, data, 1500, seed, tag=k))
            res["val_bpb"], _ = eval_len(m0, data.val, 1024, nwin=64)
            for L in (2048, 4096): res[f"val_bpb_len{L}"], _ = eval_len(m0, data.val, L, nwin=32)
            res["val_bpb_full"], res["full_windows"] = eval_len(m0, data.val, 1024, nwin=10 ** 9)
            R[k] = res; save(); log("DONE", k, {a: round(b, 4) for a, b in res.items() if a.startswith("val_bpb") or a == "tok_s"})
            del m0; torch.cuda.empty_cache(); torch._dynamo.reset()
log("ALL DONE")
