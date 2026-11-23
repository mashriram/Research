"""Reviewer item 4 (additional dataset): repeats the 3-seed, 1,500-step, compiled comparison on WikiText-103 (raw,
byte-level) instead of enwik8. Same three models and exact same recipe as gpu_splitwd_3seed.py /
gpu_multiseed_compiled.py (uniform-WD AdamW, lr 2e-3, B=16, ctx 1024, bf16, torch.compile).
Data: data/wikitext103_raw.bytes = first 40M bytes of the WikiText-103-raw train split + first 1M bytes of the
official validation split (benchmarks/data_prep/build_wikitext103_bytes.py), i.e. the same 40M/1M sizes as enwik8.
Eval: repo convention (val_bpb = 64 windows @1024, 32 windows @2048/@4096) plus val_bpb_full = all 976 windows
@1024 of the 1M-byte validation slice."""
import sys, os, json, torch; sys.path.insert(0, ".")
from hsnlm import ByteData
from benchmarks.deepa_common import build, train_lm, eval_len, log
assert torch.cuda.is_available()
torch.backends.cudnn.benchmark = True
KINDS = ("alibi-hybrid-h6", "alibi-pure", "rope-gpt")
SEEDS = tuple(int(a) for a in sys.argv[1:]) or (0, 1, 2)
OUT = "benchmark_results/paper/wikitext103_3seed/gpu_wikitext103_3seed.json"; os.makedirs("benchmark_results", exist_ok=True)
R = json.load(open(OUT)) if os.path.exists(OUT) else {}
def save(): json.dump(R, open(OUT, "w"), indent=1)

data = ByteData("data/wikitext103_raw.bytes", ctx=1024, n_train=40_000_000, n_val=1_000_000)
assert len(data.train) == 40_000_000 and len(data.val) == 1_000_000
R["_meta"] = dict(dataset="WikiText-103 raw (byte-level)", n_train=40_000_000, n_val=1_000_000, steps=1500, lr=0.002,
                  batch=16, ctx=1024, wd=0.05, compiled=True, eval="val_bpb: 64 win @1024; len2048/4096: 32 win; full: all win @1024")
for seed in SEEDS:
    for kind in KINDS:
        k = f"{kind}|seed{seed}"
        if k in R: continue
        torch.manual_seed(seed); m0 = build(kind)
        res = dict(kind=kind, seed=seed, **train_lm(m0, data, 1500, seed, tag=k))
        res["val_bpb"], _ = eval_len(m0, data.val, 1024, nwin=64)
        for L in (2048, 4096): res[f"val_bpb_len{L}"], _ = eval_len(m0, data.val, L, nwin=32)
        res["val_bpb_full"], res["full_windows"] = eval_len(m0, data.val, 1024, nwin=10 ** 9)
        R[k] = res; save(); log("DONE", k, {a: b for a, b in res.items() if a != "curve"})
        del m0; torch.cuda.empty_cache(); torch._dynamo.reset()
log("ALL DONE")
