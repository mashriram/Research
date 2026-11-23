"""B4 -- is the WikiText-103 loss (gpu_wikitext103_3seed.json: pure ALiBi beat the hybrid by ~0.012 bpb on the full val set
at 1,500 steps / lr 2e-3) real, or an artefact of (a) a learning rate tuned on enwik8 or (b) a short training horizon?
(a) LR sweep for BOTH models: lr in {1e-3, 2e-3 (existing runs), 3e-3}, 1,500 steps, seed 0.
(b) Each model at its own best LR from (a), trained to 5 tokens/param (4,406 steps, the gpu_scaling_law.py horizon), seeds 0, 1.
Same data / eval as gpu_wikitext103_3seed.py. ALiBi uses the exact fused path for both models (deepa_common.ALIBI_IMPL)."""
import sys, os, json, torch; sys.path.insert(0, ".")
from hsnlm import ByteData
import benchmarks.deepa_common as C
from benchmarks.deepa_common import build, train_lm, eval_len, log
C.ALIBI_IMPL = "flash"
torch.backends.cudnn.benchmark = True
KINDS = ("alibi-hybrid-h6", "alibi-pure")
OUT = "benchmark_results/paper/wikitext103_lr_and_horizon/gpu_wikitext103_fair.json"; os.makedirs("benchmark_results", exist_ok=True)
R = json.load(open(OUT)) if os.path.exists(OUT) else {}
def save(): json.dump(R, open(OUT, "w"), indent=1)
W = json.load(open("benchmark_results/paper/wikitext103_3seed/gpu_wikitext103_3seed.json"))
data = ByteData("data/wikitext103_raw.bytes", ctx=1024, n_train=40_000_000, n_val=1_000_000)

def run(kind, lr, steps, seed, tag):
    torch.manual_seed(seed); m0 = build(kind)
    res = dict(kind=kind, seed=seed, **train_lm(m0, data, steps, seed, lr=lr, tag=tag, log_every=max(steps // 5, 1)))
    res["val_bpb"], _ = eval_len(m0, data.val, 1024, nwin=64)
    for L in (2048, 4096): res[f"val_bpb_len{L}"], _ = eval_len(m0, data.val, L, nwin=32)
    res["val_bpb_full"], res["full_windows"] = eval_len(m0, data.val, 1024, nwin=10 ** 9)
    del m0; torch.cuda.empty_cache(); torch._dynamo.reset(); return res

for kind in KINDS:
    R[f"sweep|{kind}|0.002"] = {a: b for a, b in W[f"{kind}|seed0"].items()}  # existing 1,500-step seed-0 run
    for lr in (1e-3, 3e-3):
        k = f"sweep|{kind}|{lr}"
        if k not in R: R[k] = run(kind, lr, 1500, 0, k); save(); log("DONE", k, round(R[k]["val_bpb_full"], 4))
    best = min((1e-3, 2e-3, 3e-3), key=lambda l: R[f"sweep|{kind}|{l}"]["val_bpb_full"]); R[f"best_lr|{kind}"] = best; save()
    for seed in (0, 1):
        k = f"long|{kind}|seed{seed}"
        if k not in R: R[k] = run(kind, best, 4406, seed, k); save(); log("DONE", k, {a: round(b, 4) for a, b in R[k].items() if a.startswith("val_bpb")})
log("ALL DONE")
