"""Reviewer item 7 (length extrapolation): trains alibi-hybrid-h6, alibi-pure and rope-gpt at ctx 1024 on enwik8
(seed 0, 1,500 steps, the standard compiled recipe), saves the weights, then evaluates each model zero-shot at
L = 1024, 2048, 4096, 8192, 16384 (up to 16x the training length).
Fair-budget eval: every length is scored on the SAME number of validation bytes (524,288 = 512 x 1024 = 32 x 16384),
all taken from the start of the standard enwik8 validation slice (bytes 40M..41M). Batch 8 @<=2048, 4 @4096,
1 @8192/16384 (fits 8 GB). Also records per-position bits/byte at L=16384 in 1024-token buckets, which shows WHERE in a
long window each model degrades (positions > 1024 were never seen in training).
ALiBi at L > 4096 uses the exact query-chunked path in deepa_common.py (dense bias would need 8.6 GB at 16k)."""
import sys, os, json, math, torch, torch.nn.functional as F; sys.path.insert(0, ".")
from hsnlm import ByteData
from benchmarks.deepa_common import build, train_lm, eval_len, log, check_alibi_chunked, VOCAB
assert torch.cuda.is_available()
torch.backends.cudnn.benchmark = True
KINDS = ("alibi-hybrid-h6", "alibi-pure", "rope-gpt")
LENS = (1024, 2048, 4096, 8192, 16384)
EVAL_BYTES = 524_288
BV = {1024: 8, 2048: 8, 4096: 4, 8192: 1, 16384: 1}
CK = "checkpoints/extrap16k"; os.makedirs(CK, exist_ok=True)
OUT = "benchmark_results/paper/length_extrapolation_16k/gpu_extrap_16k.json"; os.makedirs("benchmark_results", exist_ok=True)
R = json.load(open(OUT)) if os.path.exists(OUT) else {}
def save(): json.dump(R, open(OUT, "w"), indent=1)

data = ByteData("data/enwik8", ctx=1024, n_train=40_000_000, n_val=1_000_000)
R["_meta"] = dict(train="enwik8 40M bytes, seed 0, 1500 steps, ctx 1024, B 16, lr 2e-3, compiled", eval_bytes_per_len=EVAL_BYTES,
                  batch_val=BV, alibi_chunk_check_maxabs={str(k): v for k, v in check_alibi_chunked().items()})

@torch.no_grad()
def position_profile(m, L=16384, nwin=EVAL_BYTES // 16384, bucket=1024):
    m.eval(); v = data.val; tot = torch.zeros(L, dtype=torch.float64)
    for j in range(nwin):
        x = v[j * L:(j + 1) * L][None].cuda(); y = v[j * L + 1:(j + 1) * L + 1][None].cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            tot += F.cross_entropy(m(x).reshape(-1, VOCAB).float(), y.reshape(-1), reduction="none").double().cpu()
    m.train(); per = tot / nwin / math.log(2)
    return [round(per[i:i + bucket].mean().item(), 4) for i in range(0, L, bucket)]

SEEDS = tuple(int(a) for a in sys.argv[1:]) or (0,)
for seed, kind in [(sd, kd) for sd in SEEDS for kd in KINDS]:
    k = f"{kind}|seed{seed}"
    if k in R and "position_bpb_16k" in R[k]: continue
    path = f"{CK}/{kind}_seed{seed}.pt"
    torch.manual_seed(seed); m0 = build(kind)
    if os.path.exists(path):
        m0.load_state_dict(torch.load(path, map_location="cuda")); res = R.get(k, {}); log("loaded", path)
    else:
        res = dict(kind=kind, seed=seed, **train_lm(m0, data, 1500, seed, tag=k)); torch.save(m0.state_dict(), path)
        torch._dynamo.reset()
    for L in LENS:
        if f"val_bpb_len{L}" in res: continue
        torch.cuda.reset_peak_memory_stats()
        res[f"val_bpb_len{L}"], n = eval_len(m0, data.val, L, nwin=EVAL_BYTES // L, Bv=BV[L])
        res[f"eval_peak_mem_mb_len{L}"] = torch.cuda.max_memory_allocated() / 1e6
        log(k, "L", L, "windows", n, "bpb", round(res[f"val_bpb_len{L}"], 4)); R[k] = res; save()
    res["position_bpb_16k"] = position_profile(m0); R[k] = res; save()
    log("DONE", k, {a: b for a, b in res.items() if a != "curve"})
    del m0; torch.cuda.empty_cache()
log("ALL DONE")
