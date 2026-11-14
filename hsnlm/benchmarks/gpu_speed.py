"""GPU training-speed/memory scaling vs context length: Transformer (fused SDPA) vs HSN eager (python chunk loop)
vs HSN fused (fla Triton chunk_gated_delta_rule). Answers G1 from findings_report.md: does a fused kernel
close/flip the CPU-measured 2-3x training-speed gap, and does the O(N) vs O(N^2) crossover appear at longer context?
Fixed batch B=16; context N swept; forward+backward timed after warmup, 3 best-of-5 iters. fp32 and bf16-autocast both measured.
usage: gpu_speed.py [smoke]"""
import sys, os, json, time, torch, torch.nn.functional as F; sys.path.insert(0, ".")
from hsnlm import GPT, HSNLM, Config
assert torch.cuda.is_available()
SMOKE = len(sys.argv) > 1 and sys.argv[1] == "smoke"
DIM, HEADS, LAYERS, VOCAB, B = 256, 8, 4, 256, 16
NS = (256, 512, 1024) if SMOKE else (256, 512, 1024, 2048, 4096, 8192)
OUT = "results/gpu_speed.json"; os.makedirs("results", exist_ok=True)
R = json.load(open(OUT)) if os.path.exists(OUT) else {}
def save(): json.dump(R, open(OUT, "w"), indent=1)
def log(*a): print(time.strftime("%H:%M:%S"), *a, flush=True)

def build(arch, N):
    c = Config(vocab=VOCAB, dim=DIM, layers=LAYERS, ctx=N, heads=HEADS)
    if arch == "gpt": c.mlp_mult = 4.0; return GPT(c).cuda()
    c.mlp_mult = 3.4
    c.mixer = "sheafdelta" if arch == "hsn-eager" else "sheafdelta-fused"
    return HSNLM(c).cuda()

def bench(arch, N, dtype):
    try:
        m = build(arch, N); x = torch.randint(0, VOCAB, (B, N), device="cuda"); y = torch.randint(0, VOCAB, (B, N), device="cuda")
        def step():
            m.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(dtype == "bf16")):
                loss = F.cross_entropy(m(x).reshape(-1, VOCAB), y.reshape(-1))
            loss.backward(); return loss
        for _ in range(3): step()  # warmup (also triggers cudnn.benchmark / triton autotune)
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        times = []
        for _ in range(5):
            torch.cuda.synchronize(); t0 = time.time(); step(); torch.cuda.synchronize(); times.append(time.time() - t0)
        times.sort()
        return dict(params=m.n_params(), ms=round(min(times) * 1000, 2), median_ms=round(times[2] * 1000, 2),
                    peak_mem_mb=round(torch.cuda.max_memory_allocated() / 1e6, 1), tok_s=round(B * N / min(times)))
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache(); return dict(oom=True)

torch.backends.cudnn.benchmark = True
for dtype in ("fp32", "bf16"):
    for N in NS:
        for arch in ("gpt", "hsn-eager", "hsn-fused"):
            k = f"{arch}|{dtype}|{N}"
            if k in R: continue
            torch.cuda.empty_cache(); r = bench(arch, N, dtype); R[k] = r; save(); log(k, r)
log("ALL DONE")
