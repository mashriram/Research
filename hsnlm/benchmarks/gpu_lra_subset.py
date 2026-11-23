"""B5 -- Long Range Arena subset (Tay et al. 2021), causal formulation: each example is left-padded, followed by a QUERY
token, and the model must emit the class token at the query position (loss + accuracy only there). Vocab 268 =
256 data bytes + QUERY(256) + 10 class tokens (257..266) + PAD(267). Same architectures as the paper, at LRA scale:
dim 256, 4 layers ("ddda" for the hybrid), attn heads 4, mixer heads 4 (head dim 64). ALiBi via the exact fused path.
  listops  ListOps (Nangia & Bowman 2018; LRA generator settings: depth <= 10, <= 10 args, lengths 500-2000), one byte per
           token ([MAX=A [MIN=I [MED=D [SM=S, ']'=], digits '0'-'9'), 10 classes. 40k train / 2k test, generated (seed 0).
  text     IMDb sentiment (stanfordnlp/imdb), raw UTF-8 bytes truncated to the LAST 4,000 bytes, 2 classes. 25k / 25k (eval
           on a fixed 5k subset of test).
  image    sequential CIFAR-10 (grayscale, 1,024 pixels, raster order), 10 classes. 50k / 10k (eval on all 10k).
5,000 steps, B=32, AdamW lr 1e-3 (the common LRA setting; identical for every model), wd 0.05, warmup 200 + cosine, bf16,
seeds 0 and 1. Reports test accuracy and training throughput."""
import sys, os, json, math, time, random, io, tarfile, torch, torch.nn.functional as F; sys.path.insert(0, ".")
import numpy as np
import benchmarks.deepa_common as C
from benchmarks.deepa_common import build, log
C.ALIBI_IMPL = "flash"
assert torch.cuda.is_available(); DEV = "cuda"
QUERY, CLS0, PAD, V = 256, 257, 267, 268
ARCHS = ("alibi-hybrid-h6", "alibi-pure", "rope-gpt")
CFG = dict(dim=256, layers=4, heads=4, mixer_heads=4, vocab=V)
STEPS, B, LR, SEEDS = 5000, 32, 1e-3, (0, 1)
MICRO = {"text": 8}  # micro-batch for gradient accumulation (effective batch stays B=32)
OUT = "benchmark_results/paper/long_range_arena/gpu_lra_subset.json"; os.makedirs("benchmark_results", exist_ok=True); os.makedirs("data/lra", exist_ok=True)
R = json.load(open(OUT)) if os.path.exists(OUT) else {}
def save(): json.dump(R, open(OUT, "w"), indent=1)

# ---------------- data (cached as .pt: list-of-uint8 arrays + labels) ----------------
def listops(n, seed):
    rng = random.Random(seed); OPS = "AIDS"
    def tree(depth):
        if depth >= 10 or rng.random() > 0.25: v = rng.randint(0, 9); return [str(v)], v  # LRA: VALUE_P=0.25, leaf when r > 0.25
        op = rng.choice(OPS); toks = ["[" + op]; vals = []
        for _ in range(rng.randint(2, 10)):
            t, v = tree(depth + 1); toks += t; vals.append(v)
        toks.append("]")
        r = max(vals) if op == "A" else min(vals) if op == "I" else int(np.median(vals)) if op == "D" else sum(vals) % 10
        return toks, r
    out = []
    while len(out) < n:
        t, v = tree(0)
        s = "".join(x[1] if x.startswith("[") else x for x in t)  # one byte per token
        if 500 <= len(s) <= 2000: out.append((np.frombuffer(s.encode(), dtype=np.uint8), v))
    return out
def get(task):
    p = f"data/lra/{task}.pt"
    if os.path.exists(p): return torch.load(p, weights_only=False)
    if task == "listops": d = dict(train=listops(40000, 0), test=listops(2000, 1))
    elif task == "text":
        import pyarrow.parquet as pq
        from huggingface_hub import hf_hub_download
        def load(split):
            t = pq.read_table(hf_hub_download("stanfordnlp/imdb", f"plain_text/{split}-00000-of-00001.parquet", repo_type="dataset", local_dir="data/raw_domains/imdb"))
            return [(np.frombuffer(x.encode("utf-8")[-4000:], dtype=np.uint8), int(y)) for x, y in zip(t.column("text").to_pylist(), t.column("label").to_pylist())]
        te = load("test"); random.Random(0).shuffle(te); d = dict(train=load("train"), test=te[:5000])
    elif task == "image":
        tf = "data/raw_domains/image/cifar-10-binary.tar.gz"
        def load(files):
            out = []
            with tarfile.open(tf, "r:gz") as t:
                for fn in files:
                    raw = np.frombuffer(t.extractfile(f"cifar-10-batches-bin/{fn}").read(), dtype=np.uint8).reshape(-1, 3073)
                    rgb = raw[:, 1:].reshape(-1, 3, 1024).astype(np.float32)
                    g = (0.299 * rgb[:, 0] + 0.587 * rgb[:, 1] + 0.114 * rgb[:, 2] + 0.5).astype(np.uint8)
                    out += [(g[i], int(raw[i, 0])) for i in range(len(raw))]
            return out
        d = dict(train=load([f"data_batch_{i}.bin" for i in range(1, 6)]), test=load(["test_batch.bin"]))
    torch.save(d, p); return d

def collate(items):
    T = max(len(a) for a, _ in items) + 1
    x = torch.full((len(items), T), PAD, dtype=torch.long)
    for i, (a, _) in enumerate(items): x[i, T - 1 - len(a):T - 1] = torch.from_numpy(a.astype(np.int64))
    x[:, -1] = QUERY; y = torch.tensor([CLS0 + c for _, c in items])
    return x.to(DEV), y.to(DEV)

def forward_cls(m, x):
    with torch.autocast("cuda", dtype=torch.bfloat16): return m(x)[:, -1].float()

@torch.no_grad()
def evaluate(m, test, bs=32):
    m.eval(); ok = 0
    for i in range(0, len(test), bs):
        x, y = collate(test[i:i + bs]); ok += (forward_cls(m, x)[:, CLS0:CLS0 + 10].argmax(-1) + CLS0 == y).sum().item()
    m.train(); return ok / len(test)

def run(task, arch, seed):
    d = get(task); torch.manual_seed(seed); rng = random.Random(seed)
    m = build(arch, **CFG); opt = torch.optim.AdamW(m.parameters(), lr=LR, betas=(0.9, 0.95), weight_decay=0.05)
    torch.cuda.reset_peak_memory_stats(); t0 = time.time(); ntok = 0
    for s in range(STEPS):
        cur = LR * min(1, (s + 1) / 200) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * s / STEPS)))
        for pg in opt.param_groups: pg["lr"] = cur
        items = rng.sample(d["train"], B); opt.zero_grad(set_to_none=True); mb = MICRO.get(task, B); loss = 0.0
        for i in range(0, B, mb):  # gradient accumulation: identical to one batch of B (mean loss), fits 8 GB at 4k tokens
            x, y = collate(items[i:i + mb]); ntok += x.numel()
            l = F.cross_entropy(forward_cls(m, x), y) * (len(y) / B); l.backward(); loss = loss + l.detach()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        if (s + 1) % 1000 == 0: log(task, arch, seed, s + 1, round(loss.item(), 4))
    torch.cuda.synchronize(); tr = time.time() - t0
    res = dict(task=task, arch=arch, seed=seed, params=m.n_params(), train_s=tr, tok_s=ntok / tr,
               peak_mem_mb=torch.cuda.max_memory_allocated() / 1e6, test_acc=evaluate(m, d["test"], bs=MICRO.get(task, 32)))
    del m, opt; torch.cuda.empty_cache(); return res

ONLY = sys.argv[1:] or ["listops", "text", "image"]
for task in ONLY:
    get(task)
    for seed in SEEDS:
        for arch in ARCHS:
            k = f"{task}|{arch}|seed{seed}"
            if k not in R: R[k] = run(task, arch, seed); save(); log("DONE", k, round(R[k]["test_acc"], 4), round(R[k]["tok_s"]))
log("ALL DONE")
