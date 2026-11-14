"""Tests the throughput PRIMITIVE behind block/self-speculative decoding (RESEARCH_V2.md section 5),
without implementing a full draft-and-verify pipeline or an actual trained diffusion draft model: does
processing a block of b positions in ONE forward pass (inference only, no backward -- this is what a
'verify this drafted block' or 'decode a block of masked positions' pass would cost) get cheaper per
token as b grows, and does that amortization differ between GPT and the sheaf-delta architectures?
This is honest about what it does NOT test: no draft model, no acceptance rate, no actual speculative
decoding loop -- just the raw cost-per-token of processing b positions at once vs one at a time,
which is the mechanism block/speculative decoding would need to pay off."""
import sys, os, copy, json, time, torch; sys.path.insert(0, ".")
from hsnlm import GPT, HSNLM, Config
from hsnlm.models import LM, HBlock, TBlock
assert torch.cuda.is_available()
torch.backends.cudnn.benchmark = True
DIM, HEADS, LAYERS, VOCAB = 1024, 16, 8, 256
BLOCKS = (1, 8, 32, 128, 512, 2048)
OUT = "results/gpu_blockdecode.json"; os.makedirs("results", exist_ok=True)
R = json.load(open(OUT)) if os.path.exists(OUT) else {}
def save(): json.dump(R, open(OUT, "w"), indent=1)
def log(*a): print(time.strftime("%H:%M:%S"), *a, flush=True)

class HybridFusedLM(LM):
    use_pos = False
    def make_block(s, i):
        c = copy.copy(s.c); ch = (s.c.pattern or "d" * s.c.layers)[i]
        if ch == "a": return TBlock(c)
        c.mixer = "sheafdelta-fused"; return HBlock(c)

def build(arch):
    if arch == "gpt": return GPT(Config(vocab=VOCAB, dim=DIM, layers=LAYERS, ctx=4096, heads=HEADS, mlp_mult=4.0)).cuda().eval()
    if arch == "hsn": return HSNLM(Config(vocab=VOCAB, dim=DIM, layers=LAYERS, ctx=4096, heads=HEADS, mlp_mult=3.4, mixer="sheafdelta-fused")).cuda().eval()
    return HybridFusedLM(Config(vocab=VOCAB, dim=DIM, layers=LAYERS, ctx=4096, heads=HEADS, mlp_mult=3.65, pattern="dddaddda")).cuda().eval()

@torch.no_grad()
def bench(arch, b):
    m = build(arch); x = torch.randint(0, VOCAB, (1, b), device="cuda")
    def step(): m(x)
    for _ in range(5): step()
    torch.cuda.synchronize(); times = []
    for _ in range(10):
        torch.cuda.synchronize(); t0 = time.time(); step(); torch.cuda.synchronize(); times.append(time.time() - t0)
    times.sort(); ms = times[5]
    return dict(block_ms=round(ms * 1000, 3), ms_per_token=round(ms * 1000 / b, 4), tok_s=round(b / ms))

for arch in ("gpt", "hsn", "hybrid"):
    for b in BLOCKS:
        k = f"{arch}|{b}"
        if k in R: continue
        try:
            R[k] = bench(arch, b)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache(); R[k] = dict(oom=True)
        save(); log(k, R[k])
log("ALL DONE")
