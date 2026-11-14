"""Isolated-process re-measurement of gpu_reconcile.py's hsn-fused and hybrid-fused main training runs,
to check whether their reported speed (50,979 / 94,346 tok/s) was deflated by CUDA-allocator pollution
from running gpt's sweep+training in the same long-lived process first (gpu_reconcile.py ran gpt, then
hsn-fused, then hybrid-fused, all sequentially in one process). Each arch here gets its own fresh
process (run this script once per arch via argv). Same recipe, same best_lr=0.002, same seed=0."""
import sys, os, copy, json, time, math, torch, torch.nn.functional as F; sys.path.insert(0, ".")
from hsnlm import GPT, Config, ByteData
from hsnlm.models import LM, HBlock, TBlock
assert torch.cuda.is_available()
torch.backends.cudnn.benchmark = True
DIM, HEADS, LAYERS, CTX, B, VOCAB = 128, 4, 8, 16384, 1, 256
STEPS = 3000
ARCH = sys.argv[1]

class HybridFusedLM(LM):
    use_pos = False
    def make_block(s, i):
        c = copy.copy(s.c); ch = (s.c.pattern or "d" * s.c.layers)[i]
        if ch == "a": return TBlock(c)
        c.mixer = "sheafdelta-fused"; return HBlock(c)

def build(arch):
    if arch == "gpt": return GPT(Config(vocab=VOCAB, dim=DIM, layers=LAYERS, ctx=CTX, heads=HEADS, mlp_mult=4.0)).cuda()
    if arch == "hsn-fused": from hsnlm import HSNLM; return HSNLM(Config(vocab=VOCAB, dim=DIM, layers=LAYERS, ctx=CTX, heads=HEADS, mlp_mult=3.4, mixer="sheafdelta-fused")).cuda()
    return HybridFusedLM(Config(vocab=VOCAB, dim=DIM, layers=LAYERS, ctx=CTX, heads=HEADS, mlp_mult=3.65, pattern="dddaddda")).cuda()

data = ByteData("data/enwik8", ctx=CTX, n_train=60_000_000, n_val=2_000_000)
lr = 0.002
torch.manual_seed(0); gen = torch.Generator().manual_seed(0)
m = build(ARCH)
opt = torch.optim.AdamW(m.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.05)
torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); t0 = time.time()
for s in range(STEPS):
    cur = lr * min(1, (s + 1) / 100) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * s / STEPS)))
    for g in opt.param_groups: g["lr"] = cur
    x, y = data.batch(B, gen); x, y = x.cuda(), y.cuda(); opt.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = F.cross_entropy(m(x).reshape(-1, VOCAB), y.reshape(-1))
    loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
    if (s + 1) % 300 == 0:
        torch.cuda.synchronize(); print(time.strftime("%H:%M:%S"), ARCH, s + 1, round(loss.item() / math.log(2), 4), round(time.time() - t0, 1), flush=True)
torch.cuda.synchronize(); train_s = time.time() - t0
print("RESULT", ARCH, dict(train_s=train_s, tok_s=STEPS * B * CTX / train_s, peak_mem_mb=torch.cuda.max_memory_allocated() / 1e6), flush=True)
