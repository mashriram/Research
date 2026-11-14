"""GPU port of experiments/rbf.py: reversible-block memory/speed tradeoff (RBlockFast policy) vs
plain autograd, at context lengths the CPU run couldn't reach cheaply. CUDA-synced timing + memory."""
import sys, time, torch, torch.nn.functional as F; sys.path.insert(0, ".")
from hsnlm import GPT, HSNLM, Config
from hsnlm.reversible import RevLM
def saved(fn):
    t = [0]
    def pk(x): t[0] += x.numel() * x.element_size(); return x
    with torch.autograd.graph.saved_tensors_hooks(pk, lambda x: x): out = fn()
    return t[0] / 1e6, out
def bench(m, x, **kw):
    m = m.cuda(); x = x.cuda()
    loss = lambda: F.cross_entropy(m(x, **kw).reshape(-1, 65), x.reshape(-1))
    loss().backward(); torch.cuda.synchronize(); best = 9e9; peak = 0
    for _ in range(3):
        m.zero_grad(); torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); t0 = time.time()
        mb, o = saved(loss); o.backward(); torch.cuda.synchronize()
        best = min(best, time.time() - t0); peak = max(peak, torch.cuda.max_memory_allocated() / 1e6)
    return best * 1000, mb, peak
print("N | model | params(M) | ms/step (8192 tok) | saved activation MB | peak alloc MB")
for N in (512, 2048, 8192):
    B = max(1, 8192 // N); x = torch.randint(0, 65, (B, N))
    rows = []
    m = GPT(Config(ctx=N, dim=128, layers=4)); rows.append(("GPT (Transformer)", m, {}))
    m = HSNLM(Config(ctx=N, dim=128, layers=4, mixer="sheafdelta", mlp_mult=3.5)); rows.append(("HSN-fast plain autograd", m, {}))
    m = HSNLM(Config(ctx=N, dim=128, layers=4, mixer="sheafdelta-fused", mlp_mult=3.5)); rows.append(("HSN-fused plain autograd", m, {}))
    c = Config(ctx=N, dim=112, layers=4, mixer="sheafdelta", mlp_mult=3.5, heads=4)
    r = RevLM(c); r.n_rev = 0; rows.append(("RBlockFast n_rev=0 (store all)", r, {}))
    r2 = RevLM(c); r2.n_rev = 2; rows.append(("RBlockFast n_rev=2 (policy)", r2, {}))
    r3 = RevLM(c); r3.n_rev = 4; rows.append(("RBlockFast n_rev=4 (all reversible)", r3, {}))
    for name, m, kw in rows:
        try:
            ms, mb, peak = bench(m, x, **kw)
            print(N, "|", name, f"| {sum(p.numel() for p in m.parameters())/1e6:.2f} | {ms:.0f} | {mb:.0f} | {peak:.0f}", flush=True)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache(); print(N, "|", name, "| OOM", flush=True)
