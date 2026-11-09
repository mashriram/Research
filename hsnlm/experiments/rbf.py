import sys, time, torch, torch.nn.functional as F; sys.path.insert(0, ".")
from hsnlm import GPT, HSNLM, Config
from hsnlm.reversible import RevLM
torch.set_num_threads(8)
def saved(fn):
    t = [0]
    def pk(x): t[0] += x.numel() * x.element_size(); return x
    with torch.autograd.graph.saved_tensors_hooks(pk, lambda x: x): out = fn()
    return t[0] / 1e6, out
def bench(m, x, **kw):
    loss = lambda: F.cross_entropy(m(x, **kw).reshape(-1, 65), x.reshape(-1))
    loss().backward(); best = 9e9
    for _ in range(3): m.zero_grad(); t = time.time(); mb, o = saved(loss); o.backward(); best = min(best, time.time() - t)
    return best * 1000, mb
print("N | model | params(M) | ms/step (8192 tok) | saved activation MB")
for N in (512, 2048):
    B = 8192 // N; x = torch.randint(0, 65, (B, N))
    rows = []
    m = GPT(Config(ctx=N, dim=128, layers=4)); rows.append(("GPT (Transformer)", m, {}))
    m = HSNLM(Config(ctx=N, dim=128, layers=4, mixer="sheafdelta", mlp_mult=3.5)); rows.append(("HSN-fast plain autograd", m, {}))
    c = Config(ctx=N, dim=112, layers=4, mixer="sheafdelta", mlp_mult=3.5, heads=4)
    r = RevLM(c); r.n_rev = 0; rows.append(("RBlockFast n_rev=0 (store all)", r, {}))
    r2 = RevLM(c); r2.n_rev = 2; rows.append(("RBlockFast n_rev=2 (policy)", r2, {}))
    r3 = RevLM(c); r3.n_rev = 4; rows.append(("RBlockFast n_rev=4 (all reversible)", r3, {}))
    for name, m, kw in rows:
        ms, mb = bench(m, x); print(N, "|", name, f"| {sum(p.numel() for p in m.parameters())/1e6:.2f} | {ms:.0f} | {mb:.0f}", flush=True)
