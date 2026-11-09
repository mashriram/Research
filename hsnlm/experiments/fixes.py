import sys, time, torch, torch.nn.functional as F; sys.path.insert(0, ".")
from hsnlm import HSNLM, GPT, Config
from hsnlm.reversible import RevLM
torch.set_num_threads(8)
def saved_bytes(fn):
    tot = [0]
    def pack(t): tot[0] += t.numel() * t.element_size(); return t
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t): out = fn()
    return tot[0], out
part = sys.argv[1]
if part == "rev":
    torch.manual_seed(0); x = torch.randint(0, 65, (8, 64)); y = torch.randint(0, 65, (8, 64))
    for mixer in ("sheaf", "delta"):
        c = Config(layers=4, dim=48, ctx=64, mixer=mixer); m = RevLM(c)
        gs = {}
        for rev in (False, True):
            m.zero_grad(); F.cross_entropy(m(x, rev).reshape(-1, 65), y.reshape(-1)).backward()
            gs[rev] = torch.cat([p.grad.flatten() for p in m.parameters()])
        print(mixer, "max grad diff rev vs autograd:", (gs[True] - gs[False]).abs().max().item(), "scale", gs[False].abs().max().item(), flush=True)
    print("L | autograd saved MB | reversible saved MB | time fwd+bwd s (std / rev)")
    for L in (4, 8, 16, 32):
        c = Config(layers=L, dim=48, ctx=64); m = RevLM(c); r = {}
        for rev in (False, True):
            m.zero_grad(); t = time.time()
            b, out = saved_bytes(lambda: F.cross_entropy(m(x, rev).reshape(-1, 65), y.reshape(-1))); out.backward(); r[rev] = (b / 1e6, time.time() - t)
        print(L, f"{r[False][0]:.1f} | {r[True][0]:.2f} | {r[False][1]:.2f} / {r[True][1]:.2f}", flush=True)
if part == "scale":
    print("N | GPT fwd+bwd ms/1k tok | HSN fwd+bwd ms/1k tok | GPT saved MB | HSN saved MB   (8192 tokens/step)")
    for N in (256, 512, 1024, 2048):
        B = 8192 // N; x = torch.randint(0, 65, (B, N)); row = []
        for M in (GPT, HSNLM):
            m = M(Config(ctx=N, layers=4)); m.zero_grad()
            F.cross_entropy(m(x).reshape(-1, 65), x.reshape(-1)).backward()
            t = time.time(); b, out = saved_bytes(lambda: F.cross_entropy(m(x).reshape(-1, 65), x.reshape(-1))); out.backward()
            row += [(time.time() - t) / 8.192 * 1000 / 1000, b / 1e6]
        print(N, f"{row[0]*1000:.0f} | {row[2]*1000:.0f} | {row[1]:.0f} | {row[3]:.0f}", flush=True)
