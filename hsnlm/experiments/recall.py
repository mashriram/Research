"""Multi-query associative recall: k1 v1 ... k8 v8 | q1 ? q2 ? ... ; predict the value after each query key."""
import sys, time, torch, torch.nn.functional as F; sys.path.insert(0, ".")
from hsnlm import HSNLM, GPT, Config
torch.set_num_threads(8); torch.manual_seed(0)
NP, NK = int(sys.argv[2]), 32
def batch(B):
    keys = torch.stack([torch.randperm(NK)[:NP] for _ in range(B)]); vals = torch.randint(NK, 2 * NK, (B, NP))
    perm = torch.stack([torch.randperm(NP) for _ in range(B)])
    qk = keys.gather(1, perm); qv = vals.gather(1, perm)
    ctx = torch.stack([keys, vals], 2).reshape(B, 2 * NP); qs = torch.stack([qk, qv], 2).reshape(B, 2 * NP)
    seq = torch.cat([ctx, qs], 1); x = seq[:, :-1]; y = seq[:, 1:]
    mask = torch.zeros_like(y, dtype=torch.bool); mask[:, 2 * NP + 0::2][:, :NP] = False
    mask = torch.zeros(B, y.shape[1], dtype=torch.bool); mask[:, 2 * NP:][:, 0::2] = True  # positions whose target is the value
    return x, y, mask
cfgs = {"attention": (GPT, Config(layers=2, dim=64, ctx=4 * NP, mixer="sheaf")),
        "sheaf-diag (current HSN)": (HSNLM, Config(layers=2, dim=64, ctx=4 * NP, mixer="sheaf")),
        "householder-delta": (HSNLM, Config(layers=2, dim=64, ctx=4 * NP, mixer="delta"))}
for name in sys.argv[1].split(","):
    cls, c = cfgs[name]; c.vocab = 2 * NK; m = cls(c); opt = torch.optim.AdamW(m.parameters(), 3e-3, weight_decay=0.0); t0 = time.time()
    for s in range(int(sys.argv[3])):
        x, y, mk = batch(64); lg = m(x); loss = F.cross_entropy(lg[mk], y[mk]); opt.zero_grad(); loss.backward(); opt.step()
        if (s + 1) % 250 == 0:
            with torch.no_grad(): x, y, mk = batch(512); acc = (m(x).argmax(-1)[mk] == y[mk]).float().mean().item()
            print(f"{name} pairs={NP} step {s+1} loss {loss.item():.3f} recall-acc {acc:.3f} ({time.time()-t0:.0f}s)", flush=True)
