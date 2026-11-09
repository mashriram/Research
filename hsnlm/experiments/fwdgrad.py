"""How good is a forward-mode (JVP) gradient estimate g = mean_k (grad.v_k) v_k vs the true gradient? cos ~ sqrt(k/d)."""
import sys, torch, torch.nn.functional as F; sys.path.insert(0, ".")
from torch.func import functional_call, jvp
from hsnlm import HSNLM, Config
torch.manual_seed(0); c = Config(layers=1, dim=64, ctx=32, vocab=65); m = HSNLM(c); x = torch.randint(0, 65, (16, 32)); y = torch.randint(0, 65, (16, 32))
params = {k: v.detach() for k, v in m.named_parameters()}; names = list(params)
loss_fn = lambda p: F.cross_entropy(functional_call(m, p, (x,)).reshape(-1, 65), y.reshape(-1))
pp = {k: v.clone().requires_grad_() for k, v in params.items()}; loss_fn(pp).backward()
true = torch.cat([pp[k].grad.flatten() for k in names]); d = true.numel(); print("params d =", d)
for k in (1, 8, 64, 512):
    est = torch.zeros(d)
    for _ in range(k):
        v = {n: torch.randn_like(params[n]) for n in names}
        _, dd = jvp(loss_fn, (params,), (v,)); est += dd * torch.cat([v[n].flatten() for n in names]) / k
    cos = F.cosine_similarity(est, true, dim=0).item(); print(f"k={k:4d} tangents  cos(est,true)={cos:.4f}   sqrt(k/d)={(k/d)**.5:.4f}   forward cost ~{k}x")
