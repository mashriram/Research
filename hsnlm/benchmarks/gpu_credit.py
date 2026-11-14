"""GPU port of experiments/credit.py: confirms the CPU finding that no backprop-alternative credit-assignment
rule saves wall-clock (each block still does its own forward+local-backward). Char-level Shakespeare,
same 0.82M-param HSN (default 'sheaf' v1 mixer, matching the original CPU experiment), 800 steps, 1 seed."""
import sys, json, time, math, torch, torch.nn.functional as F; sys.path.insert(0, ".")
from hsnlm import HSNLM, Config, CharData
DEV = "cuda"; assert torch.cuda.is_available()
steps = 800
data = CharData("data/shakespeare.txt", 128); cfg = Config(vocab=data.V, mlp_mult=3.5)
def head_ce(m, h, y): return F.cross_entropy(m.head(h).reshape(-1, cfg.vocab), y.reshape(-1))
@torch.no_grad()
def evaluate_bpc(model, data, B=64, max_batches=40):
    model.eval(); tot = n = 0
    for i, (x, y) in enumerate(data.val_batches(B)):
        if i >= max_batches: break
        x, y = x.to(DEV), y.to(DEV)
        tot += F.cross_entropy(model(x).reshape(-1, cfg.vocab), y.reshape(-1), reduction="sum").item(); n += y.numel()
    model.train(); return tot / n / math.log(2)

def run(name, step_fn, model, params, B=32, lr=3e-3):
    gen = torch.Generator().manual_seed(0); opt = torch.optim.AdamW(params, lr=lr, betas=(0.9, .99), weight_decay=0.01)
    torch.cuda.synchronize(); t0 = time.time()
    for s in range(steps):
        cur = lr * min(1, (s + 1) / 50) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * s / steps)))
        for g in opt.param_groups: g["lr"] = cur
        x, y = data.batch(B, gen); x, y = x.to(DEV), y.to(DEV); opt.zero_grad(set_to_none=True); step_fn(x, y)
        torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
        if (s + 1) % (steps // 4) == 0:
            torch.cuda.synchronize(); print(name, s + 1, evaluate_bpc(model, data), flush=True)
    torch.cuda.synchronize(); return dict(val_bpc=evaluate_bpc(model, data), train_s=time.time() - t0)

res = {}
torch.manual_seed(0); T = HSNLM(cfg).to(DEV)
def bp(x, y): F.cross_entropy(T(x).reshape(-1, cfg.vocab), y.reshape(-1)).backward()
res["backprop(teacher)"] = run("bp", bp, T, list(T.parameters()))

torch.manual_seed(0); M = HSNLM(cfg).to(DEV); R = [torch.randn(cfg.dim, cfg.dim, device=DEV) / cfg.dim ** .5 for _ in M.blocks]
def dfa(x, y):
    h = M.embed_in(x).detach(); hs = []
    for b in M.blocks: h = b(h.detach()); hs.append(h)
    hl = hs[-1].detach().requires_grad_(); head_ce(M, hl, y).backward(); e = hl.grad * hl.numel() / cfg.dim
    for i, h in enumerate(hs[:-1]): h.backward(e @ R[i] / 1.0)
    hs[-1].backward(e)
res["DFA(random feedback)"] = run("dfa", dfa, M, list(M.parameters()))

torch.manual_seed(1); S = HSNLM(cfg).to(DEV)
S.embed.weight.data.copy_(T.embed.weight); S.norm_f.load_state_dict(T.norm_f.state_dict())
params = [p for b in S.blocks for p in b.parameters()]
def layerwise(x, y):
    with torch.no_grad():
        ts = [T.embed_in(x)]
        for b in T.blocks: ts.append(b(ts[-1]))
    for i, b in enumerate(S.blocks): out = b(ts[i]); (F.mse_loss(out, ts[i + 1]) / ts[i + 1].var()).backward()
res["layerwise teacher-target (no x-block grad)"] = run("lw", layerwise, S, params)

json.dump(res, open("results/gpu_credit.json", "w"), indent=1); print(res)
