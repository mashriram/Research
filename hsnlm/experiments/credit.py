"""Compare credit-assignment rules on the same HSN LM (no cross-block gradient chain except baseline)."""
import sys, json, time, math, copy, torch, torch.nn.functional as F; sys.path.insert(0, ".")
from hsnlm import HSNLM, Config, CharData, evaluate_bpc
torch.set_num_threads(8)
steps = int(sys.argv[1]); which = sys.argv[2].split(",")
data = CharData("data/shakespeare.txt", 128); cfg = Config(vocab=data.V, mlp_mult=3.5)
def head_ce(m, h, y): return F.cross_entropy(m.head(h).reshape(-1, cfg.vocab), y.reshape(-1))

def run(name, step_fn, model, params, B=32, lr=3e-3):
    gen = torch.Generator().manual_seed(0); opt = torch.optim.AdamW(params, lr=lr, betas=(0.9, .99), weight_decay=0.01); t0 = time.time()
    for s in range(steps):
        cur = lr * min(1, (s + 1) / 50) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * s / steps)))
        for g in opt.param_groups: g["lr"] = cur
        x, y = data.batch(B, gen); opt.zero_grad(set_to_none=True); step_fn(x, y)
        torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
        if (s + 1) % (steps // 4) == 0: print(name, s + 1, evaluate_bpc(model, data), flush=True)
    return dict(val_bpc=evaluate_bpc(model, data), train_s=time.time() - t0)

res = {}
if "teacher" in which:
    torch.manual_seed(0); T = HSNLM(cfg)
    def bp(x, y): head_ce(T, T.embed_in(x), y) if False else F.cross_entropy(T(x).reshape(-1, cfg.vocab), y.reshape(-1)).backward()
    res["backprop(teacher)"] = run("bp", bp, T, list(T.parameters())); torch.save(T.state_dict(), "experiments/teacher.pt")
if "dfa" in which:
    torch.manual_seed(0); M = HSNLM(cfg); R = [torch.randn(cfg.dim, cfg.dim) / cfg.dim ** .5 for _ in M.blocks]
    def dfa(x, y):
        h = M.embed_in(x).detach(); hs = []
        with torch.no_grad(): pass
        for b in M.blocks: h = b(h.detach()); hs.append(h)
        hl = hs[-1].detach().requires_grad_(); head_ce(M, hl, y).backward(); e = hl.grad * hl.numel() / cfg.dim  # dL/dh_L, rescaled
        for i, h in enumerate(hs[:-1]): h.backward(e @ R[i] / 1.0)   # random-feedback error, local VJP only
        hs[-1].backward(e)
    res["DFA(random feedback)"] = run("dfa", dfa, M, list(M.parameters()))
if "distill" in which:
    T = HSNLM(cfg); T.load_state_dict(torch.load("experiments/teacher.pt")); T.eval()
    torch.manual_seed(1); S = HSNLM(cfg)
    S.embed.weight.data.copy_(T.embed.weight); S.norm_f.load_state_dict(T.norm_f.state_dict())   # frozen shared interface
    params = [p for b in S.blocks for p in b.parameters()]
    def layerwise(x, y):
        with torch.no_grad():
            ts = [T.embed_in(x)]
            for b in T.blocks: ts.append(b(ts[-1]))
        for i, b in enumerate(S.blocks):  # each block: independent regression to teacher's hidden state, teacher input
            out = b(ts[i]); (F.mse_loss(out, ts[i + 1]) / ts[i + 1].var()).backward()
    res["layerwise teacher-target (no x-block grad)"] = run("lw", layerwise, S, params)
json.dump(res, open(f"experiments/credit_{sys.argv[3]}.json", "w"), indent=1); print(res)
