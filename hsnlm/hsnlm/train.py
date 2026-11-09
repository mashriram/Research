import time, math, torch, torch.nn.functional as F

def ce(model, h, y): return F.cross_entropy(model.head(h).reshape(-1, model.c.vocab), y.reshape(-1))

def local_step(model, x, y):
    """DAD-style backprop-free-across-depth step: every block is trained on its own local
    next-token loss with a detached input, graph freed immediately => O(1) activation memory in depth."""
    h = model.embed_in(x); total = 0.0
    for i, b in enumerate(model.blocks):
        hin = h if i == 0 else h.detach()
        h = b(hin); loss = ce(model, h, y); loss.backward(); total += loss.item()
    return loss.item()  # last block's loss

def global_step(model, x, y):
    loss = F.cross_entropy(model(x).reshape(-1, model.c.vocab), y.reshape(-1)); loss.backward(); return loss.item()

@torch.no_grad()
def evaluate_bpc(model, data, B=64, max_batches=40):
    model.eval(); tot = n = 0
    for i, (x, y) in enumerate(data.val_batches(B)):
        if i >= max_batches: break
        tot += F.cross_entropy(model(x).reshape(-1, model.c.vocab), y.reshape(-1), reduction="sum").item(); n += y.numel()
    model.train(); return tot / n / math.log(2)

def train(model, data, steps, mode="backprop", B=32, lr=3e-3, wd=0.01, log=100, seed=0):
    torch.manual_seed(seed); gen = torch.Generator().manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.99), weight_decay=wd)
    step_fn = global_step if mode == "backprop" else local_step
    hist, t0 = [], time.time()
    for s in range(steps):
        cur = lr * min(1, (s + 1) / 50) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * s / steps)))
        for g in opt.param_groups: g["lr"] = cur
        x, y = data.batch(B, gen); opt.zero_grad(set_to_none=True)
        loss = step_fn(model, x, y)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if (s + 1) % log == 0:
            hist.append((s + 1, loss, evaluate_bpc(model, data), time.time() - t0)); print(hist[-1], flush=True)
    return hist, time.time() - t0
