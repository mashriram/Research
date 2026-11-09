"""Exactly-reversible stack: y1 = x1 + F(x2); y2 = x2 + G(y1). Only the FINAL output is stored; inputs of every layer are
reconstructed in the backward pass => activation memory O(1) in depth, gradients exact (up to fp round-off), +~33% compute."""
import torch, torch.nn as nn, torch.nn.functional as F
from .models import Config, RMSNorm, MLP, make_mixer

class RevBlock(nn.Module):
    def __init__(s, c):
        super().__init__(); s.nf = RMSNorm(c.dim); s.ng = RMSNorm(c.dim)
        s.f = make_mixer(c); s.g = MLP(c.dim, c.mlp_mult, True)
    def F(s, x): return s.f(s.nf(x))
    def G(s, x): return s.g(s.ng(x))

class _RevStack(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x1, x2, blocks):
        ctx.blocks = blocks
        with torch.no_grad():
            for b in blocks: x1 = x1 + b.F(x2); x2 = x2 + b.G(x1)
        ctx.save_for_backward(x1.detach(), x2.detach()); return x1, x2
    @staticmethod
    def backward(ctx, d1, d2):
        y1, y2 = ctx.saved_tensors
        for b in reversed(ctx.blocks):
            with torch.enable_grad():
                y1 = y1.detach().requires_grad_(); g = b.G(y1); g.backward(d2)
            with torch.no_grad(): x2 = y2 - g; d1 = d1 + y1.grad
            with torch.enable_grad():
                x2 = x2.detach().requires_grad_(); f = b.F(x2); f.backward(d1)
            with torch.no_grad(): y1 = y1 - f; d2 = d2 + x2.grad; y2 = x2
        return d1, d2, None

class RevLM(nn.Module):
    """dim in cfg = per-stream width d; model width = 2d."""
    def __init__(s, c: Config):
        super().__init__(); s.c = c; s.embed = nn.Embedding(c.vocab, 2 * c.dim); s.blocks = nn.ModuleList(RevBlock(c) for _ in range(c.layers))
        s.norm_f = RMSNorm(2 * c.dim); nn.init.normal_(s.embed.weight, std=0.02)
        for m in s.modules():
            if isinstance(m, nn.Linear): nn.init.normal_(m.weight, std=0.02)
    def head(s, h): return F.linear(s.norm_f(h), s.embed.weight)
    n_rev = None  # how many (leading) layers are reversible; None = all. Others use ordinary autograd storage.
    def forward(s, x, reversible=True):
        n = (len(s.blocks) if s.n_rev is None else s.n_rev) if reversible else 0
        x1, x2 = s.embed(x).chunk(2, -1)
        if n: x1, x2 = _RevStack.apply(x1, x2.contiguous(), list(s.blocks[:n]))
        for b in list(s.blocks)[n:]: x1 = x1 + b.F(x2); x2 = x2 + b.G(x1)
        return s.head(torch.cat([x1, x2], -1))
    def fit_memory(s, x, budget_bytes):
        """RBlockFast policy: reverse (pay +1 fwd/layer) only as many layers as needed to fit the activation budget."""
        s.n_rev = 0; tot = [0]
        def pack(t): tot[0] += t.numel() * t.element_size(); return t
        with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t): s(x)
        per = tot[0] / len(s.blocks); s.n_rev = min(len(s.blocks), max(0, -(-int(tot[0] - budget_bytes) // int(per)))); return s.n_rev
    def n_params(s): return sum(p.numel() for p in s.parameters())
