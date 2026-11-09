"""Two LMs behind one interface:
  GPT    - standard pre-LN Transformer (baseline, KV-cache inference)
  HSNLM  - Householder Sheaf Network: causal dyadic sheaf diffusion with Householder
           (isometric) stalk transport + data-dependent gated linear recurrence.
           Training O(N log N) / inference O(1) state per token.
Both expose: embed, blocks, norm_f, tied head, forward(x)->logits,
             generate-step API (init_state/step) for fair inference benchmarks."""
import math
from dataclasses import dataclass
import torch, torch.nn as nn, torch.nn.functional as F

@dataclass
class Config:
    vocab: int = 65
    dim: int = 128
    layers: int = 4
    heads: int = 4
    ctx: int = 128
    householder: int = 2     # reflections per transport map
    chunk: int = 16          # scan chunk
    mlp_mult: float = 4.0
    rope: bool = True        # RoPE in attention (GPT / hybrid attention layers)
    pattern: str = ""        # hybrid layer pattern, e.g. "ddad" (d = sheaf-delta, a = attention); empty = homogeneous
    mixer: str = "sheaf"    # 'sheaf' (diag gated) | 'delta' (Householder delta-rule matrix memory)

class RMSNorm(nn.Module):
    def __init__(s, d, eps=1e-6):
        super().__init__(); s.w = nn.Parameter(torch.ones(d)); s.eps = eps
    def forward(s, x): return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + s.eps) * s.w

class MLP(nn.Module):
    def __init__(s, d, mult, gated):
        super().__init__(); s.gated = gated
        h = int(d * mult * (2 / 3 if gated else 1))
        s.up = nn.Linear(d, h * (2 if gated else 1), bias=False); s.down = nn.Linear(h, d, bias=False)
    def forward(s, x):
        u = s.up(x)
        if s.gated: a, b = u.chunk(2, -1); return s.down(F.silu(a) * b)
        return s.down(F.gelu(u))

# ---------------- Transformer baseline ----------------
def rope(x, pos, base=10000.0):  # x [B,h,T,dh]
    dh = x.shape[-1]; inv = 1.0 / (base ** (torch.arange(0, dh, 2, device=x.device).float() / dh))
    ang = pos[:, None].float() * inv[None]; cos, sin = ang.cos(), ang.sin(); x1, x2 = x[..., :dh // 2], x[..., dh // 2:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1)

class Attn(nn.Module):
    def __init__(s, c):
        super().__init__(); s.rope = c.rope; s.h = c.heads; s.qkv = nn.Linear(c.dim, 3 * c.dim, bias=False); s.o = nn.Linear(c.dim, c.dim, bias=False)
    def forward(s, x):
        B, T, D = x.shape
        q, k, v = s.qkv(x).view(B, T, 3, s.h, D // s.h).permute(2, 0, 3, 1, 4)
        if s.rope: pos = torch.arange(T, device=x.device); q, k = rope(q, pos), rope(k, pos)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return s.o(y.transpose(1, 2).reshape(B, T, D))
    def step(s, x, st):  # x [B,D]; st = (K,V) caches [B,h,t,dh]
        B, D = x.shape
        q, k, v = s.qkv(x).view(B, 3, s.h, 1, D // s.h).permute(1, 0, 2, 3, 4)
        if s.rope:
            pos = torch.tensor([0 if st is None else st[0].shape[2]], device=x.device); q, k = rope(q, pos), rope(k, pos)
        if st is not None: k = torch.cat([st[0], k], 2); v = torch.cat([st[1], v], 2)
        y = F.scaled_dot_product_attention(q, k, v)
        return s.o(y.reshape(B, D)), (k, v)

class TBlock(nn.Module):
    def __init__(s, c):
        super().__init__(); s.n1 = RMSNorm(c.dim); s.n2 = RMSNorm(c.dim); s.mix = Attn(c); s.mlp = MLP(c.dim, c.mlp_mult, False)
    def forward(s, x): x = x + s.mix(s.n1(x)); return x + s.mlp(s.n2(x))
    def step(s, x, st):
        y, st = s.mix.step(s.n1(x), st); x = x + y; return x + s.mlp(s.n2(x)), st

# ---------------- Householder Sheaf mixer ----------------
class SheafMixer(nn.Module):
    def __init__(s, c):
        super().__init__(); D = c.dim; s.D, s.H, s.M, s.chunk = D, c.heads, c.householder, c.chunk
        s.offsets = [2 ** i for i in range(int(math.log2(c.ctx)))]  # causal dyadic edges 1,2,4,...
        s.K = len(s.offsets)
        s.inp = nn.Linear(D, 4 * D, bias=False)         # u, gate z, recurrence-decay a, recurrence-input b
        s.edge_gate = nn.Linear(D, s.K * s.H, bias=True)  # data-dependent sheaf edge weights per head
        s.V = nn.Parameter(torch.randn(s.K, s.M, D))      # Householder reflection vectors per edge
        s.out = nn.Linear(D, D, bias=False)
    def Q(s):  # [K,D,D] product of M Householder reflections per edge (exactly orthogonal)
        V = F.normalize(s.V, dim=-1); I = torch.eye(s.D, device=V.device)
        Q = I.expand(s.K, s.D, s.D)
        for m in range(s.M):
            v = V[:, m]; Q = Q - 2 * torch.bmm(Q, v[:, :, None]) * v[:, None, :]  # Q (I - 2vv^T)
        return Q
    def transport(s, x):  # x [K,...,D]
        return torch.einsum("kij,k...j->k...i", s.Q(), x)
    def scan(s, a, b):  # h_t = a_t h_{t-1} + b_t, chunked & stable; a in [.5,1)
        B, T, D = a.shape; c = min(s.chunk, T); h_prev = a.new_zeros(B, D); out = []
        for i in range(0, T, c):
            la = torch.log(a[:, i:i + c]); A = la.cumsum(1)
            hc = torch.exp(A) * (torch.exp(-A) * b[:, i:i + c]).cumsum(1) + torch.exp(A) * h_prev[:, None]
            out.append(hc); h_prev = hc[:, -1]
        return torch.cat(out, 1)
    def forward(s, x):
        B, T, D = x.shape
        u, z, a, b = s.inp(x).chunk(4, -1)
        a = 0.5 + 0.5 * torch.sigmoid(a); b = (1 - a) * b
        g = torch.sigmoid(s.edge_gate(x)).view(B, T, s.K, s.H)
        tu = (u @ s.Q().permute(2, 0, 1).reshape(D, s.K * D)).view(B, T, s.K, D)  # transport all edges: one GEMM
        sh = torch.stack([F.pad(tu[:, :, k], (0, 0, d, 0))[:, :T] for k, d in enumerate(s.offsets)], 2)  # [B,T,K,D]
        diff = (g[..., None] * sh.view(B, T, s.K, s.H, D // s.H)).sum(2).reshape(B, T, D) / s.K ** 0.5
        y = (diff + s.scan(a, b)) * F.silu(z)
        return s.out(y)
    def init_state(s, B, dev): return (torch.zeros(B, s.offsets[-1], s.D, device=dev), torch.zeros(B, s.D, device=dev))
    def step(s, x, st):
        buf, h = st if st is not None else s.init_state(x.shape[0], x.device)
        B = x.shape[0]
        u, z, a, b = s.inp(x).chunk(4, -1)
        a = 0.5 + 0.5 * torch.sigmoid(a); b = (1 - a) * b
        g = torch.sigmoid(s.edge_gate(x)).view(B, s.K, s.H)
        sh = torch.stack([buf[:, d - 1] for d in s.offsets])  # [K,B,D]
        sh = s.transport(sh).view(s.K, B, s.H, D := s.D // s.H)
        diff = torch.einsum("bkh,kbhd->bhd", g, sh).reshape(B, s.D) / s.K ** 0.5
        h = a * h + b
        buf = torch.cat([u[:, None], buf[:, :-1]], 1)
        return s.out((diff + h) * F.silu(z)), (buf, h)

class DeltaMixer(nn.Module):
    """Matrix-state memory with Householder-type transitions (gated delta rule):
       S_t = a_t * S_{t-1} (I - b_t k k^T) + b_t v k^T,  b in [0,2] (b=2: exact reflection, eigenvalue -1 => state tracking),
       o_t = S_t q_t.  Training here = reference sequential loop (chunked WY kernel = engineering item, see fla)."""
    def __init__(s, c):
        super().__init__(); s.D, s.H = c.dim, c.heads; s.dh = c.dim // c.heads
        s.qkv = nn.Linear(c.dim, 3 * c.dim, bias=False); s.ba = nn.Linear(c.dim, 2 * c.heads, bias=True)
        s.g = nn.Linear(c.dim, c.dim, bias=False); s.out = nn.Linear(c.dim, c.dim, bias=False)
        s.onorm = RMSNorm(s.dh); nn.init.constant_(s.ba.bias[c.heads:], 3.0)  # decay a ~ sigmoid(3)=.95
    def _proj(s, x):
        sh = x.shape[:-1]; q, k, v = s.qkv(x).view(*sh, 3, s.H, s.dh).unbind(-3)
        q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)
        b, a = s.ba(x).view(*sh, 2, s.H).unbind(-2); return q, k, v, 2 * torch.sigmoid(b), torch.sigmoid(a)
    def _upd(s, S, q, k, v, b, a):  # S [B,H,dv,dk]
        Sk = torch.einsum("bhvk,bhk->bhv", S, k)
        S = a[..., None, None] * (S - b[..., None, None] * Sk[..., None] * k[..., None, :]) + b[..., None, None] * v[..., None] * k[..., None, :]
        return S, torch.einsum("bhvk,bhk->bhv", S, q)
    def forward(s, x):
        B, T, D = x.shape; q, k, v, b, a = s._proj(x); S = x.new_zeros(B, s.H, s.dh, s.dh); o = []
        for t in range(T): S, ot = s._upd(S, q[:, t], k[:, t], v[:, t], b[:, t], a[:, t]); o.append(ot)
        o = s.onorm(torch.stack(o, 1)).reshape(B, T, D); return s.out(o * F.silu(s.g(x)))
    def step(s, x, S):
        q, k, v, b, a = s._proj(x); S = x.new_zeros(x.shape[0], s.H, s.dh, s.dh) if S is None else S
        S, o = s._upd(S, q, k, v, b, a); return s.out(s.onorm(o).reshape(x.shape) * F.silu(s.g(x))), S

def make_mixer(c):
    if c.mixer == "sheaf": return SheafMixer(c)
    if c.mixer == "delta": return DeltaMixer(c)
    from .fastmixer import SheafDeltaMixer; return SheafDeltaMixer(c)

class HBlock(nn.Module):
    def __init__(s, c):
        super().__init__(); s.n1 = RMSNorm(c.dim); s.n2 = RMSNorm(c.dim); s.mix = make_mixer(c); s.mlp = MLP(c.dim, c.mlp_mult, True)
    def forward(s, x): x = x + s.mix(s.n1(x)); return x + s.mlp(s.n2(x))
    def step(s, x, st):
        y, st = s.mix.step(s.n1(x), st); x = x + y; return x + s.mlp(s.n2(x)), st

class LM(nn.Module):
    Block = None
    def __init__(s, c: Config):
        super().__init__(); s.c = c
        s.embed = nn.Embedding(c.vocab, c.dim)
        s.pos = nn.Embedding(c.ctx, c.dim) if (s.use_pos and not c.rope) else None
        s.blocks = nn.ModuleList([s.make_block(i) for i in range(c.layers)])
        s.norm_f = RMSNorm(c.dim)
        s.apply(s._init)
    def make_block(s, i): return s.Block(s.c)
    def _init(s, m):
        if isinstance(m, (nn.Linear, nn.Embedding)): nn.init.normal_(m.weight, std=0.02)
    def head(s, h): return F.linear(s.norm_f(h), s.embed.weight)  # tied
    def embed_in(s, x):
        e = s.embed(x)
        return e + s.pos(torch.arange(x.shape[1], device=x.device)) if s.pos is not None else e
    def forward(s, x):
        h = s.embed_in(x)
        for b in s.blocks: h = b(h)
        return s.head(h)
    def n_params(s): return sum(p.numel() for p in s.parameters())
    @torch.no_grad()
    def generate(s, prompt, n):  # prompt [B,T0]; O(1)-state (HSN) / KV-cache (GPT) decoding
        B = prompt.shape[0]; sts = [None] * len(s.blocks); out = prompt; logits = None
        for t in range(prompt.shape[1] + n - 1):
            tok = out[:, t]
            h = s.embed(tok) + (s.pos.weight[t] if s.pos is not None else 0)
            for i, b in enumerate(s.blocks): h, sts[i] = b.step(h, sts[i])
            if t >= prompt.shape[1] - 1:
                nxt = torch.multinomial(F.softmax(s.head(h), -1), 1); out = torch.cat([out, nxt], 1)
        return out

class GPT(LM): use_pos = True; Block = TBlock
class HSNLM(LM): use_pos = False; Block = HBlock
class HybridLM(LM):
    """pattern like 'ddad': d = Householder-sheaf-delta block, a = attention block."""
    use_pos = False
    def make_block(s, i):
        import copy; c = copy.copy(s.c); ch = (s.c.pattern or "d" * s.c.layers)[i]
        if ch == "a": return TBlock(c)
        c.mixer = "sheafdelta"; return HBlock(c)
