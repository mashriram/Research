"""RBlockFast ingredients.
1) thin Householder transport: H_1..H_M x = x - 2 sum_m c_m v_m  (c via a tiny triangular recursion) -> O(M D) not O(D^2) per edge.
2) chunkwise gated delta rule (S_t = a_t (S_{t-1} + b_t (v_t - S_{t-1} k_t) k_t^T)): exact, O(N), matmul-only inner loop over chunks.
3) SheafDeltaMixer = delta matrix memory (global recall) + dyadic Householder sheaf diffusion (local, isometric transport)."""
import math, torch, torch.nn as nn, torch.nn.functional as F
from .models import RMSNorm

def hh_coef(a, G):
    c = []
    for m in range(a.shape[-1]):
        cm = a[..., m]
        for j in range(m): cm = cm - 2 * c[j] * G[:, m, j]
        c.append(cm)
    return torch.stack(c, -1)

def delta_seq(q, k, v, b, a):  # reference loop. q,k,v [B,T,H,d], b,a [B,T,H]
    B, T, H, d = q.shape; S = q.new_zeros(B, H, d, d); out = []
    for t in range(T):
        Sk = torch.einsum("bhvk,bhk->bhv", S, k[:, t])
        S = a[:, t, :, None, None] * (S + b[:, t, :, None, None] * (v[:, t] - Sk)[..., None] * k[:, t, :, None, :])
        out.append(torch.einsum("bhvk,bhk->bhv", S, q[:, t]))
    return torch.stack(out, 1), S

def delta_chunked(q, k, v, b, a, C=32, S0=None):
    B, T, H, d = q.shape; pad = (-T) % C
    if pad:
        z = lambda x, val=0.: F.pad(x, (0, 0, 0, 0, 0, pad) if x.dim() == 4 else (0, 0, 0, pad), value=val)
        q, k, v, b, a = z(q), z(k), z(v), z(b), z(a, 1.0)
    Tp = T + pad; nc = Tp // C
    tr = lambda x: x.transpose(1, 2).reshape(B, H, nc, C, *x.shape[3:])
    q, k, v = tr(q), tr(k), tr(v); b = b.transpose(1, 2).reshape(B, H, nc, C); a = a.transpose(1, 2).reshape(B, H, nc, C)
    cum = torch.log(a).cumsum(-1); g = cum.exp(); gprev = (cum - torch.log(a)).exp()
    vt = v / gprev[..., None]
    KK = torch.tril(k @ k.transpose(-1, -2), -1)
    I = torch.eye(C, dtype=q.dtype, device=q.device)
    Tinv = torch.linalg.solve_triangular(I + b[..., None] * KK, I.expand_as(KK), upper=False)
    TB = Tinv * b[..., None, :]                      # (I + diag(b) L)^-1 diag(b)
    QK = torch.tril(q @ k.transpose(-1, -2))
    S = q.new_zeros(B, H, d, d) if S0 is None else S0; outs = []
    for c in range(nc):
        kc, qc = k[:, :, c], q[:, :, c]
        U = TB[:, :, c] @ (vt[:, :, c] - kc @ S.transpose(-1, -2))
        outs.append((qc @ S.transpose(-1, -2) + QK[:, :, c] @ U) * g[:, :, c, :, None])
        S = g[:, :, c, -1, None, None] * (S + U.transpose(-1, -2) @ kc)
    o = torch.cat(outs, 2).reshape(B, H, Tp, d).transpose(1, 2)[:, :T]
    return o, S

class SheafDeltaMixer(nn.Module):
    def __init__(s, c):
        super().__init__(); D, H = c.dim, c.heads; s.D, s.H, s.dh, s.M, s.chunk = D, H, D // H, c.householder, c.chunk
        s.offsets = [2 ** i for i in range(min(5, int(math.log2(c.ctx))))]; s.K = len(s.offsets)
        s.qkv = nn.Linear(D, 3 * D, bias=False); s.ba = nn.Linear(D, 2 * H); nn.init.constant_(s.ba.bias[H:], 3.0)
        s.g = nn.Linear(D, D, bias=False); s.out = nn.Linear(D, D, bias=False); s.edge = nn.Linear(D, s.K * H)
        s.V = nn.Parameter(torch.randn(s.K, s.M, D)); s.onorm = RMSNorm(s.dh)
    def _proj(s, x):
        sh = x.shape[:-1]; q, k, v = s.qkv(x).view(*sh, 3, s.H, s.dh).unbind(-3)
        b, a = s.ba(x).view(*sh, 2, s.H).unbind(-2)
        return F.normalize(q, dim=-1), F.normalize(k, dim=-1), v, 2 * torch.sigmoid(b), 0.8 + 0.2 * torch.sigmoid(a)
    def _VG(s):
        V = F.normalize(s.V, dim=-1); return V, torch.einsum("kmd,kjd->kmj", V, V)
    def dyadic(s, v, g):  # v [B,T,D], g [B,T,K,H] -> sum_k g_k * shift_k(H_k v)   (H_k isometric)
        B, T, D = v.shape; V, G = s._VG()
        c = hh_coef(torch.einsum("btd,kmd->btkm", v, V), G)                      # [B,T,K,M]
        off = torch.tensor(s.offsets, device=v.device)[:, None]; idx = torch.arange(T, device=v.device)[None] - off  # [K,T]
        ok = (idx >= 0).to(v.dtype); idx = idx.clamp(min=0)
        vs = (v[:, idx] * ok[None, :, :, None]).view(B, s.K, T, s.H, s.dh)          # one gather for all K edges
        cs = c.permute(0, 2, 1, 3).gather(2, idx[None, :, :, None].expand(B, s.K, T, s.M)) * ok[None, :, :, None]
        p1 = torch.einsum("btkh,bkthd->bthd", g, vs)
        p2 = torch.einsum("btkh,bktm,kmhd->bthd", g, cs, V.view(s.K, s.M, s.H, s.dh))
        return ((p1 - 2 * p2) / s.K ** 0.5).reshape(B, T, D)
    def forward(s, x):
        B, T, D = x.shape; q, k, v, b, a = s._proj(x)
        o, _ = delta_chunked(q, k, v, b, a, s.chunk)
        g = torch.sigmoid(s.edge(x)).view(B, T, s.K, s.H)
        return s.out((s.onorm(o).reshape(B, T, D) + s.dyadic(v.reshape(B, T, D), g)) * F.silu(s.g(x)))
    def step(s, x, st):
        B = x.shape[0]; S, buf = st if st is not None else (x.new_zeros(B, s.H, s.dh, s.dh), x.new_zeros(B, s.offsets[-1], s.D))
        q, k, v, b, a = s._proj(x)
        S = a[..., None, None] * (S + b[..., None, None] * (v - torch.einsum("bhvk,bhk->bhv", S, k))[..., None] * k[..., None, :])
        o = torch.einsum("bhvk,bhk->bhv", S, q); vf = v.reshape(B, s.D); V, G = s._VG()
        cur = torch.stack([vf] + [buf[:, d - 2] for d in s.offsets[1:]], 0) if False else None
        g = torch.sigmoid(s.edge(x)).view(B, s.K, s.H); acc = 0
        for i, d in enumerate(s.offsets):
            u = vf if d == 0 else buf[:, d - 1]          # v_{t-d}; buf[:,j] = v_{t-1-j}
            c = hh_coef(torch.einsum("bd,kmd->bkm", u, V), G)[:, i]   # [B,M]
            tr = (u - 2 * c @ V[i]).view(B, s.H, s.dh); acc = acc + g[:, i, :, None] * tr
        buf = torch.cat([vf[:, None], buf[:, :-1]], 1)
        y = s.onorm(o).reshape(B, s.D) + (acc / s.K ** 0.5).reshape(B, s.D)
        return s.out(y * F.silu(s.g(x))), (S, buf)
