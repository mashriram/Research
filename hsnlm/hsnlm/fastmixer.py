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
        super().__init__(); D, H = c.dim, (c.mixer_heads or c.heads); s.D, s.H, s.dh, s.M, s.chunk = D, H, D // H, c.householder, c.chunk
        s.offsets = [2 ** i for i in range(min(5, int(math.log2(c.ctx))))]; s.K = len(s.offsets)
        s.qkv = nn.Linear(D, 3 * D, bias=False); s.ba = nn.Linear(D, 2 * H)
        if c.decay_init == "multiscale": s.ba.bias.data[H:] = torch.linspace(-1.0, 4.5, H)
        else: nn.init.constant_(s.ba.bias[H:], 3.0)
        s.g = nn.Linear(D, D, bias=False); s.out = nn.Linear(D, D, bias=False); s.edge = nn.Linear(D, s.K * H)
        s.V = nn.Parameter(torch.randn(s.K, s.M, D)); s.onorm = RMSNorm(s.dh)
        s.postnorm, s.postnorm_scaled, s.token_shift = c.postnorm, c.postnorm_scaled, c.token_shift
        if s.token_shift: s.alpha_q = nn.Parameter(torch.full((H, 1), 0.5)); s.alpha_k = nn.Parameter(torch.full((H, 1), 0.5))
    def _proj_raw(s, x):
        sh = x.shape[:-1]; q, k, v = s.qkv(x).view(*sh, 3, s.H, s.dh).unbind(-3)
        b, a = s.ba(x).view(*sh, 2, s.H).unbind(-2)
        return q, k, v, 2 * torch.sigmoid(b), 0.8 + 0.2 * torch.sigmoid(a)
    def _proj(s, x):  # forward(): x is [B,T,D], shift (if on) done vectorized over T
        q, k, v, b, a = s._proj_raw(x)
        if s.token_shift:
            q = q + s.alpha_q * F.pad(q[:, :-1], (0, 0, 0, 0, 1, 0))
            k = k + s.alpha_k * F.pad(k[:, :-1], (0, 0, 0, 0, 1, 0))
        return F.normalize(q, dim=-1), F.normalize(k, dim=-1), v, b, a
    def _proj_step(s, x, qprev, kprev):  # step(): x is [B,D]; qprev/kprev [B,H,dh] raw q,k from t-1 (zeros at t=0)
        q, k, v, b, a = s._proj_raw(x)
        if s.token_shift:
            qn, kn = F.normalize(q + s.alpha_q * qprev, dim=-1), F.normalize(k + s.alpha_k * kprev, dim=-1)
        else:
            qn, kn = F.normalize(q, dim=-1), F.normalize(k, dim=-1)
        return qn, kn, v, b, a, q, k
    def _VG(s):
        V = F.normalize(s.V, dim=-1); return V, torch.einsum("kmd,kjd->kmj", V, V)
    def dyadic(s, v, g):  # v [B,T,D], g [B,T,K,H] -> sum_k g_k * shift_k(H_k v)   (H_k isometric)
        B, T, D = v.shape; V, G = s._VG()
        c = hh_coef(torch.einsum("btd,kmd->btkm", v, V), G)                      # [B,T,K,M]
        vh = v.view(B, T, s.H, s.dh); p1 = torch.zeros_like(vh)
        gc = c.new_zeros(B, T, s.K, s.H, s.M)
        for k, d in enumerate(s.offsets):
            if d < T:
                p1[:, d:] = p1[:, d:] + g[:, d:, k, :, None] * vh[:, :T - d]
                gc[:, d:, k] = g[:, d:, k, :, None] * c[:, :T - d, k, None, :]
        p2 = torch.einsum("btkhm,kmhd->bthd", gc, V.view(s.K, s.M, s.H, s.dh))
        return ((p1 - 2 * p2) / s.K ** 0.5).reshape(B, T, D)
    def forward(s, x):
        B, T, D = x.shape; q, k, v, b, a = s._proj(x)
        o, _ = delta_chunked(q, k, v, b, a, s.chunk)
        g = torch.sigmoid(s.edge(x)).view(B, T, s.K, s.H)
        dy = s.dyadic(v.reshape(B, T, D), g)
        if s.postnorm or s.postnorm_scaled: y = s.onorm((o.reshape(B, T, D) + dy).view(B, T, s.H, s.dh)).reshape(B, T, D)  # eager path has no implicit readout scale, so "scaled" == plain postnorm here
        else: y = s.onorm(o).reshape(B, T, D) + dy
        return s.out(y * F.silu(s.g(x)))
    def step(s, x, st):
        B = x.shape[0]
        if s.token_shift:
            if st is None: S, buf, qprev, kprev = x.new_zeros(B, s.H, s.dh, s.dh), x.new_zeros(B, s.offsets[-1], s.D), x.new_zeros(B, s.H, s.dh), x.new_zeros(B, s.H, s.dh)
            else: S, buf, qprev, kprev = st
            q, k, v, b, a, qraw, kraw = s._proj_step(x, qprev, kprev)
        else:
            S, buf = st if st is not None else (x.new_zeros(B, s.H, s.dh, s.dh), x.new_zeros(B, s.offsets[-1], s.D))
            q, k, v, b, a = s._proj(x)
        S = a[..., None, None] * (S + b[..., None, None] * (v - torch.einsum("bhvk,bhk->bhv", S, k))[..., None] * k[..., None, :])
        o = torch.einsum("bhvk,bhk->bhv", S, q); vf = v.reshape(B, s.D); V, G = s._VG()
        g = torch.sigmoid(s.edge(x)).view(B, s.K, s.H)
        u = buf[:, [d - 1 for d in s.offsets]]                                       # [B,K,D]
        c = hh_coef(torch.einsum("bkd,kmd->bkm", u, V), G)                           # [B,K,M]
        tr = (u - 2 * torch.einsum("bkm,kmd->bkd", c, V)).view(B, s.K, s.H, s.dh)
        acc = (g[..., None] * tr).sum(1) / s.K ** 0.5                                # [B,H,dh]
        buf = torch.cat([vf[:, None], buf[:, :-1]], 1)
        if s.postnorm or s.postnorm_scaled: y = s.onorm(o + acc).reshape(B, s.D)
        else: y = s.onorm(o).reshape(B, s.D) + acc.reshape(B, s.D)
        out = s.out(y * F.silu(s.g(x)))
        return out, ((S, buf, qraw, kraw) if s.token_shift else (S, buf))
