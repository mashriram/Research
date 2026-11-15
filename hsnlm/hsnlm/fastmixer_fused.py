"""GPU-fused variant of SheafDeltaMixer: replaces the eager Python chunk loop (delta_chunked) with
fla's Triton chunk_gated_delta_rule kernel for the matrix-memory part. Dyadic Householder part (cheap,
already a few GEMMs) is unchanged.

Note on formula: SheafDeltaMixer's own recurrence (delta_chunked/delta_seq) is
  S_t = a_t * (S_{t-1} + b_t (v_t - S_{t-1} k_t) k_t^T)                      (write term also decayed by a_t)
fla's chunk_gated_delta_rule implements the textbook Gated DeltaNet recurrence (the one quoted in
findings_report.md section 4):
  S_t = a_t * S_{t-1} (I - b_t k_t k_t^T) + b_t v_t k_t^T                    (write term NOT decayed by a_t)
These differ by a factor of a_t (in [0.8,1]) on the write term -- not bit-identical to the rest of this
codebase, but is the literature-standard formula this project's own report cites. Only used here to get
a real fused-kernel speed/memory measurement; correctness is checked against fla's own naive reference,
not against delta_chunked.

Note on readout scale: chunk_gated_delta_rule defaults `scale=None`, which it resolves internally to
`k.shape[-1] ** -0.5` (see fla.ops.gated_delta_rule.naive.naive_recurrent_gated_delta_rule: `q = q *
scale` before the readout `o = q @ h`) -- so forward()'s o is implicitly S_t @ q_t * dh**-0.5. step()'s
readout below multiplies by the same factor to match, so the two stay numerically consistent (verified:
a from-scratch rollout comparison without this factor showed a 0.049 max-abs-diff at fp32 between
forward() and a step()-by-step rollout of the same model; with it, this drops to float-rounding noise)."""
import torch, torch.nn.functional as F
from fla.ops.gated_delta_rule import chunk_gated_delta_rule
from .fastmixer import SheafDeltaMixer, hh_coef

class FusedSheafDeltaMixer(SheafDeltaMixer):
    def forward(s, x):
        B, T, D = x.shape
        q, k, v, b, a = s._proj(x)
        dt = v.dtype; q, k, b, g = q.to(dt), k.to(dt), b.to(dt), torch.log(a).to(dt)  # fla kernel wants one uniform dtype (autocast leaves these mismatched)
        o, _ = chunk_gated_delta_rule(q, k, v, g=g, beta=b)
        g = torch.sigmoid(s.edge(x)).view(B, T, s.K, s.H)
        dy = s.dyadic(v.reshape(B, T, D), g)
        if s.postnorm: y = s.onorm((o.reshape(B, T, D) + dy).view(B, T, s.H, s.dh)).reshape(B, T, D)
        else: y = s.onorm(o).reshape(B, T, D) + dy
        return s.out(y * F.silu(s.g(x)))

    def step(s, x, st):
        B = x.shape[0]; S, buf = st if st is not None else (x.new_zeros(B, s.H, s.dh, s.dh), x.new_zeros(B, s.offsets[-1], s.D))
        q, k, v, b, a = s._proj(x)
        Sk = torch.einsum("bhvk,bhk->bhv", S, k)
        S = a[..., None, None] * (S - b[..., None, None] * Sk[..., None] * k[..., None, :]) + b[..., None, None] * v[..., None] * k[..., None, :]
        o = torch.einsum("bhvk,bhk->bhv", S, q) * s.dh ** -0.5; vf = v.reshape(B, s.D); V, G = s._VG()
        g = torch.sigmoid(s.edge(x)).view(B, s.K, s.H)
        u = buf[:, [d - 1 for d in s.offsets]]                                       # [B,K,D]
        c = hh_coef(torch.einsum("bkd,kmd->bkm", u, V), G)                           # [B,K,M]
        tr = (u - 2 * torch.einsum("bkm,kmd->bkd", c, V)).view(B, s.K, s.H, s.dh)
        acc = (g[..., None] * tr).sum(1) / s.K ** 0.5                                # [B,H,dh]
        buf = torch.cat([vf[:, None], buf[:, :-1]], 1)
        if s.postnorm: y = s.onorm(o + acc).reshape(B, s.D)
        else: y = s.onorm(o).reshape(B, s.D) + acc.reshape(B, s.D)
        return s.out(y * F.silu(s.g(x))), (S, buf)
