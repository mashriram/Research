import torch, sys; sys.path.insert(0, ".")
from hsnlm import Config
from hsnlm.fastmixer import delta_seq, delta_chunked, SheafDeltaMixer, hh_coef
def test_chunk_equals_seq():
    torch.manual_seed(0); B, T, H, d = 2, 70, 3, 16   # T not multiple of chunk -> padding path
    q, k = [torch.nn.functional.normalize(torch.randn(B, T, H, d), dim=-1) for _ in range(2)]; v = torch.randn(B, T, H, d)
    b = 2 * torch.rand(B, T, H); a = 0.8 + 0.2 * torch.rand(B, T, H)
    o1, S1 = delta_seq(q, k, v, b, a); o2, S2 = delta_chunked(q, k, v, b, a, 16)
    assert torch.allclose(o1, o2, atol=1e-4), (o1 - o2).abs().max(); assert torch.allclose(S1, S2, atol=1e-4)
def test_thin_transport_matches_explicit_reflections():
    torch.manual_seed(0); m = SheafDeltaMixer(Config(dim=64, ctx=64)); x = torch.randn(5, 64); V, G = m._VG()
    c = hh_coef(torch.einsum("bd,kmd->bkm", x, V), G); thin = x[:, None] - 2 * torch.einsum("bkm,kmd->bkd", c, V)
    ref = x[:, None].expand(5, m.K, 64).clone()
    for mm in range(m.M):
        v = V[:, mm]; ref = ref - 2 * (ref * v).sum(-1, keepdim=True) * v
    assert torch.allclose(thin, ref, atol=1e-5) and torch.allclose(thin.norm(dim=-1), x[:, None].norm(dim=-1).expand(5, m.K), atol=1e-4)
def test_step_equals_parallel_and_causal():
    from hsnlm import HSNLM
    torch.manual_seed(0); c = Config(layers=2, dim=64, ctx=64, mixer="sheafdelta"); m = HSNLM(c).eval(); x = torch.randint(0, 65, (2, 64))
    with torch.no_grad():
        full = m(x); sts = [None] * 2; outs = []
        for t in range(64):
            h = m.embed(x[:, t])
            for i, bl in enumerate(m.blocks): h, sts[i] = bl.step(h, sts[i])
            outs.append(m.head(h))
        x2 = x.clone(); x2[:, 40:] = 0
        assert torch.allclose(m(x2)[:, :40], full[:, :40], atol=1e-4)
    assert torch.allclose(full, torch.stack(outs, 1), atol=1e-3), (full - torch.stack(outs, 1)).abs().max()
