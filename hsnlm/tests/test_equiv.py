import torch, sys; sys.path.insert(0, ".")
from hsnlm import HSNLM, GPT, Config
def test_step_matches_parallel():
    torch.manual_seed(0); c = Config(layers=2, dim=64, ctx=32)
    for M in (HSNLM, GPT):
        m = M(c).eval(); x = torch.randint(0, 65, (2, 32))
        with torch.no_grad():
            full = m(x); sts = [None] * c.layers; outs = []
            for t in range(32):
                h = m.embed(x[:, t]) + (m.pos.weight[t] if m.pos is not None else 0)
                for i, b in enumerate(m.blocks): h, sts[i] = b.step(h, sts[i])
                outs.append(m.head(h))
        assert torch.allclose(full, torch.stack(outs, 1), atol=1e-4), M.__name__
def test_householder_isometry():
    from hsnlm.models import SheafMixer
    m = SheafMixer(Config()); x = torch.randn(7, 3, 5, 128)
    assert torch.allclose(m.transport(x).norm(dim=-1), x.norm(dim=-1), atol=1e-4)
def test_causal():
    m = HSNLM(Config(layers=2, dim=64, ctx=32)).eval(); x = torch.randint(0, 65, (1, 32)); x2 = x.clone(); x2[0, 20:] = 0
    assert torch.allclose(m(x)[:, :20], m(x2)[:, :20], atol=1e-5)
