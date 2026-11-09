"""Per-token decode cost of ONE layer vs. context length already consumed (batch 1, D=1024, 16 heads)."""
import sys, time, torch; sys.path.insert(0, ".")
from hsnlm import Config
from hsnlm.models import Attn
from hsnlm.fastmixer import SheafDeltaMixer
torch.set_num_threads(8); torch.manual_seed(0)
c = Config(dim=1024, heads=16, ctx=8192); att = Attn(c).eval(); sd = SheafDeltaMixer(c).eval(); x = torch.randn(1, 1024)
def t(f, n=30):
    f(); t0 = time.time()
    for _ in range(n): f()
    return (time.time() - t0) / n * 1e3
print("context | attention step ms | KV cache MB | sheaf-delta step ms | state MB")
with torch.no_grad():
    for n in (256, 1024, 4096, 16384):
        kv = (torch.randn(1, 16, n, 64), torch.randn(1, 16, n, 64)); st = (torch.randn(1, 16, 64, 64), torch.randn(1, sd.offsets[-1], 1024))
        print(n, f"| {t(lambda: att.step(x, kv)):.2f} | {2*n*1024*4/1e6:.0f} | {t(lambda: sd.step(x, st)):.2f} | {(16*64*64 + sd.offsets[-1]*1024)*4/1e6:.2f}")
