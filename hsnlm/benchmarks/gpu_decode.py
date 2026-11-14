"""GPU port of experiments/decode_o1.py: per-token decode cost of ONE layer vs context length already
consumed (batch 1, D=1024, 16 heads). CUDA-synced timing; GPU decode is latency-bound by kernel-launch
overhead at batch 1, so this checks whether the CPU-measured O(1) vs O(N) shape survives on GPU."""
import sys, time, torch; sys.path.insert(0, ".")
from hsnlm import Config
from hsnlm.models import Attn
from hsnlm.fastmixer import SheafDeltaMixer
torch.manual_seed(0)
c = Config(dim=1024, heads=16, ctx=8192)
att = Attn(c).cuda().eval(); sd = SheafDeltaMixer(c).cuda().eval()
x = torch.randn(1, 1024, device="cuda")
def t(f, n=100):
    for _ in range(10): f()  # warmup
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(n): f()
    torch.cuda.synchronize(); return (time.time() - t0) / n * 1e3
print("context | attention step ms | KV cache MB | sheaf-delta step ms | state MB")
with torch.no_grad():
    for n in (256, 1024, 4096, 16384, 65536):
        kv = (torch.randn(1, 16, n, 64, device="cuda"), torch.randn(1, 16, n, 64, device="cuda"))
        st = (torch.randn(1, 16, 64, 64, device="cuda"), torch.randn(1, sd.offsets[-1], 1024, device="cuda"))
        print(n, f"| {t(lambda: att.step(x, kv)):.3f} | {2*n*1024*4/1e6:.0f} | {t(lambda: sd.step(x, st)):.3f} | {(16*64*64 + sd.offsets[-1]*1024)*4/1e6:.2f}", flush=True)
