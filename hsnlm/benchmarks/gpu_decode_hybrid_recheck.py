"""Checks a specific overclaim: does the vectorized SheafDeltaMixer.step() (fastmixer.py) +
FusedSheafDeltaMixer.step() formula fix make the WINNING RoPE 3:1 hybrid (dddaddda, full/unwindowed
RoPE attention) decode flat/O(1) out to ctx 65536? It does not, and shouldn't -- the 2 full-attention
layers still carry a growing KV cache, same mechanism as INFERENCE_ADVANTAGE.md section 4 always said.
This measures the REAL post-fix number: same architecture as section 4 (8 layers, dddaddda, width
1024, 16 heads), current (fixed) mixer code, full RoPE attention with a real growing KV cache --
not the sliding-window-ALiBi variant from gpu_swa_alibi_hybrid.py's bench_decode_o1, which is a
genuinely different (and lower-quality, per gpu_swa_alibi_hybrid.json) architecture."""
import sys, copy, time, json, torch; sys.path.insert(0, ".")
from hsnlm import Config
from hsnlm.models import LM, HBlock, TBlock
assert torch.cuda.is_available()
DIM, HEADS, LAYERS, VOCAB = 1024, 16, 8, 256
PATTERN = "dddaddda"
OUT = "results/gpu_decode_hybrid_recheck.json"

class HybridFusedLM(LM):
    use_pos = False
    def make_block(s, i):
        c = copy.copy(s.c); ch = (s.c.pattern or "d" * s.c.layers)[i]
        if ch == "a": return TBlock(c)
        c.mixer = "sheafdelta-fused"; return HBlock(c)

torch.manual_seed(0)
c = Config(vocab=VOCAB, dim=DIM, layers=LAYERS, ctx=131072, heads=HEADS, mlp_mult=3.65, pattern=PATTERN)
m = HybridFusedLM(c).cuda().eval()

@torch.no_grad()
def bench(ctx_len, iters=20):
    x = torch.randn(1, DIM, device="cuda")
    sts = []
    for b in m.blocks:
        if isinstance(b, TBlock):
            sts.append((torch.randn(1, HEADS, ctx_len, DIM // HEADS, device="cuda"),
                        torch.randn(1, HEADS, ctx_len, DIM // HEADS, device="cuda")))
        else:
            sd = b.mix
            sts.append((torch.randn(1, sd.H, sd.dh, sd.dh, device="cuda"),
                        torch.randn(1, sd.offsets[-1], sd.D, device="cuda")))
    kv_bytes = sum(sum(t.numel() * t.element_size() for t in st) for st, blk in zip(sts, m.blocks) if isinstance(blk, TBlock))
    state_bytes = sum(sum(t.numel() * t.element_size() for t in st) for st, blk in zip(sts, m.blocks) if not isinstance(blk, TBlock))
    def step():
        h = x
        for b, st in zip(m.blocks, sts): h, _ = b.step(h, st)
        return h
    for _ in range(5): step()
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(iters): step()
    torch.cuda.synchronize(); ms = (time.time() - t0) / iters * 1000
    return dict(step_ms=round(ms, 3), kv_cache_mb=round(kv_bytes / 1e6, 3), sheafdelta_state_mb=round(state_bytes / 1e6, 3))

out = {}
for ctx_len in (256, 1024, 4096, 16384, 65536):
    out[str(ctx_len)] = bench(ctx_len)
    print(ctx_len, out[str(ctx_len)], flush=True)
json.dump(out, open(OUT, "w"), indent=1)
print("ALL DONE")
