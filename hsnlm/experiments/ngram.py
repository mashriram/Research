"""Gradient-free 'training' = counting. Interpolated Witten-Bell char n-gram on the SAME split as the neural runs."""
import sys, math, time; from collections import defaultdict; sys.path.insert(0, ".")
text = open("data/shakespeare.txt", encoding="utf8").read(); n = int(len(text) * 0.9); tr, va = text, text
tr, va = text[:n], text[n:]; V = len(set(text))
for N in (3, 4, 5, 6, 7, 8):
    t0 = time.time(); cnt = [defaultdict(lambda: defaultdict(int)) for _ in range(N)]
    for i in range(len(tr)):
        for k in range(N):
            if i - k >= 0: cnt[k][tr[i - k:i]][tr[i]] += 1
    train_s = time.time() - t0
    def p(ctx, c):
        pr = 1.0 / V
        for k in range(0, N):
            if k > len(ctx): break
            d = cnt[k].get(ctx[len(ctx) - k:] if k else "")
            if not d: break
            tot = sum(d.values()); T = len(d); pr = (d.get(c, 0) + T * pr) / (tot + T)
        return pr
    bits = sum(-math.log2(p(va[max(0, i - N + 1):i], va[i])) for i in range(len(va))) / len(va)
    print(f"order {N}: val bits/char {bits:.3f}   'training' time {train_s:.1f}s", flush=True)
