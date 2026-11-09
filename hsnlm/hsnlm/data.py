import torch
class CharData:
    def __init__(self, path, ctx=128, split=0.9):
        text = open(path, encoding="utf8").read()
        self.chars = sorted(set(text)); self.V = len(self.chars)
        self.stoi = {c: i for i, c in enumerate(self.chars)}
        ids = torch.tensor([self.stoi[c] for c in text])
        n = int(len(ids) * split)
        self.train, self.val, self.ctx = ids[:n], ids[n:], ctx
    def batch(self, B, gen=None):
        ix = torch.randint(len(self.train) - self.ctx - 1, (B,), generator=gen)
        x = torch.stack([self.train[i:i + self.ctx] for i in ix])
        y = torch.stack([self.train[i + 1:i + self.ctx + 1] for i in ix])
        return x, y
    def val_batches(self, B):
        d, c = self.val, self.ctx
        n = (len(d) - 1) // c
        x = d[:n * c].view(n, c); y = d[1:n * c + 1].view(n, c)
        for i in range(0, n, B): yield x[i:i + B], y[i:i + B]
    def decode(self, ids): return "".join(self.chars[i] for i in ids)

class ByteData(CharData):
    """Byte-level LM data from a big file: train = first n_train bytes, val = next n_val bytes."""
    def __init__(self, path, ctx=256, n_train=8_000_000, n_val=1_000_000):
        raw = open(path, "rb").read(n_train + n_val)
        ids = torch.frombuffer(bytearray(raw), dtype=torch.uint8).long()
        self.train, self.val, self.ctx, self.V = ids[:n_train], ids[n_train:], ctx, 256
