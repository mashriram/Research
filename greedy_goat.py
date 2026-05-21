"""
greedy_goat.py — Phase-1 Pipeline-Parallel GOAT + Phase-2 Fine-Tune

Speed strategy:
  Phase 1 — Pipeline Parallel Greedy Pre-Training
    ┌──────────────────────────────────────────────────────────────────┐
    │ Batch: 1    2    3    4    5    6  ...                           │
    │ Layer1: B1   B2   B3   B4   B5   B6  (computes + local update)  │
    │ Layer2: --   B1   B2   B3   B4   B5  (from Layer1 act. buffer)  │
    │ Layer3: --   --   B1   B2   B3   B4  (from Layer2 act. buffer)  │
    └──────────────────────────────────────────────────────────────────┘
    Each layer runs on a separate Python thread — PyTorch releases GIL
    during tensor math so ALL layers compute their updates truly in parallel.
    This gives near-linear speedup with depth.

  Phase 2 — Lightweight Global Fine-tune (head only, 1 epoch)
    All GOATLinear params frozen. Standard backprop through in_proj → head.
    Converges in ~1 epoch because Phase-1 features are already rich.
"""

import os, time, math, queue, threading
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ── Environment ────────────────────────────────────────────────────────────────
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
torch.set_num_threads(1)

import torch._dynamo
torch._dynamo.config.allow_unspec_int_on_nn_module = True

# ═══════════════════════════════════════════════════════════════════════════════
# 1.  LoSparse Analytical Kernel
# ═══════════════════════════════════════════════════════════════════════════════
class GOATPipelineLayer(nn.Module):
    """
    One GOAT LoSparse layer with:
      - Analytical AdamW (no autograd overhead)
      - Persistent sparse mask (refreshed every `mask_every` steps)
      - Thread-safe `update(x, z, out, target)` called by the pipeline worker
    """
    def __init__(self, in_f, out_f, num_classes, rank=None, sparsity=0.85, lr=1e-3):
        super().__init__()
        self.sparsity = sparsity
        self.lr       = lr
        self.num_classes = num_classes

        if rank is None:
            rank = max(8, int(min(in_f, out_f) ** 0.5 * 4))
        self.rank = rank

        self.A    = nn.Parameter(torch.randn(out_f, rank) * 0.02)
        self.B    = nn.Parameter(torch.randn(rank, in_f) * 0.02)
        self.S    = nn.Parameter(torch.randn(out_f, in_f) * 0.001)
        self.bias = nn.Parameter(torch.zeros(out_f))

        self.W_loc = nn.Parameter(torch.randn(num_classes, out_f) * 0.02)
        self.b_loc = nn.Parameter(torch.zeros(num_classes))

        # AdamW moment buffers
        for n, p in [('A', self.A), ('B', self.B), ('S', self.S),
                     ('bias', self.bias), ('W', self.W_loc), ('bl', self.b_loc)]:
            self.register_buffer(f'm_{n}', torch.zeros_like(p))
            self.register_buffer(f'v_{n}', torch.zeros_like(p))

        self.register_buffer('_mask_flag', torch.ones(1, dtype=torch.bool))
        self.register_buffer('cached_mask', torch.zeros(out_f, in_f))
        self.step_counter  = 0
        self.mask_every    = 100
        self._lock         = threading.Lock()   # for thread-safe counter access

    # ── mask ───────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def _rebuild_mask(self):
        k = max(1, int(self.S.numel() * (1 - self.sparsity)))
        thr = torch.topk(self.S.abs().view(-1), k).values[-1]
        self.cached_mask.copy_((self.S.abs() >= thr).float())
        self._mask_flag.fill_(False)

    def get_mask(self):
        if self._mask_flag.item():
            self._rebuild_mask()
        return self.cached_mask

    # ── forward (pure inference, no update) ────────────────────────────────────
    @torch.no_grad()
    def infer(self, x):
        """Run LoSparse forward, return (out, z, U) for later update."""
        mask = self.get_mask()
        U    = x @ self.B.t()
        z    = U @ self.A.t() + F.linear(x, self.S * mask) + self.bias
        out  = torch.relu(z)
        return out, z, U

    # ── analytical AdamW ───────────────────────────────────────────────────────
    @torch.no_grad()
    def _adamw(self, p, g, m, v, t, wd=0.01):
        b1, b2, eps = 0.9, 0.999, 1e-8
        m.mul_(b1).add_(g, alpha=1-b1)
        v.mul_(b2).addcmul_(g, g, value=1-b2)
        denom = (v.sqrt() / math.sqrt(1 - b2**t)).add_(eps)
        p.data.mul_(1 - self.lr * wd)
        p.data.addcdiv_(m, denom, value=-(self.lr / (1 - b1**t)))

    # ── local update (called by pipeline thread) ────────────────────────────────
    @torch.no_grad()
    def update(self, x_d, z, out, U, target, mask):
        """Compute analytical gradients and apply AdamW locally."""
        with self._lock:
            t = self.step_counter + 1

        # Local logits
        y_pred   = F.linear(out, self.W_loc, self.b_loc)
        probs    = F.softmax(y_pred, dim=-1)
        y_true   = F.one_hot(target, num_classes=self.num_classes).float()
        d_pred   = (probs - y_true) / x_d.size(0)

        # Gradients
        g_W   = d_pred.t() @ out
        g_bl  = d_pred.sum(0)
        d_h   = d_pred @ self.W_loc
        d_z   = d_h * (z > 0).float()
        g_A   = d_z.t() @ U
        g_B   = (d_z @ self.A).t() @ x_d
        g_S   = (d_z.t() @ x_d) * mask
        g_bias = d_z.sum(0)

        self._adamw(self.W_loc, g_W,   self.m_W,   self.v_W,   t)
        self._adamw(self.b_loc, g_bl,  self.m_bl,  self.v_bl,  t)
        self._adamw(self.A,     g_A,   self.m_A,   self.v_A,   t)
        self._adamw(self.B,     g_B,   self.m_B,   self.v_B,   t)
        self._adamw(self.S,     g_S,   self.m_S,   self.v_S,   t)
        self._adamw(self.bias,  g_bias,self.m_bias,self.v_bias, t)

        with self._lock:
            self.step_counter += 1
            if self.step_counter % self.mask_every == 0:
                self._mask_flag.fill_(True)


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  Pipeline Trainer
# ═══════════════════════════════════════════════════════════════════════════════
_SENTINEL = object()   # signals pipeline threads to stop

class LayerWorker(threading.Thread):
    """
    Each layer lives in its own thread.
    It reads activations from `in_queue`, runs inference + local update,
    and writes activations to `out_queue` for the next layer.
    """
    def __init__(self, layer: GOATPipelineLayer, in_q: queue.Queue,
                 out_q: queue.Queue, is_last: bool):
        super().__init__(daemon=True)
        self.layer   = layer
        self.in_q    = in_q
        self.out_q   = out_q
        self.is_last = is_last

    def run(self):
        while True:
            item = self.in_q.get()
            if item is _SENTINEL:
                self.out_q.put(_SENTINEL)
                break
            x, target = item
            mask = self.layer.get_mask()
            out, z, U = self.layer.infer(x)
            self.layer.update(x, z, out, U, target, mask)
            # Pass detached activations downstream (next layer only needs features)
            self.out_q.put((out.detach(), target))
            self.in_q.task_done()


class GOATPipelineMLP(nn.Module):
    """
    Full model: in_proj → [GOATPipelineLayer × N] → classifier
    """
    def __init__(self, in_features=784, hidden_dims=None, num_classes=10):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 256, 256]

        self.in_proj = nn.Linear(in_features, hidden_dims[0])
        self.goat_layers = nn.ModuleList([
            GOATPipelineLayer(
                hidden_dims[i],
                hidden_dims[i+1] if i < len(hidden_dims)-1 else hidden_dims[-1],
                num_classes=num_classes
            )
            for i in range(len(hidden_dims))
        ])
        self.classifier = nn.Linear(hidden_dims[-1], num_classes)

    def forward_inference(self, x):
        """Standard sequential inference (Phase 2 eval + fine-tune)."""
        x = torch.relu(self.in_proj(x.view(x.size(0), -1)))
        with torch.no_grad():
            for layer in self.goat_layers:
                out, _, _ = layer.infer(x)
                x = out
        return self.classifier(x)

    def freeze_goat(self):
        for layer in self.goat_layers:
            for p in layer.parameters():
                p.requires_grad_(False)

    def unfreeze_goat(self):
        for layer in self.goat_layers:
            for p in layer.parameters():
                p.requires_grad_(True)


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  Phase-1  Pipeline Training
# ═══════════════════════════════════════════════════════════════════════════════
def phase1_pipeline_train(model: GOATPipelineMLP, loader, epochs: int = 3):
    """
    Greedy pipeline-parallel Phase-1 training.

    - Builds a chain of queues: data → Q[0] → Layer0 → Q[1] → Layer1 → ... → sink
    - Feeds each mini-batch into Q[0]; all layer threads process simultaneously.
    - NO global backward. Pure analytical AdamW inside each layer thread.
    """
    model.train()
    n_layers = len(model.goat_layers)

    # Build queue chain
    queues = [queue.Queue(maxsize=4) for _ in range(n_layers + 1)]
    workers = []
    for i, layer in enumerate(model.goat_layers):
        w = LayerWorker(
            layer=layer,
            in_q=queues[i],
            out_q=queues[i + 1],
            is_last=(i == n_layers - 1)
        )
        workers.append(w)
        w.start()

    # Sink thread: drains the final queue (prevents blocking)
    def sink():
        while True:
            item = queues[-1].get()
            if item is _SENTINEL:
                break
    sink_thread = threading.Thread(target=sink, daemon=True)
    sink_thread.start()

    t0 = time.time()
    for epoch in range(epochs):
        total = 0
        for x_raw, target in loader:
            # Compute in_proj projection on main thread (fast, shared)
            with torch.no_grad():
                x_proj = torch.relu(model.in_proj(x_raw.view(x_raw.size(0), -1)))
            # Push to pipeline — will be processed by all layer threads in parallel
            queues[0].put((x_proj.detach(), target))
            total += 1
        print(f"  Phase1 ep{epoch+1}: {total} batches pushed through pipeline")

    # Drain and shutdown
    queues[0].put(_SENTINEL)
    sink_thread.join()
    for w in workers:
        w.join()

    elapsed = time.time() - t0
    print(f"  Phase1 total: {elapsed:.1f}s")
    return elapsed


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  Phase-2  Fast Fine-Tune (head + in_proj only)
# ═══════════════════════════════════════════════════════════════════════════════
def phase2_finetune(model: GOATPipelineMLP, loader, epochs: int = 2):
    """
    Freeze all GOAT layers. Just train in_proj + classifier head via backprop.
    Because Phase-1 features are already rich, this converges in 1-2 epochs.
    """
    model.freeze_goat()

    opt = optim.AdamW(
        list(model.in_proj.parameters()) + list(model.classifier.parameters()),
        lr=1e-3)

    model.train()
    t0 = time.time()
    for epoch in range(epochs):
        total_loss = 0
        for x, y in loader:
            opt.zero_grad()
            logits = model.forward_inference(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            opt.step()
            total_loss += loss.item()
        print(f"  Phase2 ep{epoch+1} loss={total_loss/len(loader):.4f}")

    elapsed = time.time() - t0
    print(f"  Phase2 total: {elapsed:.1f}s")
    return elapsed


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  Standard Baseline
# ═══════════════════════════════════════════════════════════════════════════════
class StandardMLP(nn.Module):
    def __init__(self, in_features=784, hidden_dims=None, classes=10):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 256, 256]
        layers, cur = [], in_features
        for h in hidden_dims:
            layers += [nn.Linear(cur, h), nn.ReLU()]
            cur = h
        self.features   = nn.Sequential(*layers)
        self.classifier = nn.Linear(cur, classes)

    def forward(self, x):
        return self.classifier(self.features(x.view(x.size(0), -1)))


def train_standard(model, loader, epochs):
    opt = optim.AdamW(model.parameters(), lr=1e-3)
    model.train()
    t0 = time.time()
    for ep in range(epochs):
        tl = 0
        for x, y in loader:
            opt.zero_grad()
            F.cross_entropy(model(x), y).backward()
            opt.step()
    return time.time() - t0


# ═══════════════════════════════════════════════════════════════════════════════
# 6.  Evaluation
# ═══════════════════════════════════════════════════════════════════════════════
def evaluate_standard(model, loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for x, y in loader:
            correct += model(x).argmax(1).eq(y).sum().item()
    return 100.0 * correct / len(loader.dataset)


def evaluate_goat(model, loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for x, y in loader:
            correct += model.forward_inference(x).argmax(1).eq(y).sum().item()
    return 100.0 * correct / len(loader.dataset)


# ═══════════════════════════════════════════════════════════════════════════════
# 7.  Benchmark Runner
# ═══════════════════════════════════════════════════════════════════════════════
def get_loaders(name, batch_size=128):
    cls  = datasets.MNIST if name == 'MNIST' else datasets.FashionMNIST
    mean = (0.1307,) if name == 'MNIST' else (0.2860,)
    std  = (0.3081,) if name == 'MNIST' else (0.3530,)
    tf   = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    return (
        DataLoader(cls('./data', train=True,  download=True,  transform=tf),
                   batch_size=batch_size, shuffle=True,  num_workers=0),
        DataLoader(cls('./data', train=False, download=False, transform=tf),
                   batch_size=1000, shuffle=False, num_workers=0),
    )


CONFIGS = [
    {"name": "Deep & Narrow",  "dims": [128, 128, 128, 128, 128]},
    {"name": "Medium",         "dims": [256, 256, 256]},
    {"name": "Wide & Shallow", "dims": [512, 512]},
]
DATASETS       = ["MNIST", "FashionMNIST"]
P1_EPOCHS      = 3    # greedy pre-train epochs
P2_EPOCHS      = 2    # fine-tune epochs
STD_EPOCHS     = P1_EPOCHS + P2_EPOCHS   # same total budget for fairness
BATCH          = 128


def main():
    results = []
    for ds in DATASETS:
        print(f"\n{'='*55}")
        print(f"  DATASET: {ds}")
        print(f"{'='*55}")
        tr, te = get_loaders(ds, BATCH)

        for cfg in CONFIGS:
            print(f"\n▸ Topology: {cfg['name']}  dims={cfg['dims']}")

            # ── Standard ─────────────────────────────────────────────────────
            print(f"  [Standard AdamW – {STD_EPOCHS} epochs]")
            std = StandardMLP(hidden_dims=cfg['dims'])
            std_t = train_standard(std, tr, STD_EPOCHS)
            std_a = evaluate_standard(std, te)
            print(f"  ✦ Standard  {std_t:.1f}s | {std_a:.2f}%")

            # ── GOAT Pipeline ─────────────────────────────────────────────────
            print(f"  [GOAT Phase1={P1_EPOCHS}ep Pipeline  +  Phase2={P2_EPOCHS}ep Fine-tune]")
            goat = GOATPipelineMLP(hidden_dims=cfg['dims'])

            t_p1 = phase1_pipeline_train(goat, tr, epochs=P1_EPOCHS)
            t_p2 = phase2_finetune(goat,       tr, epochs=P2_EPOCHS)
            goat_t = t_p1 + t_p2
            goat_a = evaluate_goat(goat, te)
            print(f"  ✦ GOAT P1={t_p1:.1f}s P2={t_p2:.1f}s total={goat_t:.1f}s | {goat_a:.2f}%")

            results.append(dict(
                Dataset=ds, Topology=cfg['name'],
                std_t=std_t, std_a=std_a,
                goat_t=goat_t, goat_a=goat_a,
                p1=t_p1, p2=t_p2,
            ))

    # ── Final table ─────────────────────────────────────────────────────────────
    w = 90
    print(f"\n{'='*w}")
    print(f"  FINAL RESULTS  ({STD_EPOCHS} total epochs, batch {BATCH}, single CPU)")
    print(f"{'='*w}")
    hdr = (f"{'Dataset':<14}{'Topology':<16}"
           f"{'Std Time':>10}{'GOAT Time':>10}"
           f"{'Std Acc':>9}{'GOAT Acc':>10}"
           f"{'Speed Δ':>9}{'Acc Δ':>8}")
    print(hdr)
    print('-'*w)
    for r in results:
        spd = (r['std_t'] - r['goat_t']) / r['std_t'] * 100
        acc = r['goat_a'] - r['std_a']
        spd_tag = f"GOAT+{spd:.0f}%" if spd > 0 else f"Std+{-spd:.0f}%"
        acc_tag = f"GOAT+{acc:.2f}" if acc > 0 else f"{acc:.2f}"
        print(f"{r['Dataset']:<14}{r['Topology']:<16}"
              f"{r['std_t']:>9.1f}s{r['goat_t']:>9.1f}s"
              f"{r['std_a']:>8.2f}%{r['goat_a']:>9.2f}%"
              f"{spd_tag:>10}{acc_tag:>9}")
    print('='*w)


if __name__ == "__main__":
    main()
