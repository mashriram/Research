import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from research_work.goat_nn import GOATDTPLinear

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
torch.set_num_threads(1)

# ============================================================
# 1. MODELS
# ============================================================
class StandardMLP(nn.Module):
    def __init__(self, in_features=784, hidden_dims=[256, 256, 256], classes=10):
        super().__init__()
        layers = []
        cur = in_features
        for h in hidden_dims:
            layers += [nn.Linear(cur, h), nn.ReLU()]
            cur = h
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(cur, classes)

    def forward(self, x):
        return self.classifier(self.features(x.view(x.size(0), -1)))


class GOATMLP(nn.Module):
    """
    GOAT MLP with Hybrid Gradient Flow:
      - Local analytical AdamW in every GOATDTPLinear layer.
      - Global backprop flows from the final loss all the way through the
        last GOATDTPLinear (whose output is NOT detached before classifier).
        This ensures the last layer's A,B,S also receive global gradient.
      - in_proj is trained by global backprop since the first layer's
        output feeds the global graph all the way back.
    """
    def __init__(self, in_features=784, hidden_dims=[256, 256, 256], classes=10):
        super().__init__()
        self.in_proj = nn.Linear(in_features, hidden_dims[0])
        dims = hidden_dims

        self.layers = nn.ModuleList([
            GOATDTPLinear(dims[i], dims[i+1] if i < len(dims)-1 else dims[-1],
                          num_classes=classes, lr=1e-3)
            for i in range(len(dims))
        ])
        self.classifier = nn.Linear(dims[-1], classes)

    def forward(self, x, target=None):
        x = x.view(x.size(0), -1)
        x = torch.relu(self.in_proj(x))
        for layer in self.layers:
            x = layer(x, local_target=target)
        # No detach — global backprop reaches A,B,S of the last layer
        return self.classifier(x)


# ============================================================
# 2. TRAINING
# ============================================================
def train_standard(model, loader, epochs):
    opt = optim.AdamW(model.parameters(), lr=1e-3)
    model.train()
    t0 = time.time()
    for ep in range(epochs):
        total = 0
        for x, y in loader:
            opt.zero_grad()
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            opt.step()
            total += loss.item()
        print(f"  Standard ep{ep+1} loss={total/len(loader):.4f}")
    return time.time() - t0


def train_goat(model, loader, epochs):
    # Train in_proj, classifier, and any global-graph params together
    opt = optim.AdamW(
        list(model.in_proj.parameters()) + list(model.classifier.parameters()),
        lr=1e-3)
    model.train()
    t0 = time.time()
    for ep in range(epochs):
        total = 0
        for x, y in loader:
            opt.zero_grad()
            loss = F.cross_entropy(model(x, target=y), y)
            loss.backward()
            opt.step()
            total += loss.item()
        print(f"  GOAT     ep{ep+1} loss={total/len(loader):.4f}")
    return time.time() - t0


def evaluate(model, loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for x, y in loader:
            correct += model(x).argmax(1).eq(y).sum().item()
    return 100.0 * correct / len(loader.dataset)


def get_loaders(name):
    cls  = datasets.MNIST if name == 'MNIST' else datasets.FashionMNIST
    mean = (0.1307,) if name == 'MNIST' else (0.2860,)
    std  = (0.3081,) if name == 'MNIST' else (0.3530,)
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    return (
        DataLoader(cls('./data', train=True,  download=True, transform=tf), batch_size=128, shuffle=True),
        DataLoader(cls('./data', train=False, download=False, transform=tf), batch_size=1000),
    )


# ============================================================
# 3. MEGA BENCHMARK
# ============================================================
EPOCHS = 5

configs = [
    {"name": "Deep & Narrow",  "dims": [128, 128, 128, 128, 128, 128]},
    {"name": "Medium",         "dims": [256, 256, 256]},
    {"name": "Wide & Shallow", "dims": [1024, 1024]},
]
datasets_to_test = ["MNIST", "FashionMNIST"]

def main():
    results = []
    for ds in datasets_to_test:
        print(f"\n{'='*45}")
        print(f" DATASET: {ds}")
        print(f"{'='*45}")
        train_loader, test_loader = get_loaders(ds)

        for cfg in configs:
            print(f"\n--- Topology: {cfg['name']} ---")

            std = StandardMLP(hidden_dims=cfg['dims'])
            std = torch.compile(std, mode="max-autotune-no-cudagraphs", dynamic=True)
            std_t = train_standard(std, train_loader, EPOCHS)
            std_a = evaluate(std, test_loader)
            print(f"  Standard -> {std_t:.1f}s | {std_a:.2f}%")

            goat = GOATMLP(hidden_dims=cfg['dims'])
            goat = torch.compile(goat, mode="max-autotune-no-cudagraphs", dynamic=True)
            goat_t = train_goat(goat, train_loader, EPOCHS)
            goat_a = evaluate(goat, test_loader)
            print(f"  GOAT     -> {goat_t:.1f}s | {goat_a:.2f}%")

            results.append(dict(
                Dataset=ds, Topology=cfg['name'],
                std_t=std_t, std_a=std_a,
                goat_t=goat_t, goat_a=goat_a
            ))

    # ── Final table ──────────────────────────────────────────
    print(f"\n{'='*85}")
    print(f" FINAL BENCHMARK  ({EPOCHS} epochs, batch 128, single CPU)")
    print(f"{'='*85}")
    hdr = f"{'Dataset':<15}{'Topology':<16}{'Std Time':>10}{'GOAT Time':>10}{'Std Acc':>9}{'GOAT Acc':>10}{'Speed':>8}{'Acc':>7}"
    print(hdr)
    print('-'*85)
    for r in results:
        speed_tag = "GOAT✓" if r['goat_t'] < r['std_t'] else f"+{(r['goat_t']/r['std_t']-1)*100:.0f}%"
        acc_tag   = "GOAT✓" if r['goat_a'] > r['std_a'] else f"-{r['std_a']-r['goat_a']:.2f}"
        print(f"{r['Dataset']:<15}{r['Topology']:<16}"
              f"{r['std_t']:>9.1f}s{r['goat_t']:>9.1f}s"
              f"{r['std_a']:>8.2f}%{r['goat_a']:>9.2f}%"
              f"{speed_tag:>9}{acc_tag:>8}")
    print('='*85)


if __name__ == "__main__":
    main()
