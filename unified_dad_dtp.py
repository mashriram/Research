"""
unified_dad_dtp.py — Hardware-Adaptive Decoupled Analytical Dense (DAD) Engine

A highly accelerated, 100% backprop-free target propagation training engine
that dynamically adapts to available hardware (CPU vs GPU) and evaluates
deep target propagation networks against backpropagation baselines.

Supported Datasets:
1. MNIST (784-dim input, 10 classes)
2. FashionMNIST (784-dim input, 10 classes)
3. CIFAR-10 (3072-dim flattened R/G/B input, 10 classes)
"""

import os
import time
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

# Optimize PyTorch Inductor config settings if using GPU
import torch._dynamo
torch._dynamo.config.allow_unspec_int_on_nn_module = True

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════════════════
# 1. Hardware Detection & Optimization Setup
# ═══════════════════════════════════════════════════════════════════════════════
def detect_hardware(force_cpu=False):
    if torch.cuda.is_available() and not force_cpu:
        device = torch.device("cuda:0")
        device_name = torch.cuda.get_device_name(0)
        bf16_supported = torch.cuda.is_bf16_supported()
        # Enable tf32 for highly optimized matmuls on Ampere/Ada GPU architectures
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        amp_dtype = torch.bfloat16 if bf16_supported else torch.float16
        hw_info = f"NVIDIA GPU [{device_name}] | BF16 Supported: {bf16_supported} | TF32: Enabled"
    elif torch.backends.mps.is_available() and not force_cpu:
        device = torch.device("mps")
        amp_dtype = torch.float32
        hw_info = "Apple Silicon MPS GPU"
    else:
        device = torch.device("cpu")
        amp_dtype = torch.float32
        hw_info = "CPU (Eager Execution Mode)"
        
    return device, amp_dtype, hw_info

# ═══════════════════════════════════════════════════════════════════════════════
# 2. Global High-Performance AdamW Update Function
# ═══════════════════════════════════════════════════════════════════════════════
def adamw_step_fn(p, g, m, v, lr, t):
    b1, b2, eps = 0.9, 0.999, 1e-8
    
    # In-place tensor updates
    m.mul_(b1).add_(g, alpha=1.0 - b1)
    v.mul_(b2).addcmul_(g, g, value=1.0 - b2)
    
    bias_correction1 = 1.0 - b1 ** t
    bias_correction2 = 1.0 - b2 ** t
    
    step_size = lr / bias_correction1
    denom = (v.sqrt() / torch.sqrt(bias_correction2)).add_(eps)
    
    p.mul_(1.0 - lr * 0.01)
    p.addcdiv_(m, denom, value=-step_size)

# ═══════════════════════════════════════════════════════════════════════════════
# 3. Decoupled Analytical Dense (DAD) Layer
# ═══════════════════════════════════════════════════════════════════════════════
class GOATDADLayer(nn.Module):
    def __init__(self, in_f, out_f, num_classes, device=None):
        super().__init__()
        self.num_classes = num_classes
        self.device = device

        # Main Weight & Bias (No Autograd tracking)
        self.W    = nn.Parameter(torch.randn(out_f, in_f, device=device) * (2.0 / in_f) ** 0.5, requires_grad=False)
        self.bias = nn.Parameter(torch.zeros(out_f, device=device), requires_grad=False)

        # Local Task Classifier Head (Trainable)
        self.W_loc = nn.Parameter(torch.randn(num_classes, out_f, device=device) * 0.02, requires_grad=False)
        self.b_loc = nn.Parameter(torch.zeros(num_classes, device=device), requires_grad=False)

        # Optimizer Moments (No Autograd tracking)
        self.m_W = nn.Parameter(torch.zeros_like(self.W, device=device), requires_grad=False)
        self.v_W = nn.Parameter(torch.zeros_like(self.W, device=device), requires_grad=False)
        self.m_bias = nn.Parameter(torch.zeros_like(self.bias, device=device), requires_grad=False)
        self.v_bias = nn.Parameter(torch.zeros_like(self.bias, device=device), requires_grad=False)

    def forward(self, x):
        z = F.linear(x, self.W, self.bias)
        out = torch.relu(z)
        return out, z

# ═══════════════════════════════════════════════════════════════════════════════
# 4. GOAT DAD MLP Model
# ═══════════════════════════════════════════════════════════════════════════════
class GOATDADMLP(nn.Module):
    def __init__(self, in_features=784, hidden_dims=None, num_classes=10, device=None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 256, 256]

        dims = [in_features] + hidden_dims
        self.goat_layers = nn.ModuleList([
            GOATDADLayer(dims[i], dims[i+1], num_classes=num_classes, device=device)
            for i in range(len(dims) - 1)
        ])
        self.classifier = nn.Linear(hidden_dims[-1], num_classes, device=device)

    def forward_inference(self, x):
        x = x.view(x.size(0), -1)
        with torch.no_grad():
            for layer in self.goat_layers:
                out, _ = layer(x)
                x = out
        return self.classifier(x)

# ═══════════════════════════════════════════════════════════════════════════════
# 5. DAD Trainer (Dynamic JIT Compilation)
# ═══════════════════════════════════════════════════════════════════════════════
class GOATDADTrainer:
    def __init__(self, model: GOATDADMLP, device: torch.device):
        self.model = model
        self.device = device
        self.num_classes = model.goat_layers[0].num_classes
        self.step_counter = 0
        
        self.t_tensor = torch.tensor(0.0, dtype=torch.float32, device=device)

        # Dynamic JIT Compilation: compile ONLY on GPU to avoid CPU JIT compile-time lag
        if device.type == 'cuda':
            self.compiled_step = torch.compile(self.unified_step)
        else:
            self.compiled_step = self.unified_step

        last_layer = model.goat_layers[-1]
        out_features = last_layer.W.size(0)
        self.HTH = torch.zeros(out_features + 1, out_features + 1, device=device)
        self.HTY = torch.zeros(out_features + 1, self.num_classes, device=device)

    def reset_step_counter(self):
        self.step_counter = 0
        self.t_tensor.zero_()

    def unified_step(self, x, y, lr, t):
        # 1. Forward Pass
        acts = [x.view(x.size(0), -1)]
        zs = []
        x_d = acts[0]
        for layer in self.model.goat_layers:
            out, z = layer(x_d)
            acts.append(out)
            zs.append(z)
            x_d = out
            
        # 2. Backward Pass
        y_true = F.one_hot(y, num_classes=self.num_classes).float()
        inv_batch_size = 1.0 / x.size(0)
        
        for i, layer in enumerate(self.model.goat_layers):
            out = acts[i+1]
            z = zs[i]
            x_prev = acts[i]
            
            # Local alignment prediction
            y_pred = F.linear(out, layer.W_loc, layer.b_loc)
            probs  = F.softmax(y_pred, dim=-1)
            d_pred = (probs - y_true) * inv_batch_size

            # Analytical local gradients
            g_W_loc = d_pred.t() @ out
            g_b_loc = d_pred.sum(0)
            d_h     = d_pred @ layer.W_loc
            d_z     = d_h * (z > 0).float()
            g_W     = d_z.t() @ x_prev
            g_bias  = d_z.sum(0)

            # In-place updates: AdamW for main, ultra-fast SGD for local classifiers
            adamw_step_fn(layer.W,     g_W,     layer.m_W,     layer.v_W,     lr, t)
            adamw_step_fn(layer.bias,  g_bias,  layer.m_bias,  layer.v_bias,  lr, t)
            layer.W_loc.add_(g_W_loc, alpha=-1e-3)
            layer.b_loc.add_(g_b_loc, alpha=-1e-3)

        return acts[-1]

    @torch.no_grad()
    def train_epoch(self, loader, amp_dtype, accumulate_solver=False):
        self.model.train()
        for x, y in loader:
            self.step_counter += 1
            self.t_tensor.add_(1.0)
            
            with torch.amp.autocast(self.device.type, dtype=amp_dtype, enabled=(self.device.type == 'cuda')):
                out_last = self.compiled_step(x, y, 1e-3, self.t_tensor)
                
                # Eager solver accumulation outside JIT to ensure 0 graph breaks
                if accumulate_solver:
                    ones = torch.ones(out_last.size(0), 1, device=self.device)
                    h_aug = torch.cat([out_last, ones], dim=1)
                    y_onehot = F.one_hot(y, num_classes=self.num_classes).float()
                    self.HTH.add_(h_aug.t() @ h_aug)
                    self.HTY.add_(h_aug.t() @ y_onehot)

    @torch.no_grad()
    def solve_head(self, lambda_reg=1e-3):
        t0 = time.time()
        out_features = self.model.goat_layers[-1].W.size(0)
        
        reg = lambda_reg * torch.eye(out_features + 1, device=self.device)
        try:
            W_aug = torch.linalg.solve(self.HTH + reg, self.HTY)
        except RuntimeError:
            W_aug = torch.linalg.pinv(self.HTH + reg) @ self.HTY

        W = W_aug[:-1, :].t()
        b = W_aug[-1, :]
        self.model.classifier.weight.copy_(W)
        self.model.classifier.bias.copy_(b)
        
        elapsed = time.time() - t0
        return elapsed

# ═══════════════════════════════════════════════════════════════════════════════
# 6. Standard MLP Baseline (Compiled & Autocasted)
# ═══════════════════════════════════════════════════════════════════════════════
class StandardMLP(nn.Module):
    def __init__(self, in_features=784, hidden_dims=None, classes=10, device=None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 256, 256]
        layers, cur = [], in_features
        for h in hidden_dims:
            layers += [nn.Linear(cur, h, device=device), nn.ReLU()]
            cur = h
        self.features   = nn.Sequential(*layers)
        self.classifier = nn.Linear(cur, classes, device=device)

    def forward(self, x):
        return self.classifier(self.features(x.view(x.size(0), -1)))

class StandardCompiledTrainer:
    def __init__(self, model: StandardMLP, device: torch.device):
        self.model = model
        self.device = device
        self.opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        if device.type == 'cuda':
            self.compiled_step = torch.compile(self._raw_step)
        else:
            self.compiled_step = self._raw_step

    def _raw_step(self, x, y):
        self.opt.zero_grad()
        logits = self.model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        self.opt.step()

    def train_epoch(self, loader, amp_dtype):
        self.model.train()
        for x, y in loader:
            with torch.amp.autocast(self.device.type, dtype=amp_dtype, enabled=(self.device.type == 'cuda')):
                self.compiled_step(x, y)

# ═══════════════════════════════════════════════════════════════════════════════
# 7. High-Performance Evaluation Functions
# ═══════════════════════════════════════════════════════════════════════════════
def evaluate_standard(model, loader, device):
    model.eval()
    correct = 0
    with torch.no_grad():
        for x, y in loader:
            correct += model(x).argmax(1).eq(y).sum().item()
    return 100.0 * correct / len(loader.dataset)

def evaluate_goat(model, loader, device):
    model.eval()
    correct = 0
    with torch.no_grad():
        for x, y in loader:
            correct += model.forward_inference(x).argmax(1).eq(y).sum().item()
    return 100.0 * correct / len(loader.dataset)

# ═══════════════════════════════════════════════════════════════════════════════
# 8. High-Performance Cached Dataset & Loader (Device-Native RAM/VRAM Caching)
# ═══════════════════════════════════════════════════════════════════════════════
class CachedLoader:
    def __init__(self, x, y, b_size, shuffle=True):
        self.x = x
        self.y = y
        self.batch_size = b_size
        self.shuffle = shuffle
        self.num_samples = x.size(0)
        self.dataset = self
        
    def __len__(self):
        return self.num_samples
        
    def __iter__(self):
        if self.shuffle:
            perm = torch.randperm(self.num_samples, device=self.x.device)
            self.x_shuffled = self.x[perm]
            self.y_shuffled = self.y[perm]
        else:
            self.x_shuffled = self.x
            self.y_shuffled = self.y
        self.idx = 0
        return self
        
    def __next__(self):
        if self.idx >= self.num_samples:
            raise StopIteration
        x_batch = self.x_shuffled[self.idx : self.idx + self.batch_size]
        y_batch = self.y_shuffled[self.idx : self.idx + self.batch_size]
        self.idx += self.batch_size
        return x_batch, y_batch

def get_loaders(name, batch_size=256, device=None):
    if device is None:
        device = torch.device('cpu')
        
    if name == 'MNIST':
        cls = datasets.MNIST
        mean, std = 0.1307, 0.3081
        in_features = 784
    elif name == 'FashionMNIST':
        cls = datasets.FashionMNIST
        mean, std = 0.2860, 0.3530
        in_features = 784
    elif name == 'CIFAR-10':
        cls = datasets.CIFAR10
        # CIFAR-10 mean and std for R, G, B channels
        mean = torch.tensor([0.4914, 0.4822, 0.4465], device=device).view(3, 1, 1)
        std = torch.tensor([0.2023, 0.1994, 0.2010], device=device).view(3, 1, 1)
        in_features = 3072
    else:
        raise ValueError(f"Unknown dataset: {name}")
        
    train_dataset = cls('./data', train=True, download=True)
    test_dataset  = cls('./data', train=False, download=False)
    
    # Load dataset into tensors
    if name == 'CIFAR-10':
        x_train = torch.tensor(train_dataset.data, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
        x_test = torch.tensor(test_dataset.data, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
        
        x_train = x_train.to(device)
        x_test = x_test.to(device)
        
        x_train = (x_train - mean) / std
        x_test = (x_test - mean) / std
        
        y_train = torch.tensor(train_dataset.targets, dtype=torch.long, device=device)
        y_test = torch.tensor(test_dataset.targets, dtype=torch.long, device=device)
    else:
        x_train = (train_dataset.data.float() / 255.0 - mean) / std
        y_train = train_dataset.targets
        x_test = (test_dataset.data.float() / 255.0 - mean) / std
        y_test = test_dataset.targets
        x_train, y_train = x_train.to(device), y_train.to(device)
        x_test, y_test = x_test.to(device), y_test.to(device)
        
    # Flatten images to 1D feature vectors
    x_train = x_train.view(x_train.size(0), -1)
    x_test = x_test.view(x_test.size(0), -1)
    
    return CachedLoader(x_train, y_train, batch_size, shuffle=True), CachedLoader(x_test, y_test, 1000, shuffle=False)

# ═══════════════════════════════════════════════════════════════════════════════
# 9. Comparative Benchmarking Harness (Sequential CPU/GPU Executions)
# ═══════════════════════════════════════════════════════════════════════════════
CONFIGS = [
    {"name": "Deep & Narrow",           "dims": [128, 128, 128, 128, 128]},
    {"name": "Medium",                  "dims": [256, 256, 256]},
    {"name": "Wide & Shallow",          "dims": [512, 512]},
    {"name": "Deep & Ultra-Narrow (10)", "dims": [128]*10},
]
DATASETS       = ["MNIST", "FashionMNIST", "CIFAR-10"]
TRAIN_EPOCHS   = 3
BATCH          = 256

def run_benchmark_on_device(force_cpu=False):
    device, amp_dtype, hw_info = detect_hardware(force_cpu=force_cpu)
    
    print("\n" + "="*80)
    print(f" 🚀 RUNNING MEGA-BENCHMARK ON: {hw_info}")
    print("="*80)
    print(f" Autocast Precision: {amp_dtype}")
    print(f" Training Epochs   : {TRAIN_EPOCHS} | Batch Size: {BATCH}")
    print("="*80 + "\n")

    # ── One-Time Global Warmup ──
    print("⚡ Performing one-time global warmups...")
    dummy_x = torch.randn(BATCH, 784, device=device)
    dummy_y = torch.randint(0, 10, (BATCH,), device=device)
    
    dummy_std = StandardMLP(hidden_dims=[128], device=device)
    dummy_std_trainer = StandardCompiledTrainer(dummy_std, device)
    dummy_std_trainer.train_epoch([(dummy_x, dummy_y)], amp_dtype)
    
    dummy_goat = GOATDADMLP(hidden_dims=[128], device=device)
    dummy_goat_trainer = GOATDADTrainer(dummy_goat, device)
    dummy_goat_trainer.train_epoch([(dummy_x, dummy_y)], amp_dtype, accumulate_solver=False)
    dummy_goat_trainer.train_epoch([(dummy_x, dummy_y)], amp_dtype, accumulate_solver=True)
    print("⚡ Warmups complete! Starting benchmarks...\n")

    device_results = []
    
    for ds in DATASETS:
        print(f"\n{'='*70}")
        print(f" 📂 DATASET: {ds}")
        print(f"{'='*70}")
        tr, te = get_loaders(ds, BATCH, device=device)
        in_features = 3072 if ds == 'CIFAR-10' else 784

        for cfg in CONFIGS:
            print(f"\n▶ Topology: {cfg['name']} | Dimensions: {cfg['dims']}")

            # ── Standard Autograd Baseline ──
            print(f"  [Standard Backprop] Initializing...")
            std = StandardMLP(in_features=in_features, hidden_dims=cfg['dims'], device=device)
            std_trainer = StandardCompiledTrainer(std, device)
            
            print("  ✦ Warming up baseline...")
            std_trainer.train_epoch(tr, amp_dtype) # minor pre-run
            
            print(f"  ✦ Running standard baseline...")
            t0 = time.time()
            for ep in range(TRAIN_EPOCHS):
                std_trainer.train_epoch(tr, amp_dtype)
            std_t = time.time() - t0
            std_a = evaluate_standard(std, te, device)
            std_throughput = (len(tr.dataset) * TRAIN_EPOCHS) / std_t
            print(f"  ✦ Standard: {std_t:.2f}s | {std_a:.2f}% | Throughput: {std_throughput:.0f} img/s")

            # ── DAD Engine (Backprop-Free) ──
            print(f"  [DAD Target Prop] Initializing...")
            goat = GOATDADMLP(in_features=in_features, hidden_dims=cfg['dims'], device=device)
            goat_trainer = GOATDADTrainer(goat, device)
            
            print("  ✦ Warming up DAD engine...")
            goat_trainer.train_epoch(tr, amp_dtype, accumulate_solver=False)
            goat_trainer.reset_step_counter()
            
            print(f"  ✦ Running decoupled target prop...")
            t0 = time.time()
            for ep in range(TRAIN_EPOCHS):
                accumulate = (ep == TRAIN_EPOCHS - 1)
                goat_trainer.train_epoch(tr, amp_dtype, accumulate_solver=accumulate)
            t_p1 = time.time() - t0
            
            t_p2 = goat_trainer.solve_head()
            goat_t = t_p1 + t_p2
            goat_a = evaluate_goat(goat, te, device)
            goat_throughput = (len(tr.dataset) * TRAIN_EPOCHS) / goat_t
            print(f"  ✦ DAD     : P1={t_p1:.2f}s P2={t_p2*1000:.2f}ms total={goat_t:.2f}s | {goat_a:.2f}% | Throughput: {goat_throughput:.0f} img/s")

            device_results.append(dict(
                Dataset=ds, Topology=cfg['name'], Device=device.type.upper(),
                std_t=std_t, std_a=std_a, std_thru=std_throughput,
                goat_t=goat_t, goat_a=goat_a, goat_thru=goat_throughput
            ))
            
    return device_results

def print_dominance_matrix(results):
    w = 110
    print("\n" + "="*w)
    print("                      🏆 COMPARATIVE DEVICE PERFORMANCE DOMINANCE MATRIX 🏆")
    print("="*w)
    hdr = (f"{'Device':<8}{'Dataset':<14}{'Topology':<20}"
           f"{'Std Time':>10}{'DAD Time':>10}"
           f"{'Std Thru':>12}{'DAD Thru':>12}"
           f"{'Std Acc':>9}{'DAD Acc':>9}"
           f"{'Speed Δ':>12}")
    print(hdr)
    print("-"*w)
    
    for r in results:
        # Determine winning algorithm
        if r['std_t'] > r['goat_t']:
            gain = (r['std_t'] / r['goat_t'])
            winner = f"DAD+{((gain - 1.0)*100):.0f}% ({gain:.2f}x)"
        else:
            gain = (r['goat_t'] / r['std_t'])
            winner = f"Std+{((gain - 1.0)*100):.0f}% ({gain:.2f}x)"
            
        row = (f"{r['Device']:<8}{r['Dataset']:<14}{r['Topology']:<20}"
               f"{r['std_t']:>9.2f}s{r['goat_t']:>9.2f}s"
               f"{r['std_thru']:>10.0f}/s{r['goat_thru']:>10.0f}/s"
               f"{r['std_a']:>8.2f}%{r['goat_a']:>8.2f}%"
               f"{winner:>12}")
        print(row)
    print("="*w + "\n")

if __name__ == "__main__":
    all_results = []
    
    # 1. Run GPU Benchmarks (if available)
    if torch.cuda.is_available():
        gpu_results = run_benchmark_on_device(force_cpu=False)
        all_results.extend(gpu_results)
        
    # 2. Run CPU Benchmarks
    cpu_results = run_benchmark_on_device(force_cpu=True)
    all_results.extend(cpu_results)
    
    # 3. Print Unified Comparative Device Dominance Matrix
    print_dominance_matrix(all_results)
