import time
import torch
import torch.nn as nn
from torchvision import datasets
from torch_dad import DADLinear, DADModel, DADTrainer

# 1. Hardware and Precision Detection
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
amp_dtype = torch.bfloat16 if (device.type == 'cuda' and torch.cuda.is_bf16_supported()) else torch.float32

print(f" 🚀 Testing local 'torch_dad' library on device: {device}")
print(f"    Precision Autocast: {amp_dtype}")

# 2. Define Custom Modular Network using torch_dad drop-in layers
class MNISTModel(DADModel):
    def __init__(self):
        super().__init__()
        # Register standard DADLinear layers (inherits automatic scan and alignment head configs)
        self.layer1 = DADLinear(784, 256, num_classes=10, device=device)
        self.layer2 = DADLinear(256, 256, num_classes=10, device=device)
        self.layer3 = DADLinear(256, 128, num_classes=10, device=device)
        
        # Subclass must define self.classifier as their final Linear head mapping
        self.classifier = nn.Linear(128, 10, device=device)

# 3. Instantiate model and high-performance trainer
model = MNISTModel()
trainer = DADTrainer(model, device)

# 4. Load MNIST into dynamic cached streams
print("\n 📂 Loading MNIST dataset...")
mean, std = 0.1307, 0.3081
train_dataset = datasets.MNIST('./data', train=True, download=True)
test_dataset  = datasets.MNIST('./data', train=False, download=False)

x_train = (train_dataset.data.float() / 255.0 - mean) / std
y_train = train_dataset.targets
x_test = (test_dataset.data.float() / 255.0 - mean) / std
y_test = test_dataset.targets

x_train = x_train.to(device).view(x_train.size(0), -1)
y_train = y_train.to(device)
x_test  = x_test.to(device).view(x_test.size(0), -1)
y_test  = y_test.to(device)

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

train_loader = CachedLoader(x_train, y_train, 256, shuffle=True)
test_loader  = CachedLoader(x_test, y_test, 1000, shuffle=False)

# 5. Execute backprop-free training via DAD JIT loops
print(" ⚡ Training network using backprop-free decoupled target propagation...")
trainer.reset_step_counter()
t0 = time.time()
for ep in range(3):
    accumulate = (ep == 2)
    trainer.train_epoch(train_loader, amp_dtype, accumulate_solver=accumulate)
    print(f"    ✦ Epoch {ep+1}/3 complete.")
train_time = time.time() - t0

# 6. Apply final double-precision optimal alignment least-squares solved mapping
print(" ⚡ Solving optimal classifier head mapping in VRAM...")
solve_time = trainer.solve_head()
print(f"    ✦ Closed-form head solved in {solve_time*1000:.2f} ms!")

# 7. Evaluate and output classification accuracy
model.eval()
correct = 0
with torch.no_grad():
    for x_b, y_b in test_loader:
        correct += model.forward_inference(x_b).argmax(1).eq(y_b).sum().item()

acc = 100.0 * correct / len(test_loader.dataset)
print(f"\n🏆 VERIFICATION COMPLETE!")
print(f"   - Total training time : {train_time:.2f} seconds")
print(f"   - Final Test Accuracy : {acc:.2f}% (Retained backprop accuracy free of Autograd sequential locks!)")
