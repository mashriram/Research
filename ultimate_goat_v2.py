import os
os.environ["OMP_NUM_THREADS"] = str(os.cpu_count())
os.environ["MKL_NUM_THREADS"] = str(os.cpu_count())
os.environ["KMP_AFFINITY"] = "granularity=fine,compact,1,0"

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import torch._functorch.config

torch._functorch.config.donated_buffer = False

class GoliathV2(optim.Optimizer):
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.95), rho=0.05, sam_freq=0.5, weight_decay=0.01):
        defaults = dict(lr=lr, betas=betas, rho=rho, sam_freq=sam_freq, weight_decay=weight_decay)
        super(GoliathV2, self).__init__(params, defaults)

    @torch.no_grad()
    def first_step(self):
        if torch.rand(1).item() > self.param_groups[0]['sam_freq']:
            return False
            
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            rho = group['rho']
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                state['old_p'] = p.data.clone()
                # Adversarial perturbation
                p.data = p.data + p.grad * (rho / (grad_norm + 1e-12))
        return True

    def _grad_norm(self):
        norm = 0.0
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    norm += p.grad.norm(2).item() ** 2
        return math.sqrt(norm)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group['betas']
            lr = group['lr']
            weight_decay = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue
                
                state = self.state[p]
                
                if len(state) == 0 or (len(state) == 1 and 'old_p' in state):
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['hessian'] = torch.ones_like(p.data)
                    
                if 'old_p' in state:
                    p.data = state.pop('old_p')

                # Hutchinson Curvature
                state['hessian'].mul_(beta2).addcmul_(p.grad, p.grad, value=1 - beta2)
                
                # Lion-Sign Update
                update = state['exp_avg'].mul(beta1).add(p.grad, alpha=1 - beta1).sign()
                
                # Adaptive step
                denom = state['hessian'].clamp(min=1e-4)
                p.data = p.data - lr * (update / denom)
                
                # Decay & Momentum
                if weight_decay != 0:
                    p.data = p.data - lr * weight_decay * p.data
                
                state['exp_avg'].mul_(beta2).add_(p.grad, alpha=1 - beta2)

        return loss

class SharedTeacher(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.brain = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim)
        )
        self.teacher_opt = optim.AdamW(self.parameters(), lr=1e-4)

    def get_instruction(self, activations):
        return self.brain(activations)

class GOATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, teacher, rank=16, sparsity=0.9):
        super().__init__()
        self.teacher = teacher
        self.sparsity = sparsity
        
        self.A = nn.Parameter(torch.randn(out_dim, rank) * 0.02)
        self.B = nn.Parameter(torch.randn(rank, in_dim) * 0.02)
        self.S = nn.Parameter(torch.randn(out_dim, in_dim) * 0.001)
        self.bias = nn.Parameter(torch.zeros(out_dim))
        
        self.mirror = nn.Sequential(nn.Linear(out_dim, out_dim))
        
        self.main_opt = GoliathV2([self.A, self.B, self.S, self.bias])
        self.mirror_opt = optim.AdamW(self.mirror.parameters(), lr=1e-4)

    @torch.no_grad()
    def get_sparse_mask(self):
        k = int(self.S.numel() * (1 - self.sparsity))
        if k <= 0: return torch.zeros_like(self.S)
        threshold = torch.topk(self.S.abs().view(-1), k).values[-1]
        return (self.S.abs() >= threshold).float()

    def forward(self, x):
        x_detached = x.detach() # CRITICAL: Prevents autograd from crossing layer boundaries
        mask = self.get_sparse_mask()
        
        lr_out = (x_detached @ self.B.t()) @ self.A.t()
        sparse_out = F.linear(x_detached, self.S * mask)
        out = torch.relu(lr_out + sparse_out + self.bias)
        
        if self.training:
            local_grad = self.mirror(out.detach())
            loss = F.mse_loss(out, local_grad.detach())
            
            self.main_opt.zero_grad()
            self.main_opt.zero_grad()
            loss.backward()
            did_sam = self.main_opt.first_step()
            
            if did_sam:
                mask_adv = self.get_sparse_mask()
                lr_out_adv = (x_detached @ self.B.t()) @ self.A.t()
                sparse_out_adv = F.linear(x_detached, self.S * mask_adv)
                out_adv = torch.relu(lr_out_adv + sparse_out_adv + self.bias)
                
                local_grad_adv = self.mirror(out_adv.detach())
                loss_adv = F.mse_loss(out_adv, local_grad_adv.detach())
                loss_adv.backward()
                
            self.main_opt.step()
            
            teacher_signal = self.teacher.get_instruction(out.detach())
            mirror_loss = F.mse_loss(local_grad, teacher_signal)
            
            self.mirror_opt.zero_grad()
            mirror_loss.backward()
            self.mirror_opt.step()
            
        return out

class TheUltimateGOATV2(nn.Module):
    def __init__(self, in_dim=784, hidden_dim=256, num_layers=4, classes=10):
        super().__init__()
        self.teacher = SharedTeacher(hidden_dim)
        self.in_proj = nn.Linear(in_dim, hidden_dim)
        self.layers = nn.ModuleList([GOATLayer(hidden_dim, hidden_dim, self.teacher) for _ in range(num_layers)])
        self.head = nn.Linear(hidden_dim, classes)

    def forward(self, x):
        x = torch.relu(self.in_proj(x))
        for layer in self.layers:
            x = layer(x)
        return self.head(x.detach())

def train():
    model = TheUltimateGOATV2(in_dim=784, hidden_dim=256, num_layers=4, classes=10)
    
    # Note: torch.compile currently does not support nested .backward() calls inside 
    # the forward pass during AOTAutograd tracing (raises graph errors).
    # Eager mode executes the NPLM architecture correctly.
    # model = torch.compile(model, mode="max-autotune-no-cudagraphs", dynamic=True)
    
    head_opt = optim.AdamW(model.head.parameters(), lr=1e-3)
    
    X = torch.randn(128, 784)
    Y = torch.randint(0, 10, (128,))
    
    print("Starting training loop...")
    model.train()
    for epoch in range(50):
        head_opt.zero_grad()
        model.teacher.teacher_opt.zero_grad()
        
        logits = model(X)
        
        loss = F.cross_entropy(logits, Y)
        
        loss.backward()
        
        head_opt.step()
        model.teacher.teacher_opt.step()
        
        print(f"Epoch {epoch + 1}/50 | Loss: {loss.item():.4f}")

if __name__ == "__main__":
    train()
