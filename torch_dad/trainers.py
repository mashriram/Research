import time
import torch
import torch.nn.functional as F
from .layers import adamw_step_fn

class DADTrainer:
    """
    Decoupled Analytical Dense (DAD) Training Engine.
    Manages JIT step training loops, autocasting, and final closed-form classifier solving.
    """
    def __init__(self, model, device: torch.device):
        self.model = model
        self.device = device
        
        # Verify model has DAD layers
        dad_layers = model.dad_layers
        if not dad_layers:
            raise ValueError("Provided model has no DADLinear layers registered!")
            
        self.num_classes = dad_layers[0].num_classes
        self.step_counter = 0
        self.t_tensor = torch.tensor(0.0, dtype=torch.float32, device=device)

        # Dynamic JIT Compilation: JIT compile only on GPU to bypass compilation latency on CPU
        if device.type == 'cuda':
            self.compiled_step = torch.compile(self.unified_step)
        else:
            self.compiled_step = self.unified_step

        # Pre-allocate closed-form solving matrices based on final layer output dimension
        last_layer = dad_layers[-1]
        out_features = last_layer.W.size(0)
        self.HTH = torch.zeros(out_features + 1, out_features + 1, device=device)
        self.HTY = torch.zeros(out_features + 1, self.num_classes, device=device)

    def reset_step_counter(self):
        """Resets the training step counters and moment updates."""
        self.step_counter = 0
        self.t_tensor.zero_()

    def unified_step(self, x, y, lr, t):
        """Unified, JIT-fusible forward-backward training step."""
        # 1. Forward Pass
        acts = [x.view(x.size(0), -1)]
        zs = []
        x_d = acts[0]
        for layer in self.model.dad_layers:
            out, z = layer(x_d)
            acts.append(out)
            zs.append(z)
            x_d = out
            
        # 2. Decoupled Backward Pass
        y_true = F.one_hot(y, num_classes=self.num_classes).float()
        inv_batch_size = 1.0 / x.size(0)
        
        for i, layer in enumerate(self.model.dad_layers):
            out = acts[i+1]
            z = zs[i]
            x_prev = acts[i]
            
            # Local alignment projection
            y_pred = F.linear(out, layer.W_loc, layer.b_loc)
            probs  = F.softmax(y_pred, dim=-1)
            d_pred = (probs - y_true) * inv_batch_size

            # Analytical local gradients calculation
            g_W_loc = d_pred.t() @ out
            g_b_loc = d_pred.sum(0)
            d_h     = d_pred @ layer.W_loc
            d_z     = d_h * (z > 0).float()
            g_W     = d_z.t() @ x_prev
            g_bias  = d_z.sum(0)

            # In-place updates: Fused AdamW for main, ultra-fast SGD for local classifiers
            adamw_step_fn(layer.W,     g_W,     layer.m_W,     layer.v_W,     lr, t)
            adamw_step_fn(layer.bias,  g_bias,  layer.m_bias,  layer.v_bias,  lr, t)
            layer.W_loc.add_(g_W_loc, alpha=-1e-3)
            layer.b_loc.add_(g_b_loc, alpha=-1e-3)

        return acts[-1]

    @torch.no_grad()
    def train_epoch(self, loader, amp_dtype, accumulate_solver=False):
        """Trains the model for one full epoch, with optional closed-form matrix accumulations."""
        self.model.train()
        for x, y in loader:
            self.step_counter += 1
            self.t_tensor.add_(1.0)
            
            with torch.amp.autocast(self.device.type, dtype=amp_dtype, enabled=(self.device.type == 'cuda')):
                out_last = self.compiled_step(x, y, 1e-3, self.t_tensor)
                
                # Eager solver accumulation outside JIT to guarantee zero graph breaks
                if accumulate_solver:
                    ones = torch.ones(out_last.size(0), 1, device=self.device)
                    h_aug = torch.cat([out_last, ones], dim=1)
                    y_onehot = F.one_hot(y, num_classes=self.num_classes).float()
                    self.HTH.add_(h_aug.t() @ h_aug)
                    self.HTY.add_(h_aug.t() @ y_onehot)

    @torch.no_grad()
    def solve_head(self, lambda_reg=1e-3):
        """Instantly solves the optimal final classifier linear mapping in VRAM."""
        t0 = time.time()
        out_features = self.model.dad_layers[-1].W.size(0)
        
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
