import torch
import torch.nn as nn
import torch.nn.functional as F

def adamw_step_fn(p, g, m, v, lr, t):
    """Fused tensor-operation AdamW optimization step."""
    b1, b2, eps = 0.9, 0.999, 1e-8
    
    # Update moment estimates in-place
    m.mul_(b1).add_(g, alpha=1.0 - b1)
    v.mul_(b2).addcmul_(g, g, value=1.0 - b2)
    
    bias_correction1 = 1.0 - b1 ** t
    bias_correction2 = 1.0 - b2 ** t
    
    step_size = lr / bias_correction1
    denom = (v.sqrt() / torch.sqrt(bias_correction2)).add_(eps)
    
    # In-place weight decay and gradient descent step
    p.mul_(1.0 - lr * 0.01)
    p.addcdiv_(m, denom, value=-step_size)

class DADLinear(nn.Module):
    """
    Decoupled Analytical Dense (DAD) target propagation layer.
    Acts as a drop-in high-performance alternative to nn.Linear for backprop-free networks.
    """
    def __init__(self, in_features, out_features, num_classes=10, device=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_classes = num_classes
        self.device = device

        # Main Weight & Bias (No autograd gradients tracked)
        self.W = nn.Parameter(
            torch.randn(out_features, in_features, device=device) * (2.0 / in_features) ** 0.5,
            requires_grad=False
        )
        self.bias = nn.Parameter(
            torch.zeros(out_features, device=device),
            requires_grad=False
        )

        # Local Task Classifier Head (Trainable)
        self.W_loc = nn.Parameter(
            torch.randn(num_classes, out_features, device=device) * 0.02,
            requires_grad=False
        )
        self.b_loc = nn.Parameter(
            torch.zeros(num_classes, device=device),
            requires_grad=False
        )

        # Optimizer Moments (No Autograd tracking)
        self.m_W = nn.Parameter(torch.zeros_like(self.W, device=device), requires_grad=False)
        self.v_W = nn.Parameter(torch.zeros_like(self.W, device=device), requires_grad=False)
        self.m_bias = nn.Parameter(torch.zeros_like(self.bias, device=device), requires_grad=False)
        self.v_bias = nn.Parameter(torch.zeros_like(self.bias, device=device), requires_grad=False)

    def forward(self, x):
        z = F.linear(x, self.W, self.bias)
        out = torch.relu(z)
        return out, z
