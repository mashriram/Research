import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Fix JIT recompilation: allow nn.Module integer attributes to be dynamic
import torch._dynamo
torch._dynamo.config.allow_unspec_int_on_nn_module = True

class GOATDTPLinear(nn.Module):
    """
    Generalistic Decoupled Target Propagation (DTP) Block.

    Key design:
      - Forward returns `out` WITHOUT detaching from the graph.
        This allows global backprop to flow gradients into A, B, S for the
        last layer in the stack (giving free representation learning).
      - The analytical local update uses x.detach() so it never conflicts
        with the global backward pass.
      - step_counter is a Python int but we gate mask refresh via a flag
        buffer to avoid torch.compile recompilation triggers.
    """
    def __init__(self, in_features, out_features, num_classes=10,
                 rank=None, sparsity=0.8, lr=1e-3):
        super().__init__()
        self.sparsity = sparsity
        self.lr = lr

        # Default rank: sqrt(min(in, out)) — keeps param count reasonable
        if rank is None:
            rank = max(8, int(min(in_features, out_features) ** 0.5 * 4))

        self.A = nn.Parameter(torch.randn(out_features, rank) * 0.02)
        self.B = nn.Parameter(torch.randn(rank, in_features) * 0.02)
        # Full dense S matrix; sparsity applied via mask
        self.S = nn.Parameter(torch.randn(out_features, in_features) * 0.001)
        self.bias = nn.Parameter(torch.zeros(out_features))

        # Local Classification Head (never participates in global graph)
        self.W_local = nn.Parameter(torch.randn(num_classes, out_features) * 0.02)
        self.b_local = nn.Parameter(torch.zeros(num_classes))

        # Analytical AdamW moment buffers (local params)
        for name in ['W', 'b_local', 'A', 'B', 'S', 'bias']:
            ref = self.W_local if name == 'W' else \
                  self.b_local if name == 'b_local' else \
                  getattr(self, name)
            self.register_buffer(f'm_{name}', torch.zeros_like(ref))
            self.register_buffer(f'v_{name}', torch.zeros_like(ref))

        # Mask refresh control — use a tensor buffer so JIT doesn't track
        # the Python int for recompilation decisions
        self.register_buffer('_refresh_flag', torch.ones(1, dtype=torch.bool))
        self.register_buffer('cached_mask', torch.zeros(out_features, in_features))
        self.step_counter = 0        # plain Python int, fine because we
        self.mask_refresh_every = 50  # gate with the bool buffer, not modulo

    # -----------------------------------------------------------------
    # Sparse mask — only recomputed when _refresh_flag is set
    # -----------------------------------------------------------------
    @torch.no_grad()
    def _refresh_mask(self):
        k = int(self.S.numel() * (1.0 - self.sparsity))
        if k <= 0:
            self.cached_mask.zero_()
            return
        vals = self.S.abs().view(-1)
        threshold = torch.topk(vals, k).values[-1]
        self.cached_mask.copy_((self.S.abs() >= threshold).float())
        self._refresh_flag.fill_(False)

    def get_sparse_mask(self):
        if self._refresh_flag.item():
            self._refresh_mask()
        return self.cached_mask

    # -----------------------------------------------------------------
    # Analytical AdamW helper
    # -----------------------------------------------------------------
    @torch.no_grad()
    def _adamw_step(self, param, grad, m, v, t):
        beta1, beta2, eps, wd = 0.9, 0.999, 1e-8, 0.01
        bc1 = 1.0 - beta1 ** t
        bc2 = 1.0 - beta2 ** t
        step_size = self.lr / bc1

        m.mul_(beta1).add_(grad, alpha=1.0 - beta1)
        v.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
        denom = (v.sqrt() / math.sqrt(bc2)).add_(eps)
        param.data.mul_(1.0 - self.lr * wd)
        param.data.addcdiv_(m, denom, value=-step_size)

    # -----------------------------------------------------------------
    # Forward
    # -----------------------------------------------------------------
    def forward(self, x, local_target=None):
        # Use detached copy ONLY for the local analytical path
        x_d = x.detach()
        mask = self.get_sparse_mask()

        # LoSparse forward (connected to global graph via A, B, S, bias)
        U = x_d @ self.B.t()            # (B, rank)
        z = U @ self.A.t() + F.linear(x_d, self.S * mask) + self.bias
        out = torch.relu(z)

        if self.training and local_target is not None:
            with torch.no_grad():
                t = self.step_counter + 1

                # Local logits + cross-entropy gradient
                y_pred = F.linear(out, self.W_local, self.b_local)
                probs  = F.softmax(y_pred, dim=-1)
                y_true = F.one_hot(local_target, num_classes=probs.size(-1)).float()
                d_pred = (probs - y_true) / x.size(0)

                # Head gradients
                g_W     = d_pred.t() @ out
                g_b_loc = d_pred.sum(0)

                # Back-project through ReLU
                d_h = d_pred @ self.W_local
                d_z = d_h * (z > 0).float()

                # GOAT LoSparse gradients
                g_A    = d_z.t() @ U
                g_B    = (d_z @ self.A).t() @ x_d
                g_S    = (d_z.t() @ x_d) * mask
                g_bias = d_z.sum(0)

                # Apply analytical AdamW to ALL local params
                self._adamw_step(self.W_local, g_W,     self.m_W,      self.v_W,      t)
                self._adamw_step(self.b_local, g_b_loc, self.m_b_local, self.v_b_local, t)
                self._adamw_step(self.A,       g_A,     self.m_A,      self.v_A,      t)
                self._adamw_step(self.B,       g_B,     self.m_B,      self.v_B,      t)
                self._adamw_step(self.S,       g_S,     self.m_S,      self.v_S,      t)
                self._adamw_step(self.bias,    g_bias,  self.m_bias,   self.v_bias,   t)

            self.step_counter += 1
            # Schedule mask refresh
            if self.step_counter % self.mask_refresh_every == 0:
                self._refresh_flag.fill_(True)

        return out
