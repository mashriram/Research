"""Phase 1 Local Pilot Trainer for DEEPA (Delta-rule Enhanced Efficient Pairing Architecture).

Features:
- Exact winning DEEPA 3:1 hybrid motif: pattern = 'ddda' * (layers // 4)
  - 'd' layers: FusedSheafDeltaMixer (Triton chunk_gated_delta_rule + zero-copy thin Householder dyadic transport)
  - 'a' layers: Pre-RMSNorm Causal Self-Attention with plain RoPE (FlashAttention SDPA) + SwiGLU MLP
- Presets for both 8 GB GPUs (RTX 2000 Ada) and 16 GB GPUs (RTX 5070 Ti):
  - --preset 50m   (~51M params, dim=576, 12 layers, 9 heads, fits in <6 GB VRAM without checkpointing)
  - --preset 110m  (~112M params, dim=768, 12 layers, 12 heads, fits in 16 GB natively or <4 GB with --ckpt)
- Whole-model torch.compile (--compile) + bf16 mixed precision + gradient accumulation
- Proper AdamW weight decay groups (0.1 on 2D+ weights, 0.0 on 1D norms/biases/gates)
- Automatic checkpoint saving/resuming, multi-length evaluation (@1024, @2048, @4096), and O(1)-state sample generation.

Usage:
  # Dry-run / memory & speed probe (3 steps):
  .venv/bin/python train_deepa_phase1.py --preset 110m --ckpt --probe

  # Full Phase 1 training on RTX 5070 Ti (16 GB) or RTX 2000 Ada (8 GB with --ckpt):
  .venv/bin/python train_deepa_phase1.py --preset 110m --compile --ckpt
"""
import os, sys, time, math, json, copy, argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from torch.utils.checkpoint import checkpoint
from hsnlm import Config, ByteData
from hsnlm.models import LM, HBlock, RMSNorm, MLP, Attn


class SwiGLUTBlock(nn.Module):
    """Pre-RMSNorm RoPE Attention block with SwiGLU MLP (matches HBlock's gated MLP)."""
    def __init__(s, c: Config):
        super().__init__()
        s.n1 = RMSNorm(c.dim)
        s.n2 = RMSNorm(c.dim)
        s.mix = Attn(c)
        s.mlp = MLP(c.dim, c.mlp_mult, gated=True)

    def forward(s, x):
        x = x + s.mix(s.n1(x))
        return x + s.mlp(s.n2(x))

    def step(s, x, st):
        y, st = s.mix.step(s.n1(x), st)
        x = x + y
        return x + s.mlp(s.n2(x)), st


class DEEPA(LM):
    """DEEPA: Delta-rule Enhanced Efficient Pairing Architecture (3:1 Sheaf-Delta : RoPE-Attention)."""
    use_pos = False

    def __init__(s, c: Config, use_ckpt: bool = False):
        s.use_ckpt = use_ckpt
        if not c.pattern:
            assert c.layers % 4 == 0, "DEEPA layers must be a multiple of 4 for the 3:1 (ddda) motif"
            c.pattern = "ddda" * (c.layers // 4)
        super().__init__(c)

    def make_block(s, i):
        c = copy.copy(s.c)
        ch = s.c.pattern[i]
        if ch == "a":
            return SwiGLUTBlock(c)
        c.mixer = "sheafdelta-fused"
        return HBlock(c)

    def forward(s, x):
        h = s.embed_in(x)
        for b in s.blocks:
            h = checkpoint(b, h, use_reentrant=False) if (s.training and s.use_ckpt) else b(h)
        return s.head(h)


PRESETS = {
    "15m":  dict(dim=384,  layers=8,  heads=8,  mlp_mult=3.65, batch=16, grad_accum=1),
    "50m":  dict(dim=576,  layers=12, heads=9,  mlp_mult=3.65, batch=8,  grad_accum=4),
    "86m":  dict(dim=768,  layers=12, heads=12, mlp_mult=3.65, batch=8,  grad_accum=8),
    "115m": dict(dim=768,  layers=16, heads=12, mlp_mult=3.65, batch=8,  grad_accum=8),
}


def build_optimizer(model: nn.Module, lr: float, weight_decay: float):
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim >= 2 and not name.endswith(".V"):
            decay.append(p)
        else:
            no_decay.append(p)
    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=lr, betas=(0.9, 0.95), fused=torch.cuda.is_available())


@torch.no_grad()
def evaluate_bpb(model: nn.Module, data: ByteData, ctx_len: int, nwin: int = 64, batch_val: int = 4) -> float:
    model.eval()
    v = data.val
    n = min(nwin, (len(v) - 1) // ctx_len)
    tot, cnt = 0.0, 0
    for i in range(0, n, batch_val):
        b_end = min(i + batch_val, n)
        xs = torch.stack([v[j * ctx_len:(j + 1) * ctx_len] for j in range(i, b_end)]).cuda()
        ys = torch.stack([v[j * ctx_len + 1:(j + 1) * ctx_len + 1] for j in range(i, b_end)]).cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(xs)
            tot += F.cross_entropy(logits.reshape(-1, data.V), ys.reshape(-1), reduction="sum").item()
        cnt += ys.numel()
    model.train()
    return tot / cnt / math.log(2)


@torch.no_grad()
def generate_sample(model: DEEPA, data: ByteData, prompt_len: int = 64, gen_len: int = 128) -> str:
    model.eval()
    prompt = data.val[:prompt_len].unsqueeze(0).cuda()
    out = model.generate(prompt, gen_len)[0].cpu().tolist()
    model.train()
    return bytes(out).decode("utf-8", errors="replace")


def main():
    p = argparse.ArgumentParser(description="Train DEEPA Phase 1 Local Pilot")
    p.add_argument("--preset", choices=list(PRESETS.keys()), default="86m")
    p.add_argument("--data", type=str, default="data/enwik8")
    p.add_argument("--ctx", type=int, default=1024)
    p.add_argument("--batch", type=int, default=None, help="Micro-batch size per step")
    p.add_argument("--grad-accum", type=int, default=None, help="Gradient accumulation steps")
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--lr", type=float, default=1.5e-3)
    p.add_argument("--wd", type=float, default=0.1)
    p.add_argument("--warmup", type=int, default=150)
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--compile", action="store_true", help="Compile whole model with torch.compile")
    p.add_argument("--ckpt", action="store_true", help="Enable block-level activation checkpointing")
    p.add_argument("--probe", action="store_true", help="Run 3 steps to verify VRAM and speed, then exit")
    p.add_argument("--out-dir", type=str, default="checkpoints/deepa_phase1")
    p.add_argument("--resume", action="store_true", help="Resume from latest.pt in out-dir")
    p.add_argument("--seed", type=int, default=42)
    args = parser_args = p.parse_args()

    assert torch.cuda.is_available(), "CUDA GPU required for Phase 1 training"
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(args.seed)
    gen = torch.Generator().manual_seed(args.seed)

    preset = PRESETS[args.preset]
    B = args.batch or preset["batch"]
    grad_accum = args.grad_accum or preset["grad_accum"]
    pattern = "ddda" * (preset["layers"] // 4)

    cfg = Config(
        vocab=256,
        dim=preset["dim"],
        layers=preset["layers"],
        heads=preset["heads"],
        ctx=args.ctx,
        mlp_mult=preset["mlp_mult"],
        pattern=pattern,
        mixer="sheafdelta-fused",
    )

    raw_model = DEEPA(cfg, use_ckpt=args.ckpt).cuda()
    n_params = raw_model.n_params()
    tok_per_step = B * args.ctx * grad_accum

    print(f"=== DEEPA Phase 1 Pilot ({args.preset.upper()}) ===")
    print(f"Device       : {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB)")
    print(f"Architecture : {cfg.layers} layers, pattern='{cfg.pattern}' (3:1 Sheaf-Delta : RoPE Attention)")
    print(f"Dimensions   : dim={cfg.dim}, heads={cfg.heads} (dh={cfg.dim // cfg.heads}), params={n_params:,} ({n_params/1e6:.2f}M)")
    print(f"Batch Config : micro_batch={B}, ctx={args.ctx}, grad_accum={grad_accum} -> {tok_per_step:,} tok/step")
    print(f"Optimization : lr={args.lr}, wd={args.wd}, compile={args.compile}, grad_ckpt={args.ckpt}")

    data = ByteData(args.data, ctx=args.ctx, n_train=90_000_000, n_val=5_000_000)
    opt = build_optimizer(raw_model, lr=args.lr, weight_decay=args.wd)

    os.makedirs(args.out_dir, exist_ok=True)
    ckpt_path = os.path.join(args.out_dir, f"deepa_{args.preset}_latest.pt")
    log_path = os.path.join(args.out_dir, f"deepa_{args.preset}_log.json")

    start_step = 0
    history = []
    if args.resume and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location="cuda")
        raw_model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        start_step = ck["step"]
        history = ck.get("history", [])
        print(f"Resumed from {ckpt_path} at step {start_step}")

    model = torch.compile(raw_model) if args.compile else raw_model

    if args.compile or args.probe:
        print("Warming up model forward/backward...")
        x0, y0 = data.batch(B, torch.Generator().manual_seed(999))
        x0, y0 = x0.cuda(), y0.cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss0 = F.cross_entropy(model(x0).reshape(-1, cfg.vocab), y0.reshape(-1))
        loss0.backward()
        opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    total_steps = (start_step + 3) if args.probe else args.steps

    for s in range(start_step, total_steps):
        # Cosine schedule with linear warmup
        warm = min(1.0, (s + 1) / max(1, args.warmup))
        prog = (s - args.warmup) / max(1, args.steps - args.warmup)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, max(0.0, prog))))
        cur_lr = args.lr * warm * (0.1 + 0.9 * cosine)
        for g in opt.param_groups:
            g["lr"] = cur_lr

        opt.zero_grad(set_to_none=True)
        step_loss = 0.0
        for _ in range(grad_accum):
            x, y = data.batch(B, gen)
            x, y = x.cuda(), y.cuda()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = F.cross_entropy(model(x).reshape(-1, cfg.vocab), y.reshape(-1)) / grad_accum
            loss.backward()
            step_loss += loss.item()

        grad_norm = torch.nn.utils.clip_grad_norm_(raw_model.parameters(), 1.0).item()
        opt.step()

        if args.probe or (s + 1) % 25 == 0:
            torch.cuda.synchronize()
            elapsed = time.time() - t0
            steps_done = s - start_step + 1
            tok_s = steps_done * tok_per_step / elapsed
            peak_mb = torch.cuda.max_memory_allocated() / 1e6
            train_bpb = step_loss / math.log(2)
            print(
                f"step {s+1:5d}/{total_steps} | loss {train_bpb:.4f} bpb | lr {cur_lr:.2e} | "
                f"gnorm {grad_norm:.2f} | {tok_s:,.0f} tok/s | peak {peak_mb:,.0f} MB | {elapsed:.1f}s",
                flush=True,
            )

        if not args.probe and ((s + 1) % args.eval_every == 0 or (s + 1) == total_steps):
            val_1024 = evaluate_bpb(raw_model, data, args.ctx, nwin=64)
            val_2048 = evaluate_bpb(raw_model, data, args.ctx * 2, nwin=32)
            rec = dict(
                step=s + 1,
                train_bpb=round(step_loss / math.log(2), 4),
                val_bpb_1024=round(val_1024, 4),
                val_bpb_2048=round(val_2048, 4),
                peak_mem_mb=round(torch.cuda.max_memory_allocated() / 1e6, 1),
                elapsed_s=round(time.time() - t0, 1),
            )
            if (s + 1) == total_steps:
                rec["val_bpb_4096"] = round(evaluate_bpb(raw_model, data, args.ctx * 4, nwin=32, batch_val=2), 4)
            history.append(rec)
            print(f"  --> EVAL @ step {s+1}: val@1024={val_1024:.4f} bpb | val@2048={val_2048:.4f} bpb", flush=True)
            torch.save({"step": s + 1, "model": raw_model.state_dict(), "opt": opt.state_dict(), "cfg": cfg.__dict__, "history": history}, ckpt_path)
            json.dump(history, open(log_path, "w"), indent=2)

    if args.probe:
        sample = generate_sample(raw_model, data, prompt_len=32, gen_len=32)
        print(f"Probe complete! Verified O(1)+KV step() generation ({len(sample)} chars). Ready for full run.")


if __name__ == "__main__":
    main()
