#!/usr/bin/env python3
"""Generate publication-grade vector PDF figures for the DEEPA Springer LNCS manuscript using PyCairo."""
import math
import os
import cairo

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs")
os.makedirs(OUT_DIR, exist_ok=True)

# Clean Springer-friendly palette
C_DEEPA = (0.09, 0.42, 0.68)      # Deep academic blue
C_DEEPA_LT = (0.86, 0.93, 0.98)   # Soft blue fill
C_ALIBI = (0.85, 0.45, 0.12)      # Warm amber/orange
C_ALIBI_LT = (0.99, 0.93, 0.86)   # Soft amber fill
C_ROPE = (0.76, 0.20, 0.22)       # Crimson red
C_ROPE_LT = (0.98, 0.89, 0.89)    # Soft red fill
C_GREEN = (0.16, 0.56, 0.34)      # Emerald green
C_GREEN_LT = (0.88, 0.96, 0.91)   # Soft green fill
C_PURPLE = (0.45, 0.28, 0.65)     # Muted purple
C_PURPLE_LT = (0.93, 0.90, 0.97)
C_GRAY = (0.35, 0.37, 0.40)
C_LGRAY = (0.85, 0.86, 0.88)
C_BG = (0.98, 0.985, 0.99)
C_DARK = (0.12, 0.14, 0.17)


def set_font(ctx, size, bold=False, italic=False, family="Sans"):
    slant = cairo.FONT_SLANT_ITALIC if italic else cairo.FONT_SLANT_NORMAL
    weight = cairo.FONT_WEIGHT_BOLD if bold else cairo.FONT_WEIGHT_NORMAL
    ctx.select_font_face(family, slant, weight)
    ctx.set_font_size(size)


def draw_text(ctx, x, y, text, size=8.5, bold=False, italic=False, color=C_DARK, align="left", valign="baseline", family="Sans"):
    set_font(ctx, size, bold, italic, family)
    ext = ctx.text_extents(text)
    tx = x
    if align == "center":
        tx = x - (ext.width / 2 + ext.x_bearing)
    elif align == "right":
        tx = x - (ext.width + ext.x_bearing)
    ty = y
    if valign == "center":
        ty = y - (ext.height / 2 + ext.y_bearing)
    elif valign == "top":
        ty = y - ext.y_bearing
    ctx.set_source_rgb(*color)
    ctx.move_to(tx, ty)
    ctx.show_text(text)


def round_rect(ctx, x, y, w, h, r=4):
    ctx.new_sub_path()
    ctx.arc(x + w - r, y + r, r, -math.pi / 2, 0)
    ctx.arc(x + w - r, y + h - r, r, 0, math.pi / 2)
    ctx.arc(x + r, y + h - r, r, math.pi / 2, math.pi)
    ctx.arc(x + r, y + r, r, math.pi, 3 * math.pi / 2)
    ctx.close_path()


def draw_box(ctx, x, y, w, h, fill, stroke, lw=1.0, r=4, dashed=False):
    round_rect(ctx, x, y, w, h, r)
    ctx.set_source_rgb(*fill)
    ctx.fill_preserve()
    ctx.set_source_rgb(*stroke)
    ctx.set_line_width(lw)
    if dashed:
        ctx.set_dash([3.5, 2.5])
    ctx.stroke()
    if dashed:
        ctx.set_dash([])


def draw_arrow(ctx, x1, y1, x2, y2, color=C_GRAY, lw=1.0, head=5.0):
    ctx.set_source_rgb(*color)
    ctx.set_line_width(lw)
    ctx.move_to(x1, y1)
    ctx.line_to(x2, y2)
    ctx.stroke()
    ang = math.atan2(y2 - y1, x2 - x1)
    ctx.move_to(x2, y2)
    ctx.line_to(x2 - head * math.cos(ang - 0.42), y2 - head * math.sin(ang - 0.42))
    ctx.line_to(x2 - head * math.cos(ang + 0.42), y2 - head * math.sin(ang + 0.42))
    ctx.close_path()
    ctx.fill()


# =====================================================================
# FIGURE 1: DEEPA 3:1 Architecture & Information Flow Diagram
# =====================================================================
def make_fig1():
    W, H = 560, 355
    path = os.path.join(OUT_DIR, "fig1_architecture.pdf")
    surf = cairo.PDFSurface(path, W, H)
    ctx = cairo.Context(surf)
    ctx.set_source_rgb(1, 1, 1)
    ctx.paint()

    # Panel (a) frame: Macro 3:1 Backbone
    draw_box(ctx, 6, 8, 208, 340, C_BG, C_LGRAY, lw=0.9, r=6)
    draw_text(ctx, 110, 22, "(a) Macro 3:1 Backbone (dddaddda)", size=9.2, bold=True, align="center")

    # Input token embedding
    draw_box(ctx, 26, 308, 168, 28, (0.94, 0.95, 0.96), C_GRAY, lw=0.9, r=4)
    draw_text(ctx, 110, 319, "Byte Token Embedding (No Abs. PE)", size=7.5, bold=True, align="center")
    draw_text(ctx, 110, 330, "x_0 in R^(B x T x 384)", size=6.8, italic=True, color=C_GRAY, align="center")

    # 8 layers bottom-to-top
    layers = [
        ("Layer 1: Sheaf-Delta (d)", "d"),
        ("Layer 2: Sheaf-Delta (d)", "d"),
        ("Layer 3: Sheaf-Delta (d)", "d"),
        ("Layer 4: ALiBi Attention (a)", "a"),
        ("Layer 5: Sheaf-Delta (d)", "d"),
        ("Layer 6: Sheaf-Delta (d)", "d"),
        ("Layer 7: Sheaf-Delta (d)", "d"),
        ("Layer 8: ALiBi Attention (a)", "a"),
    ]
    y_bot = 276
    step_y = 29
    for idx, (label, kind) in enumerate(layers):
        y = y_bot - idx * step_y
        if kind == "d":
            draw_box(ctx, 26, y, 134, 21, C_DEEPA_LT, C_DEEPA, lw=0.9, r=3.5)
            draw_text(ctx, 93, y + 10.5, label, size=7.3, bold=True, color=C_DEEPA, align="center", valign="center")
            draw_text(ctx, 165, y + 10.5, "O(1) state", size=6.4, color=C_DEEPA, valign="center")
        else:
            draw_box(ctx, 26, y, 134, 21, C_ALIBI_LT, C_ALIBI, lw=1.1, r=3.5)
            draw_text(ctx, 93, y + 10.5, label, size=7.3, bold=True, color=C_ALIBI, align="center", valign="center")
            draw_text(ctx, 165, y + 10.5, "KV cache", size=6.4, bold=True, color=C_ALIBI, valign="center")
        # arrow from below
        if idx == 0:
            draw_arrow(ctx, 93, 308, 93, y + 21, C_GRAY, lw=0.9, head=4.2)
        else:
            draw_arrow(ctx, 93, y + step_y, 93, y + 21, C_GRAY, lw=0.9, head=4.2)

    # Output head
    draw_box(ctx, 26, 36, 168, 26, (0.94, 0.95, 0.96), C_GRAY, lw=0.9, r=4)
    draw_text(ctx, 110, 46, "RMSNorm + Tied LM Head", size=7.5, bold=True, align="center")
    draw_text(ctx, 110, 56, "Logits in R^(B x T x 256)", size=6.8, italic=True, color=C_GRAY, align="center")
    draw_arrow(ctx, 93, y_bot - 7 * step_y, 93, 62, C_GRAY, lw=0.9, head=4.2)

    # Summary badge in Panel (a)
    draw_box(ctx, 16, 28, 188, 0, C_BG, C_BG, lw=0)

    # Panel (b) frame: Micro Sheaf-Delta (d) Block
    draw_box(ctx, 222, 8, 332, 232, C_BG, C_DEEPA, lw=1.0, r=6)
    draw_text(ctx, 388, 22, "(b) Micro View: Sheaf-Delta (d) Block (H_mix=6, d_h=64)", size=9.0, bold=True, color=C_DEEPA, align="center")

    # Input & RMSNorm
    draw_box(ctx, 326, 206, 124, 24, (0.94, 0.95, 0.96), C_GRAY, lw=0.9, r=3.5)
    draw_text(ctx, 388, 215, "Input x -> Pre-RMSNorm", size=7.3, bold=True, align="center")
    draw_text(ctx, 388, 225, "[B, T, 384]", size=6.6, italic=True, color=C_GRAY, align="center")

    # Branch 1 (Left): Global Gated Delta-Rule Notebook
    draw_box(ctx, 232, 92, 152, 98, C_DEEPA_LT, C_DEEPA, lw=0.95, r=4.5)
    draw_text(ctx, 308, 105, "Global Gated Delta-Rule", size=7.7, bold=True, color=C_DEEPA, align="center")
    draw_text(ctx, 308, 116, "q, k, v in [B, 6, T, 64]", size=6.7, color=C_DARK, align="center")
    draw_text(ctx, 308, 127, "L2-norm(q), L2-norm(k)", size=6.7, color=C_DARK, align="center")
    draw_text(ctx, 308, 138, "Gates: a in (0.8, 1), b in (0, 2)", size=6.7, color=C_DARK, align="center")
    draw_box(ctx, 240, 144, 136, 24, (1, 1, 1), C_DEEPA, lw=0.8, r=3)
    draw_text(ctx, 308, 153.5, "S_t = a_t S_{t-1}(I - b_t k k^T) + b_t v k^T", size=6.4, bold=True, color=C_DEEPA, align="center")
    draw_text(ctx, 308, 163.5, "Triton WY Chunk Kernel -> o_delta", size=6.3, italic=True, color=C_GRAY, align="center")
    draw_box(ctx, 256, 172, 104, 14, (1, 1, 1), C_DEEPA, lw=0.8, r=2.5)
    draw_text(ctx, 308, 179, "Independent RMSNorm (onorm)", size=6.3, bold=True, color=C_DEEPA, align="center", valign="center")

    # Branch 2 (Right): Local Multi-Scale Dyadic Sheaf
    draw_box(ctx, 394, 92, 150, 98, C_GREEN_LT, C_GREEN, lw=0.95, r=4.5)
    draw_text(ctx, 469, 105, "Local Dyadic Sheaf Diffusion", size=7.7, bold=True, color=C_GREEN, align="center")
    draw_text(ctx, 469, 116, "Causal lags d in {1, 2, 4, 8, 16}", size=6.7, color=C_DARK, align="center")
    draw_text(ctx, 469, 127, "Zero-copy slice shift v_{t-d}", size=6.7, color=C_DARK, align="center")
    draw_box(ctx, 402, 135, 134, 28, (1, 1, 1), C_GREEN, lw=0.8, r=3)
    draw_text(ctx, 469, 145, "M=2 Thin Householder U_{t,d}", size=6.5, bold=True, color=C_GREEN, align="center")
    draw_text(ctx, 469, 156, "Isometric transport (||U||_2 = 1)", size=6.3, italic=True, color=C_GRAY, align="center")
    draw_text(ctx, 469, 175, "o_dyadic = Sum_d g_{t,d} U_{t,d} v_{t-d}", size=6.4, bold=True, color=C_GREEN, align="center")
    draw_text(ctx, 469, 185, "Edge gates g_{t,d} in (0, 1)", size=6.2, italic=True, color=C_GRAY, align="center")

    # Split arrows from RMSNorm to both branches
    draw_arrow(ctx, 355, 206, 308, 190, C_DEEPA, lw=1.0, head=4.5)
    draw_arrow(ctx, 421, 206, 469, 190, C_GREEN, lw=1.0, head=4.5)

    # Combine sum + SiLU output gate + Out Proj + Residual + SwiGLU
    draw_box(ctx, 292, 58, 192, 22, (1, 1, 1), C_DARK, lw=0.95, r=3.5)
    draw_text(ctx, 388, 69, "(onorm(o_delta)+o_dyadic) * SiLU(W_g u) -> W_O", size=6.4, bold=True, align="center", valign="center")
    draw_arrow(ctx, 308, 92, 355, 80, C_DEEPA, lw=1.0, head=4.5)
    draw_arrow(ctx, 469, 92, 421, 80, C_GREEN, lw=1.0, head=4.5)

    draw_box(ctx, 294, 30, 188, 20, C_PURPLE_LT, C_PURPLE, lw=0.9, r=3.5)
    draw_text(ctx, 388, 40, "+ Residual -> Pre-RMSNorm -> SwiGLU MLP + Res.", size=6.8, bold=True, color=C_PURPLE, align="center", valign="center")
    draw_arrow(ctx, 388, 58, 388, 50, C_GRAY, lw=1.0, head=4.2)

    # Panel (c): Micro ALiBi Attention (a) Block
    draw_box(ctx, 222, 248, 332, 100, C_BG, C_ALIBI, lw=1.0, r=6)
    draw_text(ctx, 388, 262, "(c) Micro View: ALiBi Attention (a) Block (H_attn=8, d_h=48)", size=9.0, bold=True, color=C_ALIBI, align="center")
    draw_box(ctx, 234, 272, 92, 34, C_ALIBI_LT, C_ALIBI, lw=0.9, r=3.5)
    draw_text(ctx, 280, 284, "Pre-RMSNorm ->", size=6.8, bold=True, color=C_ALIBI, align="center")
    draw_text(ctx, 280, 295, "Q, K, V [B, 8, T, 48]", size=6.5, align="center")

    draw_box(ctx, 340, 272, 114, 34, (1, 1, 1), C_ALIBI, lw=0.9, r=3.5)
    draw_text(ctx, 397, 284, "Flash-Folded ALiBi SDPA", size=6.8, bold=True, color=C_ALIBI, align="center")
    draw_text(ctx, 397, 295, "Bias -m_h|i-j| folded in Q,K", size=6.3, italic=True, color=C_GRAY, align="center")

    draw_box(ctx, 468, 272, 76, 34, C_PURPLE_LT, C_PURPLE, lw=0.9, r=3.5)
    draw_text(ctx, 506, 284, "Out Proj + Res", size=6.7, bold=True, color=C_PURPLE, align="center")
    draw_text(ctx, 506, 295, "+ GELU MLP (3.65x)", size=6.2, color=C_PURPLE, align="center")

    draw_arrow(ctx, 326, 289, 340, 289, C_ALIBI, lw=1.0, head=4.2)
    draw_arrow(ctx, 454, 289, 468, 289, C_ALIBI, lw=1.0, head=4.2)

    draw_text(ctx, 388, 322, "Information Flow: 6 Sheaf-Delta layers maintain unrotated shift-invariant stalks & matrix state;", size=6.2, color=C_DARK, align="center")
    draw_text(ctx, 388, 334, "2 ALiBi layers perform exact long-range retrieval without rotating coordinates (4x smaller KV cache).", size=6.2, color=C_DARK, align="center")

    surf.finish()


# =====================================================================
# FIGURE 2: 16x Extrapolation (3-seed) + 2M-38M Parameter Scaling Law
# =====================================================================
def make_fig2():
    W, H = 560, 225
    path = os.path.join(OUT_DIR, "fig2_extrap_scaling.pdf")
    surf = cairo.PDFSurface(path, W, H)
    ctx = cairo.Context(surf)
    ctx.set_source_rgb(1, 1, 1)
    ctx.paint()

    # ---- Panel (a): 16x Extrapolation ----
    px, py, pw, ph = 44, 30, 218, 152
    draw_text(ctx, px + pw / 2, 16, "(a) Zero-Shot Extrapolation on enwik8 (3 Seeds)", size=8.8, bold=True, align="center")
    draw_box(ctx, px, py, pw, ph, (0.99, 0.992, 0.995), C_LGRAY, lw=0.8, r=0)

    # Y axis: 1.3 to 5.6
    y_min, y_max = 1.2, 5.6
    def ym(v): return py + ph - (v - y_min) / (y_max - y_min) * ph
    xs_labels = ["1k", "2k", "4k", "8k", "16k"]
    def xm(i): return px + 14 + i * (pw - 28) / 4.0

    # Gridlines
    for yv in [1.5, 2.5, 3.5, 4.5, 5.5]:
        yy = ym(yv)
        ctx.set_source_rgb(0.90, 0.91, 0.93)
        ctx.set_line_width(0.6)
        ctx.move_to(px, yy); ctx.line_to(px + pw, yy); ctx.stroke()
        draw_text(ctx, px - 4, yy, f"{yv:.1f}", size=7.0, color=C_GRAY, align="right", valign="center")

    for i, lab in enumerate(xs_labels):
        xx = xm(i)
        ctx.set_source_rgb(0.92, 0.93, 0.94)
        ctx.set_line_width(0.5)
        ctx.move_to(xx, py); ctx.line_to(xx, py + ph); ctx.stroke()
        draw_text(ctx, xx, py + ph + 11, lab, size=7.2, color=C_DARK, align="center")

    # Train length vertical dashed line at 1k
    ctx.set_source_rgb(0.45, 0.45, 0.5)
    ctx.set_dash([2.5, 2.5]); ctx.set_line_width(0.8)
    ctx.move_to(xm(0), py); ctx.line_to(xm(0), py + ph); ctx.stroke()
    ctx.set_dash([])
    draw_text(ctx, xm(0) + 3, py + 11, "Train ctx (1k)", size=6.3, italic=True, color=C_GRAY)

    series_a = [
        ("ALiBi-DEEPA (h6)", [1.507, 1.487, 1.477, 1.472, 1.469], C_DEEPA, False),
        ("Pure ALiBi", [1.528, 1.500, 1.486, 1.479, 1.475], C_ALIBI, True),
        ("RoPE GPT", [1.588, 2.352, 3.454, 4.519, 5.299], C_ROPE, False),
    ]
    for name, vals, col, dash in series_a:
        ctx.set_source_rgb(*col)
        ctx.set_line_width(1.7)
        if dash: ctx.set_dash([4, 2.5])
        for i, v in enumerate(vals):
            if i == 0: ctx.move_to(xm(i), ym(v))
            else: ctx.line_to(xm(i), ym(v))
        ctx.stroke()
        if dash: ctx.set_dash([])
        for i, v in enumerate(vals):
            ctx.arc(xm(i), ym(v), 2.6, 0, 2 * math.pi)
            ctx.fill()

    # Endpoint annotations
    draw_text(ctx, xm(4) - 4, ym(5.299) - 5, "5.30 (collapses)", size=6.6, bold=True, color=C_ROPE, align="right")

    # Zoomed inset inside Panel (a) showing ALiBi-DEEPA vs Pure ALiBi (1.45 - 1.54)
    ix, iy, iw, ih = px + 92, py + 58, 120, 62
    draw_box(ctx, ix, iy, iw, ih, (1, 1, 1), C_GRAY, lw=0.75, r=2.5)
    draw_text(ctx, ix + iw / 2, iy + 9, "Zoom: ALiBi-DEEPA vs Pure ALiBi", size=5.8, bold=True, color=C_DARK, align="center")
    iy_min, iy_max = 1.455, 1.540
    def iym(v): return iy + ih - 10 - (v - iy_min) / (iy_max - iy_min) * (ih - 22)
    def ixm(i): return ix + 18 + i * (iw - 26) / 4.0
    for yv in [1.46, 1.50, 1.54]:
        yy = iym(yv)
        ctx.set_source_rgb(0.92, 0.93, 0.95); ctx.set_line_width(0.45)
        ctx.move_to(ix + 16, yy); ctx.line_to(ix + iw - 4, yy); ctx.stroke()
        draw_text(ctx, ix + 14, yy, f"{yv:.2f}", size=5.2, color=C_GRAY, align="right", valign="center")
    for vals, col, dash in [([1.528, 1.500, 1.486, 1.479, 1.475], C_ALIBI, True), ([1.507, 1.487, 1.477, 1.472, 1.469], C_DEEPA, False)]:
        ctx.set_source_rgb(*col); ctx.set_line_width(1.25)
        if dash: ctx.set_dash([3, 2])
        for i, v in enumerate(vals):
            if i == 0: ctx.move_to(ixm(i), iym(v))
            else: ctx.line_to(ixm(i), iym(v))
        ctx.stroke()
        if dash: ctx.set_dash([])
        for i, v in enumerate(vals):
            ctx.arc(ixm(i), iym(v), 1.8, 0, 2 * math.pi); ctx.fill()
    draw_text(ctx, ixm(4), iym(1.469) + 6.5, "1.469", size=5.4, bold=True, color=C_DEEPA, align="right")
    draw_text(ctx, ixm(4), iym(1.475) - 4.5, "1.475", size=5.4, color=C_ALIBI, align="right")

    # Legend (a)
    lx, ly = px + 14, py + 24
    for idx, (name, _, col, dash) in enumerate(series_a):
        yy = ly + idx * 11
        ctx.set_source_rgb(*col); ctx.set_line_width(1.6)
        if dash: ctx.set_dash([3, 2])
        ctx.move_to(lx, yy); ctx.line_to(lx + 14, yy); ctx.stroke()
        if dash: ctx.set_dash([])
        draw_text(ctx, lx + 18, yy, name, size=6.8, color=C_DARK, valign="center")

    draw_text(ctx, px + pw / 2, py + ph + 24, "Evaluation Context Length (tokens)", size=7.5, bold=True, align="center")
    # Y label rotated
    ctx.save()
    ctx.translate(14, py + ph / 2)
    ctx.rotate(-math.pi / 2)
    draw_text(ctx, 0, 0, "Validation bpb (lower is better)", size=7.5, bold=True, align="center", valign="center")
    ctx.restore()

    # ---- Panel (b): Parameter Scaling Law (2M -> 38M) ----
    px2, py2, pw2, ph2 = 316, 30, 232, 152
    draw_text(ctx, px2 + pw2 / 2, 16, "(b) Parameter Scaling at 5 Tokens/Param (Full Val)", size=8.8, bold=True, align="center")
    draw_box(ctx, px2, py2, pw2, ph2, (0.99, 0.992, 0.995), C_LGRAY, lw=0.8, r=0)

    y2_min, y2_max = 1.15, 2.50
    def ym2(v): return py2 + ph2 - (v - y2_min) / (y2_max - y2_min) * ph2
    params_m = [1.85, 6.44, 14.44, 38.34]
    def xm2(p):
        lo, hi = math.log10(1.5), math.log10(45.0)
        return px2 + 12 + (math.log10(p) - lo) / (hi - lo) * (pw2 - 24)

    for yv in [1.2, 1.5, 1.8, 2.1, 2.4]:
        yy = ym2(yv)
        ctx.set_source_rgb(0.90, 0.91, 0.93); ctx.set_line_width(0.6)
        ctx.move_to(px2, yy); ctx.line_to(px2 + pw2, yy); ctx.stroke()
        draw_text(ctx, px2 - 4, yy, f"{yv:.1f}", size=7.0, color=C_GRAY, align="right", valign="center")

    for p, lab in zip(params_m, ["1.9M", "6.4M", "14.4M", "38.3M"]):
        xx = xm2(p)
        ctx.set_source_rgb(0.92, 0.93, 0.94); ctx.set_line_width(0.5)
        ctx.move_to(xx, py2); ctx.line_to(xx, py2 + ph2); ctx.stroke()
        draw_text(ctx, xx, py2 + ph2 + 11, lab, size=7.2, color=C_DARK, align="center")

    d_vals = [2.1117, 1.5983, 1.3938, 1.2349]
    a_vals = [2.4094, 1.6196, 1.4301, 1.2624]
    deltas = ["-0.298", "-0.021", "-0.036", "-0.028"]

    for vals, col, dash in [(a_vals, C_ALIBI, True), (d_vals, C_DEEPA, False)]:
        ctx.set_source_rgb(*col); ctx.set_line_width(1.7)
        if dash: ctx.set_dash([4, 2.5])
        for i, (p, v) in enumerate(zip(params_m, vals)):
            if i == 0: ctx.move_to(xm2(p), ym2(v))
            else: ctx.line_to(xm2(p), ym2(v))
        ctx.stroke()
        if dash: ctx.set_dash([])
        for p, v in zip(params_m, vals):
            ctx.arc(xm2(p), ym2(v), 2.7, 0, 2 * math.pi); ctx.fill()

    # Delta labels
    for i, (p, dv, av, dl) in enumerate(zip(params_m, d_vals, a_vals, deltas)):
        xx = xm2(p)
        yy = ym2(dv) + (11 if i > 0 else 12)
        draw_text(ctx, xx, yy, f"{dv:.3f}", size=6.3, bold=True, color=C_DEEPA, align="center")
        draw_text(ctx, xx, ym2(av) - 5, dl, size=6.2, bold=True, color=C_GREEN, align="center")

    # Legend (b)
    lx2, ly2 = px2 + pw2 - 102, py2 + 16
    for idx, (name, col, dash) in enumerate([("ALiBi-DEEPA (h6)", C_DEEPA, False), ("Pure ALiBi", C_ALIBI, True)]):
        yy = ly2 + idx * 11
        ctx.set_source_rgb(*col); ctx.set_line_width(1.6)
        if dash: ctx.set_dash([3, 2])
        ctx.move_to(lx2, yy); ctx.line_to(lx2 + 14, yy); ctx.stroke()
        if dash: ctx.set_dash([])
        draw_text(ctx, lx2 + 18, yy, name, size=6.8, color=C_DARK, valign="center")

    draw_text(ctx, px2 + pw2 / 2, py2 + ph2 + 24, "Model Parameters (log scale, matched 5 tok/param)", size=7.5, bold=True, align="center")
    surf.finish()


# =====================================================================
# FIGURE 3: Multi-Domain & Multi-Modal Quality Margin vs Pure ALiBi
# =====================================================================
def make_fig3():
    W, H = 560, 215
    path = os.path.join(OUT_DIR, "fig3_multidomain.pdf")
    surf = cairo.PDFSurface(path, W, H)
    ctx = cairo.Context(surf)
    ctx.set_source_rgb(1, 1, 1)
    ctx.paint()

    draw_text(ctx, W / 2, 15, "Multi-Domain Quality Advantage: ALiBi-DEEPA (h6) vs. Pure ALiBi (Positive = DEEPA Better)", size=8.8, bold=True, align="center")

    # Horizontal bar chart of (Pure ALiBi - ALiBi-DEEPA) in bpb across all 8 domains
    # Plus callout for RoPE-DEEPA vs RoPE GPT on CIFAR-10
    domains = [
        ("CIFAR-10 Pixels (RGB)", 4.6570 - 4.4382, "4.438 vs 4.657 (RoPE-DEEPA 3.947 < RoPE 3.966)"),
        ("Audio (16 kHz u-law)", 3.4436 - 3.3300, "3.330 vs 3.444 (beats RoPE GPT 3.412 by +0.082)"),
        ("DNA (hg38 chr20)", 1.8389 - 1.7420, "1.742 vs 1.839 (beats RoPE GPT 1.788 by +0.046)"),
        ("Hindi Wikipedia (UTF-8)", 0.9586 - 0.8745, "0.875 vs 0.959 (beats RoPE GPT 1.008 by +0.134)"),
        ("Python Source Code", 1.1883 - 1.1416, "1.142 vs 1.188 (beats RoPE GPT 1.213 by +0.072)"),
        ("WikiText-103 (5 tok/p)", 1.4404 - 1.4250, "1.425 vs 1.440 (2 seeds, 4,406 steps)"),
        ("enwik8 (18k, 20 tok/p)", 0.7863 - 0.7794, "0.779 vs 0.786 (3/3 seeds; +0.036 full-val 14M)"),
        ("Chinese Wikipedia (1.5k)", 2.0250 - 2.0581, "2.058 vs 2.025 (beats RoPE GPT 2.262 by +0.204)"),
    ]

    px, py, pw, ph = 128, 26, 218, 162
    draw_box(ctx, px, py, pw, ph, (0.99, 0.992, 0.995), C_LGRAY, lw=0.8, r=0)

    x_min, x_max = -0.06, 0.25
    def xm(v): return px + (v - x_min) / (x_max - x_min) * pw

    # Gridlines
    for xv in [-0.05, 0.0, 0.05, 0.10, 0.15, 0.20, 0.25]:
        xx = xm(xv)
        ctx.set_source_rgb(*(C_DARK if xv == 0 else (0.90, 0.91, 0.93)))
        ctx.set_line_width(1.0 if xv == 0 else 0.6)
        ctx.move_to(xx, py); ctx.line_to(xx, py + ph); ctx.stroke()
        draw_text(ctx, xx, py + ph + 10, f"{xv:+.2f}" if xv != 0 else "0.00", size=6.8, color=C_GRAY, align="center")

    row_h = ph / len(domains)
    for i, (lab, diff, note) in enumerate(domains):
        yc = py + (i + 0.5) * row_h
        bh = row_h * 0.60
        draw_text(ctx, px - 6, yc, lab, size=7.1, bold=True, color=C_DARK, align="right", valign="center")
        x0 = xm(0.0)
        x1 = xm(diff)
        col = C_DEEPA if diff >= 0 else C_ALIBI
        bx = min(x0, x1)
        bw = abs(x1 - x0)
        draw_box(ctx, bx, yc - bh / 2, bw, bh, col, col, lw=0.5, r=1.5)
        if diff >= 0:
            draw_text(ctx, x1 + 3, yc, f"{diff:+.3f}", size=6.6, bold=True, color=C_DEEPA, valign="center")
        else:
            draw_text(ctx, x1 - 3, yc, f"{diff:+.3f}", size=6.6, bold=True, color=C_ALIBI, align="right", valign="center")
        # Right annotation
        draw_text(ctx, px + pw + 8, yc, note, size=6.2, color=C_GRAY, valign="center")

    draw_text(ctx, px + pw / 2, py + ph + 22, "bpb Reduction vs. Pure ALiBi (bits per byte / base / dim)", size=7.3, bold=True, align="center")
    surf.finish()


# =====================================================================
# FIGURE 4: Systems Efficiency — Training Crossover & 8 GB Serving
# =====================================================================
def make_fig4():
    W, H = 560, 222
    path = os.path.join(OUT_DIR, "fig4_systems_overhead.pdf")
    surf = cairo.PDFSurface(path, W, H)
    ctx = cairo.Context(surf)
    ctx.set_source_rgb(1, 1, 1)
    ctx.paint()

    # ---- Panel (a): Training Step Latency (1k -> 16k) ----
    px, py, pw, ph = 46, 30, 218, 150
    draw_text(ctx, px + pw / 2, 16, "(a) Training Step Latency (14.4M, B=1, Compiled)", size=8.6, bold=True, align="center")
    draw_box(ctx, px, py, pw, ph, (0.99, 0.992, 0.995), C_LGRAY, lw=0.8, r=0)

    y_min, y_max = 0, 480
    def ym(v): return py + ph - (v - y_min) / (y_max - y_min) * ph
    ls = ["1k", "2k", "4k", "8k", "16k"]
    def xm(i): return px + 14 + i * (pw - 28) / 4.0

    for yv in [0, 100, 200, 300, 400]:
        yy = ym(yv)
        ctx.set_source_rgb(0.90, 0.91, 0.93); ctx.set_line_width(0.6)
        ctx.move_to(px, yy); ctx.line_to(px + pw, yy); ctx.stroke()
        draw_text(ctx, px - 4, yy, f"{yv}", size=7.0, color=C_GRAY, align="right", valign="center")

    for i, lab in enumerate(ls):
        xx = xm(i)
        ctx.set_source_rgb(0.92, 0.93, 0.94); ctx.set_line_width(0.5)
        ctx.move_to(xx, py); ctx.line_to(xx, py + ph); ctx.stroke()
        draw_text(ctx, xx, py + ph + 11, lab, size=7.2, color=C_DARK, align="center")

    deepa_ms = [22.8, 31.8, 54.0, 115.2, 281.3]
    alibi_ms = [17.0, 28.1, 57.4, 137.9, 410.5]
    rope_ms = [18.3, 31.1, 62.5, 150.9, 440.4]

    for vals, col, dash in [(rope_ms, C_ROPE, True), (alibi_ms, C_ALIBI, True), (deepa_ms, C_DEEPA, False)]:
        ctx.set_source_rgb(*col); ctx.set_line_width(1.7)
        if dash: ctx.set_dash([4, 2.5])
        for i, v in enumerate(vals):
            if i == 0: ctx.move_to(xm(i), ym(v))
            else: ctx.line_to(xm(i), ym(v))
        ctx.stroke()
        if dash: ctx.set_dash([])
        for i, v in enumerate(vals):
            ctx.arc(xm(i), ym(v), 2.5, 0, 2 * math.pi); ctx.fill()

    draw_text(ctx, xm(4) - 4, ym(440.4) - 6, "440.4 ms (RoPE)", size=6.3, bold=True, color=C_ROPE, align="right")
    draw_text(ctx, xm(4) - 4, ym(410.5) + 9, "410.5 ms (ALiBi)", size=6.3, bold=True, color=C_ALIBI, align="right")
    draw_text(ctx, xm(4) - 2, ym(281.3) + 20, "281.3 ms (1.46-1.57x)", size=6.3, bold=True, color=C_DEEPA, align="right")

    lx, ly = px + 10, py + 14
    for idx, (name, col, dash) in enumerate([("ALiBi-DEEPA (Flash-ALiBi)", C_DEEPA, False), ("Pure ALiBi (Flash-ALiBi)", C_ALIBI, True), ("RoPE GPT (FlashAttn)", C_ROPE, True)]):
        yy = ly + idx * 10.5
        ctx.set_source_rgb(*col); ctx.set_line_width(1.6)
        if dash: ctx.set_dash([3, 2])
        ctx.move_to(lx, yy); ctx.line_to(lx + 14, yy); ctx.stroke()
        if dash: ctx.set_dash([])
        draw_text(ctx, lx + 18, yy, name, size=6.4, color=C_DARK, valign="center")

    draw_text(ctx, px + pw / 2, py + ph + 24, "Sequence Length (tokens)", size=7.5, bold=True, align="center")
    ctx.save()
    ctx.translate(14, py + ph / 2)
    ctx.rotate(-math.pi / 2)
    draw_text(ctx, 0, 0, "Step Time (ms, lower is better)", size=7.5, bold=True, align="center", valign="center")
    ctx.restore()

    # ---- Panel (b): 8 GB Serving Throughput (Grouped Bars) ----
    px2, py2, pw2, ph2 = 322, 30, 226, 150
    draw_text(ctx, px2 + pw2 / 2, 16, "(b) Max Serving Throughput in 8 GB VRAM", size=8.6, bold=True, align="center")
    draw_box(ctx, px2, py2, pw2, ph2, (0.99, 0.992, 0.995), C_LGRAY, lw=0.8, r=0)

    y2_max = 9500
    def ym2(v): return py2 + ph2 - v / y2_max * ph2
    for yv in [0, 2000, 4000, 6000, 8000]:
        yy = ym2(yv)
        ctx.set_source_rgb(0.90, 0.91, 0.93); ctx.set_line_width(0.6)
        ctx.move_to(px2, yy); ctx.line_to(px2 + pw2, yy); ctx.stroke()
        draw_text(ctx, px2 - 4, yy, f"{yv // 1000}k" if yv else "0", size=7.0, color=C_GRAY, align="right", valign="center")

    groups = [
        ("4k ctx", [8537, 2781, 2283], "3.7x vs RoPE"),
        ("16k ctx", [2653, 700, 578], "4.6x vs RoPE"),
        ("64k ctx", [688, 163, 144], "4.8x vs RoPE"),
    ]
    cols = [C_DEEPA, C_ALIBI, C_ROPE]
    gw = pw2 / len(groups)
    bw = 15
    for gi, (glab, vals, mult) in enumerate(groups):
        gx = px2 + (gi + 0.5) * gw
        draw_text(ctx, gx, py2 + ph2 + 11, glab, size=7.2, bold=True, color=C_DARK, align="center")
        for bi, (v, col) in enumerate(zip(vals, cols)):
            bx = gx + (bi - 1.5) * (bw + 2) + 1
            bh = v / y2_max * ph2
            draw_box(ctx, bx, py2 + ph2 - bh, bw, bh, col, col, lw=0.5, r=1.5)
        draw_text(ctx, gx - bw, py2 + ph2 - vals[0] / y2_max * ph2 - 5, mult, size=6.2, bold=True, color=C_DEEPA, align="center")

    # Legend (b)
    lx2, ly2 = px2 + pw2 - 96, py2 + 14
    for idx, (name, col) in enumerate([("ALiBi-DEEPA (4x seqs)", C_DEEPA), ("Pure ALiBi", C_ALIBI), ("RoPE GPT", C_ROPE)]):
        yy = ly2 + idx * 11
        draw_box(ctx, lx2, yy - 3.5, 9, 7, col, col, lw=0.5, r=1)
        draw_text(ctx, lx2 + 12, yy, name, size=6.6, color=C_DARK, valign="center")

    draw_text(ctx, px2 + pw2 / 2, py2 + ph2 + 24, "Context Length (4x concurrent users in 8 GB)", size=7.5, bold=True, align="center")
    ctx.save()
    ctx.translate(290, py2 + ph2 / 2)
    ctx.rotate(-math.pi / 2)
    draw_text(ctx, 0, 0, "Aggregate Decode tok/s", size=7.5, bold=True, align="center", valign="center")
    ctx.restore()

    surf.finish()


if __name__ == "__main__":
    make_fig1()
    make_fig2()
    make_fig3()
    make_fig4()
    print("Generated 4 vector PDF figures in", OUT_DIR)
