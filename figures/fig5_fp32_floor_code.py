from pathlib import Path
"""
Figure 5 (Optional): FP32 Representation Floor Phenomenon
Manifold residual ∝ cosh²(r)·2⁻²³ as Theorem 1 empirics + Limitations.
This is a theoretical/computed illustration (no raw experiment data needed).
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

plt.style.use('seaborn-v0_8-paper')
mpl.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.labelweight': 'bold',
    'axes.titlesize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'lines.linewidth': 2.0,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 150,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})

BLUE   = '#0072B2'
ORANGE = '#E69F00'
VERMIL = '#D55E00'
TEAL   = '#009E73'
GRAY   = '#BBBBBB'

# Theorem 1: manifold residual = ||Exp_x(v) - (x+v)||²  ∝ cosh²(||x||)·ε_machine
# For a hyperbolic point at radius r, the representation error scales as cosh²(r)·2⁻²³
# FP32 machine epsilon = 2⁻²³ ≈ 1.19e-7
eps_fp32 = 2**-23   # ≈ 1.19e-7
eps_fp64 = 2**-52   # ≈ 2.22e-16

r_vals = np.linspace(0, 10, 500)

def floor_residual(r, eps, curvature=1.0):
    """FP32/FP64 manifold residual lower bound (Theorem 1)."""
    return np.cosh(np.sqrt(curvature) * r)**2 * eps

fp32_residual = floor_residual(r_vals, eps_fp32, curvature=1.0)
fp64_residual = floor_residual(r_vals, eps_fp64, curvature=1.0)

# Different curvatures
fp32_k2 = floor_residual(r_vals, eps_fp32, curvature=2.0)   # K=2
fp32_k05 = floor_residual(r_vals, eps_fp32, curvature=0.5)  # K=0.5

# ─── 2-panel figure ──
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

# ── Panel A: FP32 vs FP64 ──
ax = axes[0]
ax.semilogy(r_vals, fp32_residual, color=VERMIL, label='FP32 ($\\varepsilon=2^{-23}$)', linewidth=2.2)
ax.semilogy(r_vals, fp64_residual, color=TEAL,   label='FP64 ($\\varepsilon=2^{-52}$)', linewidth=2.2, linestyle='--')

# Shade the "precision floor" zone for FP32
ax.axhline(eps_fp32, color=VERMIL, linewidth=1.0, linestyle=':', alpha=0.5, label='FP32 $\\varepsilon$')
ax.axhline(eps_fp64, color=TEAL,   linewidth=1.0, linestyle=':', alpha=0.5, label='FP64 $\\varepsilon$')

# Mark typical WN18RR embedding radii (~2-5)
ax.axvspan(2, 5, alpha=0.08, color=ORANGE, label='Typical embed.\nradius (WN18RR)')
ax.annotate('Typical embed.\nradius (r≈2–5)',
            xy=(3.5, fp32_residual[np.searchsorted(r_vals, 3.5)]),
            xytext=(0.3, 1e1),
            arrowprops=dict(arrowstyle='->', color=ORANGE, lw=1.2,
                            connectionstyle='arc3,rad=0.3'),
            fontsize=8, color=ORANGE,
            bbox=dict(boxstyle='round,pad=0.2', facecolor='#FFF9E6', alpha=0.9))

# Annotate FP32 floor value at r=5
r5_idx = np.searchsorted(r_vals, 5.0)
fp32_at5 = fp32_residual[r5_idx]
ax.scatter([5.0], [fp32_at5], color=VERMIL, s=60, zorder=5)
ax.annotate(f'FP32 floor @ r=5\n≈{fp32_at5:.1e}',
            xy=(5.0, fp32_at5),
            xytext=(6.8, fp32_at5 * 200),
            arrowprops=dict(arrowstyle='->', color=VERMIL, lw=1.2,
                            connectionstyle='arc3,rad=-0.3'),
            fontsize=8, color=VERMIL,
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.9))

ax.set_xlabel('Geodesic radius $r = d_K(\\mathbf{x}, o)$')
ax.set_ylabel('Manifold Residual Lower Bound')
ax.set_title('(A) FP32 vs FP64 Precision Floor\n$\\delta(r) = \\cosh^2(r) \\cdot \\varepsilon_{\\rm machine}$', fontsize=9.5)
ax.legend(fontsize=7.5, loc='upper left', framealpha=0.85)
ax.set_xlim(0, 10)
ax.set_ylim(1e-17, 1e5)

# ── Panel B: Effect of curvature K ──
ax2 = axes[1]
ax2.semilogy(r_vals, fp32_k05, color=TEAL,  label='$K=0.5$ (flat)', linewidth=2.2)
ax2.semilogy(r_vals, fp32_residual, color=BLUE, label='$K=1.0$ (standard)', linewidth=2.2)
ax2.semilogy(r_vals, fp32_k2, color=VERMIL, label='$K=2.0$ (high K)', linewidth=2.2, linestyle='--')

ax2.axhline(eps_fp32, color='black', linewidth=1.0, linestyle=':', alpha=0.4, label='FP32 $\\varepsilon$')

# Annotate divergence at r=5
for k, fp32_k, lbl, c in [
    (0.5, fp32_k05,       'K=0.5', TEAL),
    (1.0, fp32_residual,  'K=1.0', BLUE),
    (2.0, fp32_k2,        'K=2.0', VERMIL),
]:
    r5_idx = np.searchsorted(r_vals, 5.0)
    val = fp32_k[r5_idx]
    ax2.scatter([5.0], [val], color=c, s=50, zorder=5)

ax2.set_xlabel('Geodesic radius $r = d_K(\\mathbf{x}, o)$')
ax2.set_ylabel('Manifold Residual Lower Bound')
ax2.set_title('(B) Effect of Curvature on FP32 Floor\n(Higher $K$ → Steeper precision degradation)', fontsize=9.5)
ax2.legend(fontsize=8, loc='upper left', framealpha=0.85)
ax2.set_xlim(0, 8)
ax2.set_ylim(1e-8, 1e6)

# Implication note
ax2.text(0.98, 0.05,
    'Implication: learnable high-K\nregions incur worse FP32\nprecision at large radii.\nFP64 mitigates this.',
    transform=ax2.transAxes, fontsize=8, ha='right', va='bottom', style='italic',
    bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFF3E0', alpha=0.85))

fig.text(0.5, -0.02,
    'Theorem 1 empirics — FP32 representation floor grows as cosh²(r)·2⁻²³, '
    'becoming significant (>10⁻³) for r>5 in FP32. '
    'Higher learned curvature K amplifies this effect. '
    'Limitation: large-K groups in B4 push embeddings to higher effective radii, '
    'potentially incurring numerical noise that partly explains the H@10 variance in WN18RR.',
    ha='center', fontsize=8, style='italic', color='#444444')

plt.tight_layout(rect=[0, 0.06, 1, 1])
out = str(Path(__file__).resolve().parent / 'fig5_fp32_floor.pdf')
plt.savefig(out, format='pdf', dpi=200)
plt.savefig(out.replace('.pdf', '.png'), dpi=150)
print(f"Saved: {out}")
plt.close()
