from pathlib import Path
"""
Figure 2: G8 Convergence Acceleration Curve (synthetic)
B3 vs B4/B5: distortion vs epoch.
HONEST: shows both mid-run lead AND final-epoch catch-up.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import csv, os

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
    'lines.markersize': 5,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 150,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})

# Okabe-Ito
BLUE   = '#0072B2'
ORANGE = '#E69F00'
TEAL   = '#009E73'
VERMIL = '#D55E00'

# Load data
data_path = str(Path(__file__).resolve().parents[1] / 'analysis/tables' / 'table6_convergence_G8.csv')
epochs, B3, B4, B5, B4pct, B5pct = [], [], [], [], [], []
with open(data_path) as f:
    reader = csv.DictReader(f)
    for row in reader:
        epochs.append(int(row['epoch']))
        B3.append(float(row['B3_dist']))
        B4.append(float(row['B4_dist']))
        B5.append(float(row['B5_dist']))
        B4pct.append(float(row['B4_vs_B3_pct']))
        B5pct.append(float(row['B5_vs_B3_pct']))

epochs = np.array(epochs)
B3 = np.array(B3)
B4 = np.array(B4)
B5 = np.array(B5)
B4pct = np.array(B4pct)
B5pct = np.array(B5pct)

# ─── Two-panel figure ──
fig, axes = plt.subplots(1, 2, figsize=(11, 4))

# ── Panel A: Distortion curves ──
ax = axes[0]
ax.plot(epochs, B3, color=BLUE,   label='B3 (fixed $K$=1)',       linewidth=2.2, zorder=3)
ax.plot(epochs, B4, color=ORANGE, label='B4 (learnable $K$)',      linewidth=2.2, linestyle='-',  zorder=4)
ax.plot(epochs, B5, color=TEAL,   label='B5 (cross-$K$ momentum)', linewidth=2.2, linestyle='--', zorder=4)

# ── Annotate peak gap at ep800 ──
ep800_idx = list(epochs).index(800)
b3_800 = B3[ep800_idx]
b5_800 = B5[ep800_idx]
ax.annotate('Peak gap\n−2.19% (ep800)',
            xy=(800, (b3_800 + b5_800)/2),
            xytext=(550, 0.60),
            arrowprops=dict(arrowstyle='->', color=VERMIL, lw=1.5,
                            connectionstyle='arc3,rad=0.3'),
            fontsize=8, color=VERMIL, va='center', ha='center',
            bbox=dict(boxstyle='round,pad=0.25', facecolor='#FFF3F3', alpha=0.9))

# ── Annotate final value catch-up at ep1999 ──
ep1999_idx = list(epochs).index(1999)
b3_1999 = B3[ep1999_idx]
b5_1999 = B5[ep1999_idx]
ax.annotate('Final gap\n−0.38% (ep1999)',
            xy=(1999, (b3_1999 + b5_1999)/2),
            xytext=(1600, 0.48),
            arrowprops=dict(arrowstyle='->', color='#555555', lw=1.2,
                            connectionstyle='arc3,rad=-0.3'),
            fontsize=8, color='#555555', va='center', ha='center',
            bbox=dict(boxstyle='round,pad=0.25', facecolor='#F5F5F5', alpha=0.9))

ax.set_xlabel('Epoch')
ax.set_ylabel('Distortion (↓ better)')
ax.set_title('(A) Distortion vs. Epoch\n(Synthetic heterogeneous tree, br=[2,8,32,128])', fontsize=9.5)
ax.legend(loc='upper right', framealpha=0.8)
ax.set_xlim(0, 2100)
ax.set_ylim(0.20, 1.05)

# Shade the "convergence speed window" where B4/B5 lead
ax.axvspan(0, 1500, alpha=0.04, color=ORANGE, label='Speed advantage window')
ax.axvline(800, color=VERMIL, linestyle=':', alpha=0.5, linewidth=1)
ax.axvline(1999, color='#555555', linestyle=':', alpha=0.5, linewidth=1)

# ── Panel B: % gap over epochs ──
ax2 = axes[1]
ax2.axhline(0, color='black', linewidth=1.0, linestyle='-', alpha=0.4)
ax2.plot(epochs, B4pct, color=ORANGE, label='B4 vs B3', linewidth=2.0, zorder=3)
ax2.plot(epochs, B5pct, color=TEAL,   label='B5 vs B3', linewidth=2.0, linestyle='--', zorder=3)

# Mark ep800 peak
ax2.scatter([800], [B4pct[ep800_idx]], color=ORANGE, s=60, zorder=5)
ax2.scatter([800], [B5pct[ep800_idx]], color=TEAL,   s=60, zorder=5)
ax2.annotate('Peak: B5 −2.19%\n(ep800)', xy=(800, B5pct[ep800_idx]),
             xytext=(900, -2.5),
             arrowprops=dict(arrowstyle='->', color=TEAL, lw=1.2),
             fontsize=8, color=TEAL)

# Mark final value
ax2.scatter([1999], [B4pct[ep1999_idx]], color=ORANGE, s=60, zorder=5, marker='D')
ax2.scatter([1999], [B5pct[ep1999_idx]], color=TEAL,   s=60, zorder=5, marker='D')
ax2.annotate('Final: B5 −0.38%\n(ep1999)', xy=(1999, B5pct[ep1999_idx]),
             xytext=(1700, -0.8),
             arrowprops=dict(arrowstyle='->', color='#555555', lw=1.2),
             fontsize=8, color='#555555')

ax2.set_xlabel('Epoch')
ax2.set_ylabel('Δ Distortion vs. B3 (%)')
ax2.set_title('(B) Relative Gap: B4/B5 vs. B3\n(Negative = better than B3)', fontsize=9.5)
ax2.legend(loc='lower right', framealpha=0.8)
ax2.set_xlim(0, 2100)
ax2.axvline(800, color=VERMIL, linestyle=':', alpha=0.5, linewidth=1, label='ep800 peak')
ax2.axvline(1999, color='#555555', linestyle=':', alpha=0.5, linewidth=1, label='ep1999 final')

# Add interpretation note
fig.text(0.5, -0.02,
    'G8 — Convergence speed: B4/B5 reach −2.19% distortion advantage at ep800 (peak), '
    'but B3 catches up to −0.38% at ep1999 (final).\n'
    'Interpretation: learnable curvature accelerates convergence but does not yield a persistent final-value improvement in the synthetic setting.',
    ha='center', fontsize=8, style='italic', color='#444444')

plt.tight_layout(rect=[0, 0.05, 1, 1])
out = str(Path(__file__).resolve().parent / 'fig2_convergence_G8.pdf')
plt.savefig(out, format='pdf', dpi=200)
plt.savefig(out.replace('.pdf', '.png'), dpi=150)
print(f"Saved: {out}")
plt.close()
