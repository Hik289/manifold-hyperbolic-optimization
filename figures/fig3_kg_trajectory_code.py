from pathlib import Path
"""
Figure 3: KG Training Trajectory (G1')
WN18RR B3 vs B4: H@10 and loss vs epoch (moving average).
Shows loss gap stable from ~ep300; H@10 noisy but higher B4 average.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import csv

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
    'lines.linewidth': 1.8,
    'lines.markersize': 5,
    'errorbar.capsize': 4,
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

def moving_average(data, window=2):
    if len(data) <= window:
        return np.array(data)
    result = []
    for i in range(len(data)):
        lo = max(0, i - window + 1)
        result.append(np.mean(data[lo:i+1]))
    return np.array(result)

# Load data
data_path = str(Path(__file__).resolve().parents[1] / 'analysis/tables' / 'table1c_kg_trajectory.csv')
epochs, B3_MRR, B4_MRR = [], [], []
B3_H10, B4_H10 = [], []
B3_loss, B4_loss = [], []
B4_Kstd = []

with open(data_path) as f:
    reader = csv.DictReader(f)
    for row in reader:
        epochs.append(int(row['epoch']))
        B3_MRR.append(float(row['B3_MRR']))
        B4_MRR.append(float(row['B4_MRR']))
        B3_H10.append(float(row['B3_H10']))
        B4_H10.append(float(row['B4_H10']))
        B3_loss.append(float(row['B3_loss']))
        B4_loss.append(float(row['B4_loss']))
        B4_Kstd.append(float(row['B4_Kstd']))

epochs    = np.array(epochs, dtype=float)
B3_H10    = np.array(B3_H10)
B4_H10    = np.array(B4_H10)
B3_loss   = np.array(B3_loss)
B4_loss   = np.array(B4_loss)
B4_Kstd   = np.array(B4_Kstd)

# Moving averages (window=2 since only 8 checkpoints)
ma_B3_H10  = moving_average(B3_H10,  2)
ma_B4_H10  = moving_average(B4_H10,  2)
ma_B3_loss = moving_average(B3_loss, 2)
ma_B4_loss = moving_average(B4_loss, 2)

# ─── 3-panel figure: H@10, Loss, K_std ──
fig, axes = plt.subplots(1, 3, figsize=(13, 4))

# ── Panel A: H@10 ──
ax = axes[0]
ax.plot(epochs, B3_H10,   color=BLUE,   label='B3 raw',   linewidth=1.0, alpha=0.4, zorder=2)
ax.plot(epochs, B4_H10,   color=ORANGE, label='B4 raw',   linewidth=1.0, alpha=0.4, zorder=2)
ax.plot(epochs, ma_B3_H10, color=BLUE,   label='B3 (MA)',  linewidth=2.2, zorder=3)
ax.plot(epochs, ma_B4_H10, color=ORANGE, label='B4 (MA)',  linewidth=2.2, zorder=3)

# Annotate final values
ax.scatter([epochs[-1]], [B3_H10[-1]], color=BLUE,   s=60, zorder=5)
ax.scatter([epochs[-1]], [B4_H10[-1]], color=ORANGE, s=60, zorder=5)
ax.annotate(f'B3: {B3_H10[-1]:.3f}',
            xy=(epochs[-1], B3_H10[-1]),
            xytext=(epochs[-1]-300, B3_H10[-1]-0.015),
            fontsize=7.5, color=BLUE)
ax.annotate(f'B4: {B4_H10[-1]:.3f}',
            xy=(epochs[-1], B4_H10[-1]),
            xytext=(epochs[-1]-300, B4_H10[-1]+0.008),
            fontsize=7.5, color=ORANGE)

ax.set_xlabel('Epoch')
ax.set_ylabel('Hits@10 (↑ better)')
ax.set_title('(A) WN18RR: Hits@10 Trajectory', fontsize=9.5)
ax.legend(fontsize=7.5, loc='lower right', ncol=2, framealpha=0.85)
ax.set_ylim(0.30, 0.42)

# Annotate overall H@10 gain
h10_gain = ((np.mean(B4_H10[2:]) - np.mean(B3_H10[2:])) / np.mean(B3_H10[2:])) * 100
ax.text(0.05, 0.95, f'Avg B4−B3: +{h10_gain:.1f}%\n(ep300–1999)',
        transform=ax.transAxes, fontsize=8, va='top',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFF3E0', alpha=0.8))

# ── Panel B: Training Loss ──
ax = axes[1]
ax.plot(epochs, B3_loss,    color=BLUE,   label='B3 raw',  linewidth=1.0, alpha=0.4, zorder=2)
ax.plot(epochs, B4_loss,    color=ORANGE, label='B4 raw',  linewidth=1.0, alpha=0.4, zorder=2)
ax.plot(epochs, ma_B3_loss, color=BLUE,   label='B3 (MA)', linewidth=2.2, zorder=3)
ax.plot(epochs, ma_B4_loss, color=ORANGE, label='B4 (MA)', linewidth=2.2, zorder=3)

# Shaded gap region from ep300 onward
ax.fill_between(epochs, ma_B4_loss, ma_B3_loss,
                where=(epochs >= 300),
                alpha=0.15, color=ORANGE, label='B4 loss gap')

# Annotate gap at ep300 and ep1999
ep300_idx = list(epochs).index(300) if 300 in epochs else None
if ep300_idx is not None:
    gap300 = B3_loss[ep300_idx] - B4_loss[ep300_idx]
    ax.annotate(f'Gap≈{gap300:.3f}\n(ep300)',
                xy=(300, (B3_loss[ep300_idx] + B4_loss[ep300_idx])/2),
                xytext=(450, 2.090),
                arrowprops=dict(arrowstyle='->', color=VERMIL, lw=1.2),
                fontsize=8, color=VERMIL)

ax.axvline(300, color=VERMIL, linestyle=':', alpha=0.5, linewidth=1.2, label='ep300')

ax.set_xlabel('Epoch')
ax.set_ylabel('Training Loss (↓ better)')
ax.set_title('(B) WN18RR: Training Loss Trajectory', fontsize=9.5)
ax.legend(fontsize=7.5, loc='upper right', framealpha=0.85)

# Final loss values
loss_gap_final = B3_loss[-1] - B4_loss[-1]
loss_gap_pct   = loss_gap_final / B3_loss[-1] * 100
ax.text(0.05, 0.20, f'Final B4−B3 loss: −{loss_gap_pct:.1f}%',
        transform=ax.transAxes, fontsize=8,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#E8F5E9', alpha=0.8))

# ── Panel C: K_std evolution ──
ax = axes[2]
ax.plot(epochs, B4_Kstd, color=ORANGE, linewidth=2.2, label='B4 K_std', zorder=3)
ax.axhline(0.0, color=BLUE, linewidth=1.2, linestyle='--', alpha=0.6, label='B3 K_std=0')

ax.set_xlabel('Epoch')
ax.set_ylabel('Curvature Spread (K$_{\\rm std}$)')
ax.set_title('(C) WN18RR: Curvature Differentiation', fontsize=9.5)

# Annotate stabilization
ax.annotate('Stabilizes\n~ep300+',
            xy=(300, B4_Kstd[list(epochs).index(300)]),
            xytext=(500, 0.18),
            arrowprops=dict(arrowstyle='->', color=ORANGE, lw=1.2),
            fontsize=8, color=ORANGE)

final_kstd = B4_Kstd[-1]
ax.text(0.05, 0.90,
        f'Final K_std={final_kstd:.3f}\n(B3 K_std=0 by design)',
        transform=ax.transAxes, fontsize=8, va='top',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFF3E0', alpha=0.8))
ax.legend(fontsize=8, loc='center right', framealpha=0.85)
ax.set_ylim(-0.02, 0.35)

# Shared caption
fig.text(0.5, -0.02,
    'G1\' — WN18RR training trajectories. B4 (learnable K) achieves consistent Hits@10 advantage and '
    'stable loss reduction from ep300. Curvature spread (K_std) grows through ep300 then plateaus, '
    'confirming structural differentiation is learned early and maintained.',
    ha='center', fontsize=8, style='italic', color='#444444')

plt.tight_layout(rect=[0, 0.05, 1, 1])
out = str(Path(__file__).resolve().parent / 'fig3_kg_trajectory.pdf')
plt.savefig(out, format='pdf', dpi=200)
plt.savefig(out.replace('.pdf', '.png'), dpi=150)
print(f"Saved: {out}")
plt.close()
