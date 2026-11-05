from pathlib import Path
"""
Figure 4: het_br Differentiation–Benefit Relationship (G3→G1' mechanism chain)
Branchings [2,8,32,128]: K differentiates per group + distortion improvement.
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
    'lines.markersize': 7,
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
TEAL   = '#009E73'
VERMIL = '#D55E00'

# ─── Data from table4_heterogeneity.csv ──
# het_br: [2, 8, 32, 128] subgraphs
branchings = [2, 8, 32, 128]
branch_labels = ['br=2', 'br=8', 'br=32', 'br=128']

# B3: fixed K=1 for all groups
B3_K = [1.0, 1.0, 1.0, 1.0]
B3_distortion = 0.2287
B3_loss = 0.11706

# B4: learned K per group (K_g0..K_g3 from table)
B4_K = [0.7681, 0.9389, 1.0109, 1.028]
B4_distortion = 0.2239
B4_loss = 0.11569

# B5: same structure
B5_K = [0.7719, 0.9452, 1.0164, 1.0328]
B5_distortion = 0.2236
B5_loss = 0.11559

# Distortion improvement
B4_dist_imp = (B3_distortion - B4_distortion) / B3_distortion * 100
B5_dist_imp = (B3_distortion - B5_distortion) / B3_distortion * 100

# K_std values
B4_Kstd = 0.1028
B5_Kstd = 0.1033

# ─── Create figure: 2 panels ──
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

# ── Panel A: K per group for B3, B4, B5 ──
ax = axes[0]
x = np.arange(len(branchings))
width = 0.25

bars_B3 = ax.bar(x - width, B3_K, width, color=BLUE,   label='B3: fixed K=1',       alpha=0.85, edgecolor='white')
bars_B4 = ax.bar(x,         B4_K, width, color=ORANGE, label='B4: learnable K',      alpha=0.85, edgecolor='white')
bars_B5 = ax.bar(x + width, B5_K, width, color=TEAL,   label='B5: cross-K mom.',     alpha=0.85, edgecolor='white')

ax.axhline(1.0, color='black', linewidth=1.0, linestyle='--', alpha=0.5, zorder=1)
ax.set_xticks(x)
ax.set_xticklabels([f'Group {i+1}\n(br={b})' for i, b in enumerate(branchings)], fontsize=9)
ax.set_ylabel('Learned Curvature K')
ax.set_title('(A) Per-Group Curvature K\n(heterogeneous branching tree)', fontsize=9.5)
ax.legend(loc='lower right', fontsize=8, framealpha=0.85)
ax.set_ylim(0.70, 1.16)

# Annotate K values
for bar in bars_B4:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h + 0.003, f'{h:.3f}',
            ha='center', va='bottom', fontsize=7.5, color=ORANGE)

# Add K_std annotation — place in lower-left to avoid annotation conflict
ax.text(0.02, 0.08,
        f'K_std: B4={B4_Kstd:.3f}, B5={B5_Kstd:.3f}, B3=0',
        transform=ax.transAxes, fontsize=7.5, va='bottom', ha='left',
        bbox=dict(boxstyle='round,pad=0.25', facecolor='white', alpha=0.85,
                  edgecolor='#CCCCCC'))

# Add monotone trend annotation — placed in upper-left (legend moved to lower-right)
ax.text(0.03, 0.97,
        'B4 K ↑ monotone with branching rate',
        transform=ax.transAxes, fontsize=8, va='top', ha='left',
        color=ORANGE, style='italic',
        bbox=dict(boxstyle='round,pad=0.2', facecolor='#FFF3E0', alpha=0.9))

# ── Panel B: Distortion comparison + K_std scatter ──
ax2 = axes[1]

methods = ['B3\n(fixed K)', 'B4\n(learnable K)', 'B5\n(cross-K mom.)']
dist_vals = [B3_distortion, B4_distortion, B5_distortion]
kstd_vals = [0.0, B4_Kstd, B5_Kstd]

colors_bars = [BLUE, ORANGE, TEAL]
bars = ax2.bar(methods, dist_vals, color=colors_bars, alpha=0.85, edgecolor='white', zorder=3)

# Annotate improvement
for i, (bar, d) in enumerate(zip(bars, dist_vals)):
    ax2.text(bar.get_x() + bar.get_width()/2, d + 0.0005, f'{d:.4f}',
             ha='center', va='bottom', fontsize=9, fontweight='bold', color='#333333')
    if i > 0:
        imp = (dist_vals[0] - d) / dist_vals[0] * 100
        ax2.text(bar.get_x() + bar.get_width()/2, d + 0.003,
                 f'−{imp:.1f}%', ha='center', va='bottom', fontsize=8.5, color=colors_bars[i])

ax2.set_ylabel('Final Distortion (↓ better)')
ax2.set_title('(B) Final Distortion by Method\n(het_br: branchings [2,8,32,128])', fontsize=9.5)
ax2.set_ylim(0.215, 0.237)

# Add secondary axis for K_std
ax2b = ax2.twinx()
ax2b.scatter(methods, kstd_vals, color=VERMIL, s=100, zorder=5, marker='D',
             label='K_std (right)')
ax2b.set_ylabel('Curvature Spread K$_{\\rm std}$ (right)', color=VERMIL, fontsize=9)
ax2b.tick_params(axis='y', labelcolor=VERMIL)
ax2b.set_ylim(-0.01, 0.16)
ax2b.spines['right'].set_visible(True)
ax2b.spines['right'].set_color(VERMIL)
# Connect K_std to distortion improvement
for i, (m, k, d) in enumerate(zip(methods, kstd_vals, dist_vals)):
    if k > 0:
        ax2b.annotate(f'K_std={k:.3f}', xy=(i, k),
                      xytext=(i + 0.05, k + 0.012), fontsize=7.5, color=VERMIL)

# Mechanism chain note
ax2.text(0.5, 0.06,
         'Higher K_std → Greater\ncurvature differentiation\n→ Lower distortion',
         transform=ax2.transAxes, fontsize=8, ha='center', va='bottom',
         style='italic',
         bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFF9C4', alpha=0.85))

# Caption
fig.text(0.5, -0.02,
    'G3→G1\' mechanism: Groups with higher branching rate learn higher K (Panel A, monotone trend). '
    'Greater K diversification (K_std B4=0.103) correlates with lower final distortion (Panel B: −2.10%). '
    'B5 achieves marginally better distortion (−2.23%) with similar K_std, confirming curvature differentiation drives the gain.',
    ha='center', fontsize=8, style='italic', color='#444444')

plt.tight_layout(rect=[0, 0.06, 1, 1])
out = str(Path(__file__).resolve().parent / 'fig4_het_br.pdf')
plt.savefig(out, format='pdf', dpi=200)
plt.savefig(out.replace('.pdf', '.png'), dpi=150)
print(f"Saved: {out}")
plt.close()
