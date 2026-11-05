from pathlib import Path
"""
Figure 1: KG Curvature Differentiation by Relation (G3 core figure)
WN18RR (11 relations) + FB15k-237 (top-15 relations)
Colorblind-safe palette (Okabe-Ito), PDF vector output.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

plt.style.use('seaborn-v0_8-paper')
mpl.rcParams.update({
    'font.family': 'serif',
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.labelweight': 'bold',
    'axes.titlesize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8.5,
    'legend.fontsize': 8,
    'lines.linewidth': 1.5,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 150,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})

# Okabe-Ito colorblind-safe palette
BLUE   = '#0072B2'
ORANGE = '#E69F00'
TEAL   = '#009E73'
VERMIL = '#D55E00'
PURPLE = '#CC79A7'
CYAN   = '#56B4E9'
YELLOW = '#F0E442'
BLACK  = '#000000'

# ─── WN18RR data (from table1b_kg_K_per_relation.csv, final epoch K_per_rel) ──
wn_rels_raw = [
    ('_derivationally_related_form', 2.0),
    ('_member_meronym',              1.5823),
    ('_hypernym',                    1.4173),
    ('_has_part',                    1.305),
    ('_instance_hypernym',           1.222),
    ('_verb_group',                  1.0938),
    ('_also_see',                    1.0888),
    ('_synset_domain_topic_of',      1.0754),
    ('_member_of_domain_region',     1.017),
    ('_member_of_domain_usage',      1.0092),
    ('_similar_to',                  1.0084),
]
# already sorted descending; clean labels
def clean_wn(s):
    s = s.lstrip('_').replace('_', ' ')
    # shorten long ones
    abbr = {
        'derivationally related form': 'deriv. related',
        'member meronym': 'member meronym',
        'hypernym': 'hypernym',
        'has part': 'has part',
        'instance hypernym': 'instance hypernym',
        'verb group': 'verb group',
        'also see': 'also see',
        'synset domain topic of': 'synset domain topic',
        'member of domain region': 'domain region',
        'member of domain usage': 'domain usage',
        'similar to': 'similar to',
    }
    return abbr.get(s, s)

wn_labels = [clean_wn(r) for r, k in wn_rels_raw]
wn_K      = [k for r, k in wn_rels_raw]

# ─── FB15k-237 top-15 data (from fbfinal_B4 final K_per_rel) ──
fb_rels_raw = [
    ('/award/award_nominee/award_nominations./award/award_nomination/award_nominee', 1.8371),
    ('/film/film/release_date_s./film/film_regional_release_date/film_release_region', 1.3486),
    ('/award/award_winner/awards_won./award/award_honor/award_winner', 1.3449),
    ('/award/award_nominee/award_nominations./award/award_nomination/award', 1.2725),
    ('/music/genre/artists', 1.1832),
    ('/people/person/profession', 1.1817),
    ('/film/film/genre', 1.1754),
    ('/award/award_category/nominees./award/award_nomination/nominated_for', 1.1678),
    ('/film/actor/film./film/performance/film', 1.1677),
    ('/award/award_category/winners./award/award_honor/award_winner', 1.1435),
    ('/film/film/other_crew./film/film_crew_gig/film_crew_role', 1.1421),
    ('/award/award_nominee/award_nominations./award/award_nomination/nominated_for', 1.1338),
    ('/people/person/nationality', 1.1093),
    ('/award/award_winning_work/awards_won./award/award_honor/award', 1.1034),
    ('/sports/sports_position/players./sports/sports_team_roster/team', 1.0966),
]

def clean_fb(s):
    """Shorten FB relation path to a readable label."""
    import re
    # Extract key semantic part
    # Try to get the first component's leaf
    parts = s.strip('/').split('/')
    # Usually: domain/type/predicate
    if len(parts) >= 3:
        domain = parts[0]
        type_  = parts[1]
        pred   = parts[2]
        # strip compound predicates
        pred = pred.split('.')[0]
        short = f"{domain[:3]}/{type_[:4]}/{pred.replace('_', ' ')}"
    else:
        short = parts[-1].replace('_', ' ')
    return short

# Use semantic short labels manually for clarity
fb_labels_manual = [
    'award/nominee',
    'film/release region',
    'award/winner (A)',
    'award/nominee→award',
    'music/genre→artists',
    'person/profession',
    'film/genre',
    'award/nominees for',
    'actor/filmography',
    'award/winners (B)',
    'film/other crew',
    'award/nominated for',
    'person/nationality',
    'work/awards won',
    'sports/team roster',
]
fb_K = [k for r, k in fb_rels_raw]

# ─── Create figure: two side-by-side panels ──
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

# Panel A: WN18RR
ax = axes[0]
y_pos = np.arange(len(wn_labels))
bars = ax.barh(y_pos, wn_K, color=BLUE, edgecolor='white', linewidth=0.5, zorder=3)
ax.set_yticks(y_pos)
ax.set_yticklabels(wn_labels, fontsize=8.5)
ax.invert_yaxis()
ax.axvline(1.0, color='black', linewidth=1.0, linestyle='--', alpha=0.6, label='Fixed K=1')
ax.set_xlabel('Learned Curvature K', fontweight='bold')
ax.set_title('(A) WN18RR — high hierarchy\n(K$_{\\rm std}$=0.30, 11 relations)',
             fontsize=9.5, pad=6)
ax.set_xlim(0.95, 2.15)
ax.grid(True, axis='x', alpha=0.3, zorder=0)
ax.grid(False, axis='y')

# Annotate bars with K values
for bar, k in zip(bars, wn_K):
    ax.text(k + 0.02, bar.get_y() + bar.get_height()/2,
            f'{k:.3f}', va='center', ha='left', fontsize=7, color='#333333')

# Color top bars for emphasis
bars[0].set_color(VERMIL)  # highest K
bars[1].set_color(ORANGE)
bars[-1].set_color(TEAL)  # lowest K (near 1)
bars[-2].set_color(TEAL)

# Small legend inside
ax.axvline(1.0, color='black', linewidth=1.0, linestyle='--', alpha=0.6)
ax.text(1.02, len(wn_labels)-0.3, 'Fixed\nK=1', fontsize=7, color='#555555',
        va='top', ha='left')

# Panel B: FB15k-237
ax = axes[1]
y_pos2 = np.arange(len(fb_labels_manual))
bars2 = ax.barh(y_pos2, fb_K, color=PURPLE, edgecolor='white', linewidth=0.5, zorder=3)
ax.set_yticks(y_pos2)
ax.set_yticklabels(fb_labels_manual, fontsize=8.5)
ax.invert_yaxis()
ax.axvline(1.0, color='black', linewidth=1.0, linestyle='--', alpha=0.6)
ax.set_xlabel('Learned Curvature K', fontweight='bold')
ax.set_title('(B) FB15k-237 — low hierarchy\n(K$_{\\rm std}$=0.07, top-15 of 237 relations)',
             fontsize=9.5, pad=6)
ax.set_xlim(0.95, 2.0)
ax.grid(True, axis='x', alpha=0.3, zorder=0)
ax.grid(False, axis='y')

# Annotate
for bar, k in zip(bars2, fb_K):
    ax.text(k + 0.015, bar.get_y() + bar.get_height()/2,
            f'{k:.3f}', va='center', ha='left', fontsize=7, color='#333333')

# Emphasize top bars
bars2[0].set_color(VERMIL)
bars2[1].set_color(ORANGE)
bars2[2].set_color(ORANGE)

ax.text(1.02, len(fb_labels_manual)-0.3, 'Fixed\nK=1', fontsize=7, color='#555555',
        va='top', ha='left')

# Shared annotation: contrast note
fig.text(0.5, -0.01,
    'G3 — Curvature differentiates by relational structure. '
    'WN18RR (taxonomic hierarchy) shows strong spread (K∈[1.008, 2.0]); '
    'FB15k-237 (mixed relations) shows modest spread (K∈[1.00, 1.84]).\n'
    'Higher K = tighter hyperbolic geometry for that relation-type.',
    ha='center', fontsize=8, style='italic', color='#444444')

plt.tight_layout(rect=[0, 0.04, 1, 1])
out = str(Path(__file__).resolve().parent / 'fig1_curvature_bars.pdf')
plt.savefig(out, format='pdf', dpi=200)
plt.savefig(out.replace('.pdf', '.png'), dpi=150)
print(f"Saved: {out}")
plt.close()
