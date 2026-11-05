"""Complete synthetic-grid comparisons, with explicitly named baseline scope."""
import json
from collections import defaultdict
from pathlib import Path
from statistics import median

METHODS = ('E', 'B2', 'B3', 'B4', 'B5', 'A5')
LABELS = {'binary': 'Binary', '4ary': '4-ary', '8ary': '8-ary', 'variable': 'Variable'}
ORDER = ('binary', '4ary', '8ary', 'variable')


def write_synthetic_evidence(completed, paper):
    tab = Path(paper) / 'tables'
    groups = defaultdict(dict)
    for config, result in completed:
        if config['family'] == 'synthetic' and result['epochs_completed'] == config['epochs']:
            groups[(config['dataset'], config['nodes'], config['dim'])][config['method']] = (config, result)
    records = []
    for (tree, nodes, dim), methods in sorted(groups.items()):
        if not set(METHODS) <= methods.keys():
            continue
        scores = {m: methods[m][1]['final']['distortion_fixed_pairs'] for m in METHODS}
        records.append(dict(tree=tree, nodes=nodes, dim=dim, scores=scores,
                            run_ids={m: methods[m][0]['id'] for m in METHODS},
                            reduction_vs_B3_percent=100 * (1 - scores['B5'] / scores['B3']),
                            relative_A5_difference_percent=100 * abs(scores['B5'] / scores['A5'] - 1)))
    whole_grid = len(records) == 48 and all(sum(r['tree'] == t for r in records) == 12 for t in ORDER)
    evidence = dict(complete_blocks=len(records), registered_blocks=48, entire_grid_complete=whole_grid,
                    records=records, comparison_scope='Descriptive fixed-configuration B5 versus B3 ablation; all six methods retained. No statistical significance or overall-best claim.')
    (tab / 'revision_synthetic_evidence.json').write_text(json.dumps(evidence, indent=2) + '\n')
    if not whole_grid:
        for name in ['revision_synthetic_summary.tex', 'revision_synthetic_focus.tex', 'revision_synthetic_all.tex']:
            (tab / name).write_text('% Awaiting all 48 complete six-method blocks.\n')
        return

    reductions = [r['reduction_vs_B3_percent'] for r in records]
    wins_b3 = sum(r['scores']['B5'] < r['scores']['B3'] for r in records)
    wins_e = sum(r['scores']['B5'] < r['scores']['E'] for r in records)
    max_a5 = max(r['relative_A5_difference_percent'] for r in records)
    summary = (
        '\\paragraph{Complete product-factor grid.}\n'
        'The revision also completes all 288 runs in the product-factor tree grid: '
        'four tree constructions, three node counts, four total dimensions, and six methods, '
        'each trained for 2000 updates with seed 42. '
        f'B5 has numerically lower final fixed-pair distortion than fixed-curvature momentum B3 '
        f'in {wins_b3}/48 configurations, with relative reductions from '
        f'${min(reductions):.4f}\\%$ to ${max(reductions):.4f}\\%$ '
        '(Table~\\ref{tab:revision-synthetic-complete}). '
        'This is a small, consistent descriptive contrast within the manifold implementation. '
        f'B5 beats the Euclidean control in {wins_e}/48 configurations; '
        f'its relative distortion difference from A5 never exceeds ${max_a5:.4f}\\%$. '
        'Thus this grid supports a narrow combined curvature-and-normalization comparison with B3, '
        'not overall superiority or a separately established transport benefit. '
        'These are fixed-step, single-seed reconstruction measurements without held-out pair selection '
        'or an equal-budget hyperparameter search. All 48 configurations and all six methods appear '
        'in Appendix~\\ref{app:complete-product-grid}; the grid is distinct from the independent-tree drift experiments.\n'
    )
    (tab / 'revision_synthetic_summary.tex').write_text(summary)
    lines = [r'% WWW_REFERENCE_STYLE', r'\begin{table*}[t]', r'\centering',
             r'\caption{\textbf{Complete product-factor grid: the B5--B3 contrast.} Each tree construction includes all $n\in\{1000,5000,10000\}$ and $d\in\{2,5,10,20\}$. Reduction is $100(1-D_{\rm B5}/D_{\rm B3})$, in percent, from unrounded final fixed-pair distortion. Counts use strict numerical comparisons, not significance tests. Every configuration is retained. The final column shows the broader Euclidean comparison.}',
             r'\label{tab:revision-synthetic-complete}',
             r'\footnotesize\setlength{\tabcolsep}{4.2pt}\renewcommand{\arraystretch}{1.12}',
             r'\begin{tabular}{lrrrr}', r'\toprule',
             r'Tree & B5 $<$ B3 & Reduction range (\%) & Median reduction (\%) & B5 $<$ E \\', r'\midrule']
    for tree in ORDER:
        rows = [r for r in records if r['tree'] == tree]
        changes = [r['reduction_vs_B3_percent'] for r in rows]
        b3 = sum(r['scores']['B5'] < r['scores']['B3'] for r in rows)
        e = sum(r['scores']['B5'] < r['scores']['E'] for r in rows)
        lines.append(f"{LABELS[tree]} & \\cellcolor{{oursbg}}{b3}/12 & {min(changes):.4f}--{max(changes):.4f} & {median(changes):.4f} & {e}/12 " + r'\\')
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table*}']
    (tab / 'revision_synthetic_focus.tex').write_text('\n'.join(lines) + '\n')

    lines = [r'% WWW_REFERENCE_STYLE', r'\begingroup',
             r'\footnotesize\setlength{\tabcolsep}{4.2pt}\renewcommand{\arraystretch}{1.12}',
             r'\begin{longtable}{lrrrrrrrrr}',
             r'\caption{Complete product-factor reconstruction grid. Final fixed-pair distortion (lower is better) after 2000 updates, seed 42. Bold indicates the best value at the displayed precision; B5 is shaded. The last column reports the B5 reduction relative to B3 in percent, calculated from unrounded records. All baselines, including the stronger Euclidean control, are retained.}\label{tab:revision-synthetic-all}\\',
             r'\toprule', r'Tree & $n$ & $d$ & E & B2 & B3 & B4 & B5 & A5 & Reduction (\%) \\',
             r'\midrule', r'\endfirsthead', r'\toprule',
             r'Tree & $n$ & $d$ & E & B2 & B3 & B4 & B5 & A5 & Reduction (\%) \\',
             r'\midrule', r'\endhead', r'\bottomrule', r'\endfoot']
    for tree in ORDER:
        for row in (r for r in records if r['tree'] == tree):
            shown = {m: f"{row['scores'][m]:.5f}" for m in METHODS}
            best = min(float(s) for s in shown.values())
            cells = [LABELS[tree], str(row['nodes']), str(row['dim'])]
            for method in METHODS:
                val = shown[method]
                if float(val) == best:
                    val = r'\textbf{' + val + '}'
                if method == 'B5':
                    val = r'\cellcolor{oursbg}' + val
                cells.append(val)
            cells.append(f"{row['reduction_vs_B3_percent']:.4f}")
            lines.append(' & '.join(cells) + r' \\')
    lines += [r'\end{longtable}', r'\endgroup']
    (tab / 'revision_synthetic_all.tex').write_text('\n'.join(lines) + '\n')
