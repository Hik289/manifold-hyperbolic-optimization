"""Rebuild the two scoped positive-contrast tables from bundled audited values."""
import argparse
import json
from pathlib import Path


def render(data, out):
    out.mkdir(parents=True, exist_ok=True)
    graph = [r for r in data['graph'] if r['delta_final_pp'] > 0]
    lines = [r'% WWW_REFERENCE_STYLE', r'\begin{table*}[t]', r'\centering',
             r'\caption{\textbf{Positive final-checkpoint graph contrasts against B3.} These five rows are an outcome-selected subset of all 12 completed dataset--encoder pairs; the complete matrix is in Table~\ref{tab:revision-breadth}. Accuracy and differences are in percent and percentage points, respectively. All runs finish 1000 epochs, seed 42. $\Delta_{\rm val}$ separately compares test accuracy at checkpoints selected by validation accuracy; it is not used to choose the displayed endpoint. A dash denotes an unavailable selected-checkpoint record. Bold marks the best displayed final accuracy, including rounded ties, not significance or a family-level win.}',
             r'\label{tab:positive-graph-contrasts}',
             r'\footnotesize\setlength{\tabcolsep}{6pt}\renewcommand{\arraystretch}{1.12}',
             r'\begin{adjustbox}{max width=\linewidth}', r'\begin{tabular}{lrrrrrr}',
             r'\toprule', r'Dataset / encoder & E & B3 & B5 & A5 & $\Delta_{\rm final}$ & $\Delta_{\rm val}$ \\', r'\midrule']
    for r in graph:
        values = [f"{100*r['final_accuracy'][m]:.3f}" for m in ['E', 'B3', 'B5', 'A5']]
        best = max(map(float, values))
        values = [r'\textbf{'+v+'}' if float(v) == best else v for v in values]
        values[2] = r'\cellcolor{oursbg}'+values[2]
        selected = '--' if r['delta_selected_pp'] is None else f"{r['delta_selected_pp']:+.3f}"
        lines.append(' & '.join([r['label']]+values+[f"{r['delta_final_pp']:+.3f}", selected])+r' \\')
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{adjustbox}', r'\end{table*}']
    (out/'positive_graph_contrasts.tex').write_text('\n'.join(lines)+'\n')

    lines = [r'% WWW_REFERENCE_STYLE', r'\begin{table*}[t]', r'\centering',
             r'\caption{\textbf{Additional positive contrasts with named controls.} Each row uses its original primary endpoint and full training length. The improvement is relative distortion reduction in the first row, absolute MRR difference in the second, and accuracy difference in percentage points in the last two. These units are not pooled. The A5 column preserves the transport comparison. Bold marks the best displayed score within a row; stronger controls outside these selected contrasts are reported in the full appendix. All are single-seed descriptive comparisons.}',
             r'\label{tab:positive-narrow-contrasts}',
             r'\footnotesize\setlength{\tabcolsep}{5pt}\renewcommand{\arraystretch}{1.12}',
             r'\begin{adjustbox}{max width=\linewidth}', r'\begin{tabular}{lllrrrr}',
             r'\toprule', r'Case & Control & Primary endpoint & Control score & B5 & A5 & Improvement \\', r'\midrule']
    for r in data['narrow']:
        values = [f"{r[k]*r['factor']:.{r['digits']}f}" for k in ['control', 'B5', 'A5']]
        best = (min if r['lower_is_better'] else max)(map(float, values))
        values = [r'\textbf{'+v+'}' if float(v) == best else v for v in values]
        values[1] = r'\cellcolor{oursbg}'+values[1]
        lines.append(' & '.join([r['label'], r['control_name'], r['metric_label']]+values+[r['improvement_tex']])+r' \\')
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{adjustbox}', r'\end{table*}']
    (out/'positive_narrow_contrasts.tex').write_text('\n'.join(lines)+'\n')


if __name__ == '__main__':
    paper = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=paper/'evidence/positive_contrasts.json')
    parser.add_argument('--output-dir', type=Path, default=paper/'tables')
    args = parser.parse_args()
    render(json.loads(args.data.read_text()), args.output_dir)
    print('Rebuilt both positive-contrast tables.')
