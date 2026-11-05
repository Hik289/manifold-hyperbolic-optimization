"""Reproduce every figure included by main.tex using bundled logged losses.

No GPU, network, training data, checkpoints, or external repository is required.
The already generated PDFs are used by LaTeX; this script is optional on Overleaf.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


COLORS = {'E':'#777777', 'B3':'#4477AA', 'B4':'#CC6677', 'B5':'#228833', 'A5':'#AA3377'}


def read(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def render(data_dir, out):
    provenance = json.loads((data_dir/'PROVENANCE.json').read_text())
    for name, digest in provenance['bundled_csv_sha256'].items():
        if hashlib.sha256((data_dir/name).read_bytes()).hexdigest() != digest:
            raise ValueError(f'Bundled source changed: {name}')
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Serif', 'font.size':8, 'axes.labelsize':8,
                         'legend.fontsize':7, 'pdf.fonttype':42,
                         'axes.spines.top':False, 'axes.spines.right':False})

    def save(fig, name):
        fig.savefig(out/f'{name}.pdf', bbox_inches='tight',
                    metadata={'CreationDate':None, 'ModDate':None})
        fig.savefig(out/f'{name}.png', dpi=220, bbox_inches='tight')
        plt.close(fig)

    historical = read(data_dir/'historical_kg_loss.csv')
    fig, ax = plt.subplots(figsize=(3.25, 2.1), layout='constrained')
    for m in ['B3', 'B4']:
        ax.plot([int(r['epoch']) for r in historical], [float(r[m+'_loss']) for r in historical],
                color=COLORS[m], marker='o', ms=2.5, lw=1.3, label=m)
    ax.set(xlabel='Epoch', ylabel='Training loss')
    ax.grid(alpha=.16); ax.legend(frameon=False)
    save(fig, 'fig_loss_historical_kg')

    cora = read(data_dir/'revision_cora_loss.csv')
    fig, ax = plt.subplots(figsize=(3.25, 2.15), layout='constrained')
    for m in ['E', 'B3', 'B4', 'B5', 'A5']:
        rows = [r for r in cora if r['method'] == m]
        assert [int(r['epoch']) for r in rows] == list(range(1000))
        ax.plot([int(r['epoch'])+1 for r in rows], [float(r['train_loss']) for r in rows],
                color=COLORS[m], lw=1.4 if m == 'B5' else 1,
                ls='--' if m == 'A5' else '-', label=m)
    ax.set(xlabel='Epoch', ylabel='Training loss', yscale='log')
    ax.grid(alpha=.16); ax.legend(frameon=False, ncol=3)
    save(fig, 'fig_loss_revision_cora')

    drift = read(data_dir/'smooth_drift_loss.csv')
    fig, axs = plt.subplots(1, 3, figsize=(6.75, 2.0), layout='constrained')
    for ax, scale in zip(axs, [2, 5, 8]):
        for m in ['B3', 'B5', 'A5']:
            rows = [r for r in drift if int(r['scale']) == scale and r['method'] == m]
            assert [int(r['epoch']) for r in rows] == list(range(2000))
            ax.plot([int(r['epoch'])+1 for r in rows], [float(r['train_loss']) for r in rows],
                    color=COLORS[m], lw=1.5 if m == 'B5' else 1.,
                    ls='--' if m == 'A5' else '-', label=m)
        ax.set(xlabel='Update', ylabel='Mean relative MSE', title=f'{scale}× drift')
        ax.grid(alpha=.16)
    axs[0].legend(frameon=False)
    save(fig, 'fig_loss_smooth_drift')


if __name__ == '__main__':
    paper = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=paper/'figure_data')
    parser.add_argument('--output-dir', type=Path, default=paper/'figures')
    args = parser.parse_args()
    render(args.data_dir, args.output_dir)
    print('Rebuilt all three manuscript figures (PDF and PNG).')
