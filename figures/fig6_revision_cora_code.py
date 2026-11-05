from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parents[1]
fig,axs=plt.subplots(1,2,figsize=(7,2.6),constrained_layout=True)
for method,style in [('E','-'),('B3','-'),('B4','--'),('B5','-'),('A5',':')]:
    path=root/'runs'/f'graph_cora_gcn_{method}'/'trajectory.jsonl'
    rows=[json.loads(l) for l in path.read_text().splitlines()];x=[r['epoch']+1 for r in rows]
    axs[0].plot(x,[r['train_loss'] for r in rows],style,label=method,lw=1.3)
    axs[1].plot(x,[r['val_accuracy']*100 for r in rows],style,label=method,lw=1.1,alpha=.8)
axs[0].set_yscale('log');axs[0].set_ylabel('Training loss');axs[1].set_ylabel('Validation accuracy (%)')
for ax in axs:ax.set_xlabel('Epoch');ax.grid(alpha=.18);ax.spines[['top','right']].set_visible(False)
axs[1].set_ylim(65,83);axs[0].legend(frameon=False,ncol=2,fontsize=8)
fig.savefig(root/'paper/figures/fig6_revision_cora.pdf');fig.savefig(root/'paper/figures/fig6_revision_cora.png',dpi=160)
