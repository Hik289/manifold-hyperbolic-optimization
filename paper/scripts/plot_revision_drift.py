"""Plot the entire registered smooth-drift series, at common initialization."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path(__file__).resolve().parents[2];runs=root/'runs'
plt.rcParams.update({'font.family':'DejaVu Serif','font.size':9,'axes.labelsize':9,
    'axes.titlesize':10,'legend.fontsize':8,'pdf.fonttype':42,'ps.fonttype':42,
    'axes.spines.top':False,'axes.spines.right':False})
fig,axs=plt.subplots(2,3,figsize=(8.4,4.8),layout='constrained',sharex=True)
records=[]
colors={'B3':'#bd7541','B5':'#176c9d','A5':'#8b64a3'}
for col,scale in enumerate([2,5,8]):
    results={};trajectories={}
    for method in ['B3','B5','A5']:
        folder=runs/f'mechanism_smooth{scale}_independent_lorentz_3_{method}'
        result=json.loads((folder/'result.json').read_text())
        history=[json.loads(line) for line in (folder/'trajectory.jsonl').read_text().splitlines()]
        assert result['epochs_completed']==2000 and [r['epoch'] for r in history]==list(range(2000))
        results[method]=result['final'];trajectories[method]=history
        axs[0,col].plot(np.arange(1,2001),[r['val_distortion'] for r in history],
            color=colors[method],lw=1.45 if method!='A5' else 1.,
            ls='--' if method=='A5' else '-',label=method)
    axs[0,col].set_title(f'{scale}x smooth drift; initial K = 1')
    if col==0:axs[0,col].set_ylabel('Validation distortion')
    axs[0,col].grid(alpha=.17)
    k=np.array([r['K'] for r in trajectories['B5']])
    for g,(branch,color) in enumerate(zip([2,4,8,16],['#176c9d','#ba7939','#45917a','#8663a5'])):
        axs[1,col].plot(np.arange(1,2001),k[:,g],color=color,lw=1.2,label=f'Group {g+1}')
    axs[1,col].set_yscale('log');axs[1,col].set_xlabel('Update');axs[1,col].grid(alpha=.17)
    if col==0:axs[1,col].set_ylabel('Learned K (B5)')
    axs[1,col].set_xlim(0,2000)
    records.append(dict(scale=scale,final_test={m:r['test_distortion'] for m,r in results.items()},
        validation_auc={m:r['validation_auc'] for m,r in results.items()},
        relative_final_improvement_vs_B3=1-results['B5']['test_distortion']/results['B3']['test_distortion'],
        relative_final_improvement_vs_A5=1-results['B5']['test_distortion']/results['A5']['test_distortion']))
axs[0,0].legend(loc='upper right',frameon=False,ncol=1)
axs[1,2].legend(loc='lower left',frameon=False,ncol=2)
dest=root/'paper/figures/fig8_revision_long_drift'
fig.savefig(dest.with_suffix('.pdf'),bbox_inches='tight');fig.savefig(dest.with_suffix('.png'),dpi=240,bbox_inches='tight');plt.close(fig)
(root/'analysis/revision/long_drift_evidence.json').write_text(json.dumps(dict(
    scope='Common initial K=1, complete 2000-update runs. The canonical-radius cap changes physical-distance capacity when K changes: these gains combine scale adaptation and feasible-set capacity. Not a best-fixed-K or isolated transport superiority claim.',
    results=records),indent=2)+'\n')
print(json.dumps(records,indent=2))
