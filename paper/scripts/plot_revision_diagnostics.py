"""Plot measured compute worker operator diagnostics, not benchmark-performance claims."""
from pathlib import Path
import json
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path(__file__).resolve().parents[2]
data=root/'analysis/theory_diagnostics'
s=pd.read_csv(data/'sensitivity.csv');t=pd.read_csv(data/'transport.csv')
plt.rcParams.update({'font.family':'DejaVu Serif','font.size':9,'axes.labelsize':10,
    'axes.titlesize':10,'legend.fontsize':8,'pdf.fonttype':42,'ps.fonttype':42,
    'axes.spines.top':False,'axes.spines.right':False})
fig,axs=plt.subplots(1,2,figsize=(8.2,3.1),layout='constrained')
ax=axs[0]
for norm,color,label in [(False,'#b45a32','Uncapped exponential'),(True,'#156c9e','Normalized step')]:
    d=s[(s.precision=='float64')&(s.K==1)&(s.normalized==norm)]
    ax.plot(d.scaled_momentum,d.exact_transverse_gain,color=color,lw=1.6,label=label)
    for i,k in enumerate([.01,.1,1.,10.,100.]):
        q=s[(s.precision=='float64')&(s.K==k)&(s.normalized==norm)].iloc[i::10]
        ax.scatter(q.scaled_momentum,q.measured_gain,s=17,marker=['o','s','^','v','D'][i],
            facecolors='none',edgecolors=color,linewidths=.65,zorder=4)
ax.axhline(1,color='#777777',lw=.7,ls=':')
ax.set_yscale('log');ax.set_xlabel(r'Scaled momentum $\sqrt{K}\,\|m\|$')
ax.set_ylabel('Transverse amplification');ax.set_title('(a) Measured sensitivity and exact differential',loc='left')
ax.text(.04,.58,r'$K\in\{0.01,0.1,1,10,100\}$'+'\nMarkers: measured; lines: exact',transform=ax.transAxes,fontsize=8)
ax.legend(loc='upper left',frameon=False);ax.grid(alpha=.17,which='both')
ax=axs[1]
for scale,color in [(2,'#d18c38'),(5,'#8360a0'),(8,'#b44d4d')]:
    q=t[(t.precision=='float32')&(t.scale_end==scale)]
    ax.plot(q.epoch+1,100*q.ablated_error,color=color,lw=1.5,label=f'Point only: {scale}x')
q=t[(t.precision=='float32')&(t.scale_end==8)]
ax.plot(q.epoch+1,100*q.full_error,color='#156c9e',lw=1.7,ls='--',label='Point + momentum')
ax.set_xlabel('Update');ax.set_ylabel('Canonical momentum error (%)')
ax.set_title('(b) Prescribed smooth curvature drift, FP32',loc='left')
ax.set_xlim(0,2000);ax.set_ylim(-.15,5.5);ax.grid(alpha=.17)
ax.legend(loc='center right',frameon=False)
ax.text(.05,.83,r'$\beta=0.98$; 2000 updates',transform=ax.transAxes,fontsize=8)
dest=root/'paper/figures/fig7_revision_geometry_diagnostics'
fig.savefig(dest.with_suffix('.pdf'),bbox_inches='tight')
fig.savefig(dest.with_suffix('.png'),dpi=240,bbox_inches='tight')
plt.close(fig)
print(dest.with_suffix('.pdf'))
