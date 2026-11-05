"""Loss-only figures and WWW-reference numerical tables, from retained records."""
import ast,csv,json,math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2];PAPER=ROOT/'paper';TAB=PAPER/'tables';RUNS=ROOT/'runs'

def read_csv(path):
    with Path(path).open() as f:return list(csv.DictReader(f))
def esc(s):return str(s).replace('_',r'\_').replace('%',r'\%')
def bold(s):return r'\textbf{'+s+'}'
def table(name,caption,label,headers,rows,wide=True):
    env='table*' if wide else 'table'
    placement='t'
    if name in ['presentation_kg_checkpoints.tex','presentation_drift_states.tex']:
        env='table';placement='H'
    lines=[r'\begin{'+env+'}['+placement+']',r'\centering',r'\caption{'+caption+'}',r'\label{'+label+'}',
        r'\footnotesize\setlength{\tabcolsep}{4.2pt}\renewcommand{\arraystretch}{1.12}',
        r'\begin{adjustbox}{max width=\linewidth}',r'\begin{tabular}{'+'l'+'r'*(len(headers)-1)+'}',r'\toprule',
        ' & '.join(headers)+r' \\',r'\midrule']
    for row in rows:lines.append(row if isinstance(row,str) else ' & '.join(row)+r' \\')
    lines += [r'\bottomrule',r'\end{tabular}',r'\end{adjustbox}',r'\end{'+env+'}']
    (TAB/name).write_text('\n'.join(lines)+'\n')

# Curvature values: the CSV and literal data retained with the original plotting source.
wn=read_csv(TAB/'table1b_kg_K_per_relation.csv')
source=ast.parse((ROOT/'figures/fig1_curvature_bars_code.py').read_text());values={}
for node in source.body:
    if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name):
        if node.targets[0].id in ['fb_rels_raw','fb_labels_manual']:values[node.targets[0].id]=ast.literal_eval(node.value)
rows=[]
for i in range(max(len(wn),len(values['fb_rels_raw']))):
    left=[esc(wn[i]['relation'].lstrip('_').replace('_',' ')),f"{float(wn[i]['K']):.4f}"] if i<len(wn) else ['--','--']
    right=[esc(values['fb_labels_manual'][i].replace('→',r'$\to$')),f"{values['fb_rels_raw'][i][1]:.4f}"] if i<len(values['fb_rels_raw']) else ['--','--']
    rows.append(left+right)
table('presentation_curvatures.tex',r'\textbf{Learned relation curvatures.} Historical B4 values. All 11 WN18RR relations and the 15 displayed FB15k-237 relations are retained; the fixed baseline is $K=1$. Curvature magnitude is a parameter, not a performance ranking.','tab:curvatures',['WN18RR relation','$K$','FB15k-237 relation','$K$'],rows)

# Historical quality metrics and full convergence checkpoint table.
kg=[r for r in read_csv(TAB/'table1_kg_wn18rr.csv') if r['run'] in ['kgfinal_B3','kgfinal_B4']]
rows=[]
for window in ['last5','last10']:
    rows.append(r'\multicolumn{5}{l}{\textit{WN18RR, '+window.replace('last','last ')+r' evaluations}} \\')
    for row in kg:
        vals=[float(row[k+'_'+window]) for k in ['MRR','H1','H10','loss']]
        strings=[]
        for j,(key,v) in enumerate(zip(['MRR','H1','H10','loss'],vals)):
            best=(min if key=='loss' else max)(float(r[key+'_'+window]) for r in kg)
            text=f'{v:.4f}';strings.append(bold(text) if v==best else text)
        if row['method']=='B4':rows.append(r'\rowcolor{oursbg}')
        rows.append([row['method']]+strings)
table('presentation_kg_main.tex',r'\textbf{Historical WN18RR ranking and fitting.} B3 and B4 use the historical Euclidean-Adam protocol; B4 learns curvature and is shaded. Best values within each window are bold, including B3 on Hits@1. These unfiltered tail-only scores are separate from the new filtered protocol.','tab:kg-main',['Method',r'MRR $\uparrow$',r'Hits@1 $\uparrow$',r'Hits@10 $\uparrow$',r'Loss $\downarrow$'],rows,False)
trajectory=read_csv(TAB/'table1c_kg_trajectory.csv');rows=[]
for r in trajectory:
    rows.append([r['epoch']]+[f"{float(r[k]):.4f}" for k in ['B3_MRR','B4_MRR','B3_H10','B4_H10','B4_Kstd']])
table('presentation_kg_checkpoints.tex',r'\textbf{Historical KG metric checkpoints.} Recorded MRR, Hits@10 and learned curvature spread. Loss alone is plotted in Figure~\ref{fig:kg-loss}.','tab:kg-checkpoints',['Epoch','B3 MRR','B4 MRR','B3 Hits@10','B4 Hits@10',r'B4 $K_{\rm std}$'],rows)
het=[r for r in read_csv(TAB/'table4_heterogeneity.csv') if r['run'].startswith('het_br')];rows=[]
for r in het:
    if r['method']=='B5':rows.append(r'\rowcolor{oursbg}')
    scores=[]
    for key in ['final_distortion','final_loss']:
        v=float(r[key]);st=f'{v:.5f}';scores.append(bold(st) if v==min(float(x[key]) for x in het) else st)
    rows.append([r['method']]+scores+[f"{float(r['K_g'+str(i)]):.4f}" for i in range(4)])
table('presentation_heterogeneity.tex',r'\textbf{Branching heterogeneity.} The four groups have branching rates $[2,8,32,128]$. Historical final distortion/loss and all learned group curvatures are reported.','tab:heterogeneity',['Method',r'Distortion $\downarrow$',r'Loss $\downarrow$','$K_1$','$K_2$','$K_3$','$K_4$'],rows)
rows=[]
for r in read_csv(TAB/'table6_convergence_G8.csv'):
    ds=[float(r[k]) for k in ['B3_dist','B4_dist','B5_dist']]
    vals=[bold(f'{v:.5f}') if v==min(ds) else f'{v:.5f}' for v in ds];vals[2]=r'\cellcolor{oursbg}'+vals[2]
    rows.append([r['epoch']]+vals+[r[k]+r'\%' for k in ['B4_vs_B3_pct','B5_vs_B3_pct']])
table('presentation_g8.tex',r'\textbf{Complete recorded G8 distortion checkpoints.} Distortion and its relative change use the same recorded checkpoints for every method. Negative changes favor the corresponding learned-curvature method.','tab:g8',['Epoch','B3','B4','B5',r'$\Delta$ B4/B3',r'$\Delta$ B5/B3'],rows)

# Drifting metric results and curvature states at fixed checkpoints.
rows=[];history={}
for scale in [2,5,8]:
    for method in ['B3','B5','A5']:
        d=RUNS/f'mechanism_smooth{scale}_independent_lorentz_3_{method}'
        assert json.loads((d/'result.json').read_text())['epochs_completed']==2000
        h=[json.loads(x) for x in (d/'trajectory.jsonl').read_text().splitlines()]
        assert [x['epoch'] for x in h]==list(range(2000));history[(scale,method)]=h
    for epoch in [0,99,499,999,1999]:
        ds=[history[(scale,m)][epoch]['val_distortion'] for m in ['B3','B5','A5']]
        vals=[bold(f'{v:.4f}') if round(v,4)==round(min(ds),4) else f'{v:.4f}' for v in ds];vals[1]=r'\cellcolor{oursbg}'+vals[1]
        rows.append([f'{scale}'+r'$\times$',str(epoch+1)]+vals+[f'{k:.3f}' for k in history[(scale,'B5')][epoch]['K']])
table('presentation_drift_states.tex',r'\textbf{Smooth-drift validation distortion and curvature states.} All methods start at $K_0=1$. $K_1,\ldots,K_4$ are B5 states. These are fixed checkpoints of the complete recorded trajectories, distinct from validation-selected initialization comparisons.','tab:drift-states',['Drift','Update','B3','B5','A5','$K_1$','$K_2$','$K_3$','$K_4$'],rows)

# Measured operator diagnostics; no analytic quantity is called a benchmark score.
sens=read_csv(ROOT/'analysis/theory_diagnostics/sensitivity.csv');rows=[]
for z in [.1,1.,10.]:
    subsets={n:[r for r in sens if r['precision']=='float64' and r['normalized']==n and math.isclose(float(r['scaled_momentum']),z,rel_tol=1e-6)] for n in ['False','True']}
    assert all(len(v)==5 for v in subsets.values())
    u=subsets['False'];v=subsets['True']
    rows.append([f'{z:g}',f"{float(u[0]['exact_transverse_gain']):.6g}",f"{float(v[0]['exact_transverse_gain']):.6g}",f"{max(float(r['relative_error']) for r in u+v):.2e}"])
table('presentation_sensitivity.tex',r'\textbf{Measured sensitivity check.} FP64 transverse finite perturbations use all five $K\in\{0.01,0.1,1,10,100\}$. Exact gains are listed; the final column is the largest measured relative discrepancy across these curvatures and both operators. No trust cap is used in this operator check.','tab:sensitivity',[r'\makecell{Scaled\\momentum}',r'\makecell{Uncapped\\gain}',r'\makecell{Normalized\\gain}',r'\makecell{Max. rel.\\error $\downarrow$}'],rows,False)
transport=read_csv(ROOT/'analysis/theory_diagnostics/transport.csv');rows=[]
for prec in ['float32','float64']:
    rows.append(r'\multicolumn{4}{l}{\textit{'+prec.replace('float','FP')+r', maximum absolute canonical momentum error over 2000 updates}} \\')
    for label,key in [('Point only','ablated_error'),('Point + momentum','full_error')]:
        vals=[max(float(r[key]) for r in transport if r['precision']==prec and float(r['scale_end'])==scale) for scale in [2,5,8]]
        if key=='full_error':rows.append(r'\rowcolor{oursbg}')
        rows.append([label]+[(bold(f'{v:.2e}') if key=='full_error' else f'{v:.2e}') for v in vals])
table('presentation_transport.tex',r'\textbf{Cross-curvature state consistency.} Curvature drift is prescribed, gradients are canonical constants, and the position does not move. This isolates optimizer-state consistency; it is not a downstream accuracy comparison.','tab:operator-transport',['State mapping',r'$2\times$ drift',r'$5\times$ drift',r'$8\times$ drift'],rows)
rows=[[str(r),f'{np.cosh(r)**2*2**-23:.3e}'] for r in [0,2,4,5,8,10]]
table('presentation_conditioning.tex',r'\textbf{Illustrative FP32 conditioning proxy.} At unit curvature, $\cosh^2(r)2^{-23}$ illustrates radius amplification. It is not a measured error or a universally attained lower bound; the rigorous roundoff bound is stated in Appendix~\ref{app:complete-theory}.','tab:conditioning',['Radius $r$',r'$\cosh^2(r)2^{-23}$'],rows,False)

# Redraw only actual logged training losses; no bars or metric/curvature plots.
plt.rcParams.update({'font.family':'DejaVu Serif','font.size':8,'axes.labelsize':8,'legend.fontsize':7,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
colors={'E':'#777777','B3':'#4477AA','B4':'#CC6677','B5':'#228833','A5':'#AA3377'}
def save(fig,name):
    fig.savefig(PAPER/'figures'/f'{name}.pdf',bbox_inches='tight');fig.savefig(PAPER/'figures'/f'{name}.png',dpi=220,bbox_inches='tight');plt.close(fig)
fig,ax=plt.subplots(figsize=(3.25,2.1),layout='constrained')
for m in ['B3','B4']:ax.plot([int(r['epoch']) for r in trajectory],[float(r[m+'_loss']) for r in trajectory],color=colors[m],marker='o',ms=2.5,lw=1.3,label=m)
ax.set(xlabel='Epoch',ylabel='Training loss');ax.grid(alpha=.16);ax.legend(frameon=False);save(fig,'fig_loss_historical_kg')
fig,ax=plt.subplots(figsize=(3.25,2.15),layout='constrained')
for m in ['E','B3','B4','B5','A5']:
    h=[json.loads(x) for x in (RUNS/f'graph_cora_gcn_{m}'/'trajectory.jsonl').read_text().splitlines()]
    assert len(h)==1000
    ax.plot([r['epoch']+1 for r in h],[r['train_loss'] for r in h],color=colors[m],lw=1.4 if m=='B5' else 1,ls='--' if m=='A5' else '-',label=m)
ax.set(xlabel='Epoch',ylabel='Training loss',yscale='log');ax.grid(alpha=.16);ax.legend(frameon=False,ncol=3);save(fig,'fig_loss_revision_cora')
fig,axs=plt.subplots(1,3,figsize=(6.75,2.0),layout='constrained')
for ax,scale in zip(axs,[2,5,8]):
    for m in ['B3','B5','A5']:
        h=history[(scale,m)];ax.plot([r['epoch']+1 for r in h],[r['train_loss'] for r in h],color=colors[m],lw=1.5 if m=='B5' else 1.,ls='--' if m=='A5' else '-',label=m)
    ax.set(xlabel='Update',ylabel='Mean relative MSE',title=f'{scale}× drift');ax.grid(alpha=.16)
axs[0].legend(frameon=False);save(fig,'fig_loss_smooth_drift')
print('Generated 9 numerical tables (plus the method table) and 3 loss-only figures from recorded data.')
