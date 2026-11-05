"""Generate paired evidence tables, retaining ties and negative results."""
import json
from collections import defaultdict
from pathlib import Path


def esc(value):
    return str(value).replace('_',r'\_')


def write_evidence(completed, paper):
    tab=Path(paper)/'tables'
    tab.mkdir(parents=True,exist_ok=True)
    graph=defaultdict(dict);mechanism=defaultdict(list)
    for c,r in completed:
        if c['family']=='graph' and 'ablation' not in c:
            graph[(c['dataset'],c['model'])][c['method']]=r
        elif c['family']=='mechanism':
            mechanism[c['dataset']].append((c,r))
    records=[]
    lines=[r'\begin{table}[t]',r'\centering\small',
        r'\caption{Completed graph comparisons on compute worker, seed 42, 1000 epochs. Final test accuracy (\%) is the common endpoint; $\Delta$ is B5 minus B3 in percentage points. Every dataset--encoder pair with both completed runs is shown, including ties and negative differences. E is an architecture control with a Euclidean linear head. Dashes mean a comparator has not finished.}',
        r'\label{tab:revision-breadth}',r'\begin{tabular}{llrrrrr}',r'\toprule',
        r'Dataset & Encoder & E & B3 & B5 & A5 & $\Delta$ \\',r'\midrule']
    for (dataset,model),runs in sorted(graph.items()):
        if not {'B3','B5'}<=runs.keys():continue
        vals={m:runs[m]['final']['test_accuracy'] for m in runs}
        delta=100*(vals['B5']-vals['B3'])
        records.append(dict(dataset=dataset,model=model,final_accuracy=vals,
            B5_minus_B3_pp=delta,validation_selected={m:r.get('selected_test') for m,r in runs.items()}))
        def value(m):return f'{100*vals[m]:.2f}' if m in vals else '--'
        lines.append(' & '.join([esc(dataset),esc(model),value('E'),value('B3'),value('B5'),value('A5'),f'{delta:+.2f}'])+r' \\')
    lines += [r'\bottomrule',r'\end{tabular}',r'\end{table}']
    (tab/'revision_graph_breadth.tex').write_text('\n'.join(lines)+'\n')
    mechrecords=[]
    lines=[r'\begin{table}[t]',r'\centering\small',
        r'\caption{Long-horizon hierarchy mechanism tests: final held-out pair distortion after 2000 updates (lower is better). All methods use seed 42 and identical pair splits. Only cases with completed B5, B3, and A5 are included. The best-fixed column requires all five B35 curvature settings and selects using final validation distortion, never test distortion. Missing comparisons remain dashes. These controlled synthetic cases are not additional real-data task families.}',
        r'\label{tab:revision-mechanism}',r'\begin{tabular}{lrrrrrr}',r'\toprule',
        r'Case & B3 & B5 & A5 & Shared & B4 & Best fixed \\',r'\midrule']
    for case,results in sorted(mechanism.items()):
        main={c['method']:r for c,r in results if 'ablation' not in c}
        if not {'B3','B5','A5'}<=main.keys():continue
        fixed=[(c,r) for c,r in results if c['method']=='B35']
        selected=min(fixed,key=lambda cr:cr[1]['final']['val_distortion']) if len(fixed)==5 else None
        vals={m:r['final']['test_distortion'] for m,r in main.items()}
        record=dict(case=case,final_test_distortion=vals,
            validation_auc={m:r['final']['validation_auc'] for m,r in main.items()},
            B5_relative_gain_vs_B3=1-vals['B5']/vals['B3'],
            B5_relative_gain_vs_A5=1-vals['B5']/vals['A5'],
            best_fixed=None if selected is None else dict(K=selected[0].get('init_K',1.),test_distortion=selected[1]['final']['test_distortion']))
        mechrecords.append(record)
        def value(m):return f'{vals[m]:.4f}' if m in vals else '--'
        fixedvalue='--' if selected is None else f"{selected[1]['final']['test_distortion']:.4f}"
        lines.append(' & '.join([esc(case),value('B3'),value('B5'),value('A5'),value('shared'),value('B4'),fixedvalue])+r' \\')
    lines += [r'\bottomrule',r'\end{tabular}',r'\end{table}']
    if not mechrecords:
        lines=[r'\noindent The predeclared long-horizon mechanism matrix is still running; no incomplete run is treated as a result.\par']
    (tab/'revision_mechanism.tex').write_text('\n'.join(lines)+'\n')
    selection_records=[]
    lines=[r'\begin{table}[t]',r'\centering\small',
        r'\caption{Matched five-initialization comparison, 2000 updates per run. B35, B5, and A5 each use $K_0\in\{0.03,0.1,0.3,1,3\}$, with configurations selected by final validation distortion. Test distortion is reported after selection. A case appears only when all 15 runs finish. The last column compares B5 with A5 at B5\textquotesingle s selected initialization, retaining the clean transport contrast.}',
        r'\label{tab:revision-selected-mechanism}',r'\begin{tabular}{lrrrrr}',r'\toprule',
        r'Case & Best fixed & B5 & Best A5 & B5 $K_0$ & Matched A5 \\',r'\midrule']
    for case,results in sorted(mechanism.items()):
        pools={m:[(c,r) for c,r in results if c['method']==m] for m in ['B35','B5','A5']}
        if not all(len(pool)==5 for pool in pools.values()):continue
        selected={m:min(pool,key=lambda cr:cr[1]['final']['val_distortion']) for m,pool in pools.items()}
        k=selected['B5'][0].get('init_K',1.)
        matched=next(r for c,r in pools['A5'] if c.get('init_K',1.)==k)
        vals={m:cr[1]['final']['test_distortion'] for m,cr in selected.items()}
        record=dict(case=case,test_distortion=vals,selected_K={m:cr[0].get('init_K',1.) for m,cr in selected.items()},
                    matched_A5_test_distortion=matched['final']['test_distortion'],
                    relative_gain_over_best_fixed=1-vals['B5']/vals['B35'])
        selection_records.append(record)
        lines.append(' & '.join([esc(case),f"{vals['B35']:.4f}",f"{vals['B5']:.4f}",f"{vals['A5']:.4f}",f'{k:g}',f"{matched['final']['test_distortion']:.4f}"])+r' \\')
    lines += [r'\bottomrule',r'\end{tabular}',r'\end{table}']
    if not selection_records:lines=[r'\noindent The matched five-initialization controls are still running; no validation-selected superiority claim is made until the complete comparison is available.\par']
    (tab/'revision_selected_mechanism.tex').write_text('\n'.join(lines)+'\n')
    positive=[r for r in selection_records if r['relative_gain_over_best_fixed']>=.005]
    focus=[r'\begin{table}[t]',r'\centering\small',
        r'\caption{Selected synthetic conditions with at least 0.5\% lower test distortion than validation-selected fixed curvature. Each method receives the same five initializations and 2000 updates. This is an outcome-selected display from six cases; all cases, including ties and negative results, appear in Appendix~\ref{app:extended-audit}. Different drift conditions are not different task families.}',
        r'\label{tab:revision-mechanism-focus}',r'\begin{tabular}{lrrrr}',r'\toprule',
        r'Condition & Best fixed & B5 & Best A5 & Reduction \\',r'\midrule']
    for r in positive:
        v=r['test_distortion']
        focus.append(' & '.join([esc(r['case']),f"{v['B35']:.4f}",f"{v['B5']:.4f}",f"{v['A5']:.4f}",f"{100*r['relative_gain_over_best_fixed']:.2f}\\%"])+r' \\')
    focus += [r'\bottomrule',r'\end{tabular}',r'\end{table}']
    (tab/'revision_mechanism_focus.tex').write_text('\n'.join(focus)+'\n')
    evidence=dict(graph=records,mechanism=mechrecords,
        validation_selected_mechanism=selection_records,
        interpretation='Single-seed descriptive comparisons. No significance or across-seed robustness claim. All complete eligible pairs retained.')
    (tab/'revision_evidence.json').write_text(json.dumps(evidence,indent=2)+'\n')
