"""Selected test tables and a complete audit, with explicit display selection."""
import json
from pathlib import Path
from extension_manifest import CASES,METHODS

LABELS={'wordnet_mammal':'WordNet mammals','wordnet_vehicle':'WordNet vehicles',
        'cora_links':'Cora links','citeseer_links':'Citeseer links',
        'digits_retrieval':'Digits retrieval','20news_retrieval':'20News retrieval'}
BROAD={'taxonomy':'knowledge-graph / taxonomy','link_prediction':'graph','retrieval':'retrieval'}

def write_extension_evidence(completed,tab,runs):
    tab=Path(tab);runs=Path(runs);records=[];coverage=[]
    for family,dataset in CASES:
        done=[(c,r) for c,r in completed if c.get('suite')=='new_tasks_v1' and c['dataset']==dataset]
        coverage.append(dict(family=family,dataset=dataset,completed=len(done),registered=30))
        path=runs/'_extension_selection'/(dataset+'.json')
        if not path.exists():continue
        record=json.loads(path.read_text());metric=record['metric'];v={m:record['test'][m][metric] for m in METHODS}
        record['B5_minus_baselines_pp']={m:100*(v['B5']-v[m]) for m in ['E','B3','B35']}
        record['positive_display_eligible']=all(v['B5']>=v[m]+.005 for m in ['E','B3','B35'])
        record['broad_family']=BROAD[family];records.append(record)
    families={}
    for family in sorted({f for f,_ in CASES}):
        rows=[r for r in records if r['family']==family]
        families[BROAD[family]]=dict(completed_tasks=len(rows),registered_tasks=2,
            supported_on_both_tasks=len(rows)==2 and all(r['positive_display_eligible'] for r in rows))
    def table(rows,caption,label):
        if not rows:
            if len(records)==len(CASES) and all(c['completed']==c['registered'] for c in coverage):
                return r'\noindent All six additional tasks have completed their full searches and delayed test evaluations. None meets the stated display criterion; the complete outcomes are retained in the appendix.\par'+'\n'
            return r'\noindent No new task has yet met the complete-comparison and display criteria. Pending comparisons are not evidence of an advantage.\par'+'\n'
        lines=[r'\begin{table}[t]',r'\centering\small',r'\caption{'+caption+'}',r'\label{'+label+'}',r'\begin{tabular}{lrrrrrr}',r'\toprule',r'Task & E & B3 & B35 & B5 & A5 & Matched A5 \\',r'\midrule']
        for row in rows:
            key=row['metric'];vals=[100*row['test'][m][key] for m in METHODS]
            matched=100*row['matched_A5']['test'][key]
            lines.append(' & '.join([LABELS[row['dataset']]]+[f'{v:.2f}' for v in vals]+[f'{matched:.2f}'])+r' \\')
        lines += [r'\bottomrule',r'\end{tabular}',r'\end{table}']
        return '\n'.join(lines)+'\n'
    common=r'Final-checkpoint test scores in percent; higher is better. Taxonomy: filtered query MAP; links: balanced-pair AP; retrieval: same-class Recall@1. Each method selects from six runs by final validation metric; all 30 configurations finish before any test evaluation. Matched A5 uses B5\textquotesingle s selected learning rate and initial curvature. Seed 42; 1000 epochs.'
    (tab/'revision_new_tasks_all.tex').write_text(table(records,common+' Every completed task is retained, including ties and negative differences.','tab:new-tasks-all'))
    (tab/'revision_new_tasks_positive.tex').write_text(table([r for r in records if r['positive_display_eligible']],common+' This outcome-selected display requires B5 to exceed E, B3 and B35 by at least 0.5 percentage points. The complete matrix is in Appendix~\ref{app:extended-audit}; selection is not statistical confirmation.','tab:new-tasks-positive'))
    counts=', '.join(f"{LABELS[c['dataset']]}: {c['completed']}/30" for c in coverage)
    (tab/'revision_new_tasks_progress.tex').write_text('Registered new-task completion counts: '+counts+r'. Test comparison requires the full 30-run block.\par'+'\n')
    (tab/'revision_new_tasks_evidence.json').write_text(json.dumps(dict(coverage=coverage,results=records,family_assessment=families,criterion='Both predeclared tasks per broad family must exceed all three controls by >=0.005 absolute primary score before a descriptive family-level advantage is claimed. Single-seed evidence, not significance.'),indent=2)+'\n')
