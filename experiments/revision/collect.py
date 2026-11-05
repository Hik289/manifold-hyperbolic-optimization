"""Read-only result aggregation; generate reviewable tables from complete runs only."""
import argparse,csv,json,time,fcntl
from pathlib import Path
from collections import Counter,defaultdict
ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--paper');ap.add_argument('--runs-dir');a=ap.parse_args();root=Path(a.root);here=Path(__file__).resolve().parent
# Multiple training lanes finish independently; serialize complete snapshots.
(root/'analysis').mkdir(exist_ok=True,parents=True)
snapshot_lock=(root/'analysis/collect.lock').open('a')
fcntl.flock(snapshot_lock,fcntl.LOCK_EX)
manifest=json.loads((here/'manifest.json').read_text());statuses=Counter();families=defaultdict(Counter);records=[];completed=[]
scope_path=here/'execution_scope.json'
scope=json.loads(scope_path.read_text()) if scope_path.exists() else None
scope_deferred=set(scope['deferred_ids']) if scope else set()
for cfg in manifest:
    p=(Path(a.runs_dir) if a.runs_dir else root/'runs')/cfg['id'];status='pending';row={}
    if (p/'result.json').exists():
        result=json.loads((p/'result.json').read_text());status='complete' if result.get('epochs_completed')==cfg['epochs'] else 'incomplete';row=result.get('final',{})
        if status=='complete':completed.append((cfg,result))
    elif (p/'deferred.json').exists():
        assert json.loads((p/'deferred.json').read_text())['status']=='deferred_scope'
        status='deferred_scope'
    elif cfg['id'] in scope_deferred:status='deferred_scope'
    elif (p/'failure.json').exists():status=json.loads((p/'failure.json').read_text()).get('status','failed')
    elif (p/'claim.json').exists():status='claimed' # A claim alone is not proof of a live process.
    statuses[status]+=1;families[cfg['family']][status]+=1
    records.append({**cfg,'status':status,**{k:v for k,v in row.items() if not isinstance(v,(list,dict))}})
out=root/'analysis';out.mkdir(exist_ok=True,parents=True)
keys=list(dict.fromkeys(k for r in records for k in r))
with (out/'coverage.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(records)
snapshot={'created_unix':time.time(),'counts':dict(statuses),'families':{k:dict(v) for k,v in families.items()},'registered':len(manifest),'claim_note':'claimed requires live process confirmation; does not imply running'}
scope_path=here/'execution_scope.json'
scope=json.loads(scope_path.read_text()) if scope_path.exists() else None
if scope:
    deferred_ids={r['id'] for r in records if r['status']=='deferred_scope'}
    if deferred_ids!=set(scope['deferred_ids']):
        snapshot['execution_scope_override']='Local runs include configurations deferred in the published scope.'
        scope=None
if scope:
    assert len(manifest)-len(deferred_ids)==scope['retained_total']
    snapshot['execution_scope']={k:scope[k] for k in ['amendment_id','original_registered','completed_at_amendment','retained_remaining_count','retained_total','deferred_count','termination_rule']}
    snapshot['retained_remaining']=scope['retained_total']-statuses['complete']
(out/'status.json').write_text(json.dumps(snapshot,indent=2))
if a.paper:
    from evidence import write_evidence
    write_evidence(completed,a.paper)
    from extension_evidence import write_extension_evidence
    write_extension_evidence(completed,Path(a.paper)/'tables',Path(a.runs_dir) if a.runs_dir else root/'runs')
    from synthetic_evidence import write_synthetic_evidence
    write_synthetic_evidence(completed, a.paper)
    from long_horizon_evidence import write_long_horizon_evidence
    write_long_horizon_evidence(completed, a.paper, Path(a.runs_dir) if a.runs_dir else root/'runs')
    paper=Path(a.paper);tab=paper/'tables';tab.mkdir(exist_ok=True)
    (tab/'revision_status.json').write_text(json.dumps(snapshot,indent=2));(tab/'revision_coverage.csv').write_text((out/'coverage.csv').read_text())
    graph=[(c,r) for c,r in completed if c['dataset']=='cora' and c['model']=='gcn' and 'ablation' not in c]
    lines=[r'\begin{table}[t]',r'\centering',r'\caption{compute worker revision: Cora/GCN, seed 42, complete 1000-epoch runs. Final checkpoint is reported consistently; no significance claim is made. E uses a Euclidean linear head; other rows use Lorentz prototypes. The encoder is shared in form and trained with AdamW.}',r'\label{tab:revision-cora}',r'\begin{tabular}{lrrrr}',r'\toprule',r'Method & Final val. acc. & Final test acc. & Test macro-F1 & Seconds \\',r'\midrule']
    for c,r in graph:
        f=r['final'];lines.append(f"{c['method']} & {f['val_accuracy']:.4f} & {f['test_accuracy']:.4f} & {f['test_macro_f1']:.4f} & {f['seconds']:.1f} \\\\")
    lines.extend([r'\bottomrule',r'\end{tabular}',r'\end{table}'])
    (tab/'revision_cora.tex').write_text('\n'.join(lines)+'\n')

    progress=(f"The generated execution snapshot contains {len(manifest)} registered configurations: "
              f"{statuses['complete']} complete, {statuses['claimed']} claimed, "
              f"{statuses['pending']} pending, {statuses['blocked_data']} blocked on data, "
              f"{statuses['failed']} failed, and {statuses['deferred_scope']} deferred by scope amendment. "
              + (f"The amended execution budget retains {scope['retained_total']} configurations, "
                 f"of which {scope['retained_total']-statuses['complete']} are unfinished. " if scope else "") +
              "A claim alone does not establish that a process is live. "
              "Only full-length completed runs appear in the generated result table.\\par\n")
    (tab/'revision_progress.tex').write_text(progress)
    def esc(x):return str(x).replace('_',r'\_').replace('%',r'\%')
    all_lines=[r'\section{Generated compute worker Completed-Run Ledger}',r'\label{app:run-ledger}',r'\input{tables/revision_progress}',r'\scriptsize',r'\begin{longtable}{llllrrr}',r'\caption{Complete runs only. Score is final test accuracy for classification, average accuracy for continual learning, MRR for KG, NDCG@20 for recommendation, fixed-pair distortion for the original synthetic suite, and held-out pair distortion for the new mechanism suite (lower is better only for distortion). For new tasks, unselected rows report the final validation primary score; selected test results are reported separately after the complete search. Rows across different datasets are not compared.}\\',r'\toprule',r'Dataset & Model & Method & Ablation & Epochs & Score & Seconds \\',r'\midrule',r'\endfirsthead',r'\toprule',r'Dataset & Model & Method & Ablation & Epochs & Score & Seconds \\',r'\midrule',r'\endhead']
    for c,r in completed:
        f=r['final'];key={'kg':'test_MRR','recommendation':'test_ndcg20','synthetic':'distortion_fixed_pairs','mechanism':'test_distortion','continual':'average_accuracy'}.get(c['family'],'test_micro_f1' if c['dataset']=='reuters' else 'test_accuracy')
        if c.get('suite')=='new_tasks_v1':key='val_'+c['selection_metric']
        score=f.get(key)
        if score is None:continue
        all_lines.append(' & '.join([esc(c['dataset']+(f" n{c['nodes']} d{c['dim']}" if 'nodes' in c else '')),esc(c['model']),esc(c['method']),esc(c.get('ablation','search_s'+str(c['search_slot']) if 'search_slot' in c else '--')),str(c['epochs']),f'{score:.4f}',f"{f['seconds']:.1f}"])+r' \\')
    all_lines.extend([r'\bottomrule',r'\end{longtable}',r'\normalsize'])
    (tab/'revision_all_results.tex').write_text('\n'.join(all_lines)+'\n')
if a.paper:
    from table_style import style_directory
    style_directory(Path(a.paper)/'tables')
print(json.dumps(snapshot,indent=2))
