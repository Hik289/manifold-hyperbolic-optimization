"""Export compact, auditable figure inputs and positive-contrast records."""
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT/'paper'
RUNS = ROOT/'runs'


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    evidence = PAPER/'evidence'; evidence.mkdir(exist_ok=True)
    data_dir = PAPER/'figure_data'; data_dir.mkdir(exist_ok=True)
    manifest = json.loads((ROOT/'experiments/revision/manifest.json').read_text())
    configs = {c['id']: c for c in manifest}
    source_records = {}

    def audit(rid):
        folder = RUNS/rid
        result = json.loads((folder/'result.json').read_text())
        rows = [json.loads(line) for line in (folder/'trajectory.jsonl').read_text().splitlines() if line]
        assert result['status'] == 'complete'
        assert result['epochs_completed'] == configs[rid]['epochs'] == len(rows)
        assert [r['epoch'] for r in rows] == list(range(len(rows)))
        assert all(rows[-1][k] == v for k,v in result['final'].items() if k in rows[-1])
        if 'selected_test' in result:
            best = max(rows, key=lambda r:r['val_accuracy'])
            assert result['selected_epoch'] == best['epoch']
            assert result['best_validation'] == best['val_accuracy']
        source_records[rid] = {'result_sha256':digest(folder/'result.json'),
                               'trajectory_sha256':digest(folder/'trajectory.jsonl'),
                               'epochs_verified':len(rows)}
        return result, rows

    def write_csv(name, headers, rows):
        with (data_dir/name).open('w', newline='') as f:
            w = csv.writer(f); w.writerow(headers); w.writerows(rows)

    original = PAPER/'tables/table1c_kg_trajectory.csv'
    with original.open() as f:
        historical = list(csv.DictReader(f))
    write_csv('historical_kg_loss.csv', ['epoch','B3_loss','B4_loss'],
              [[r[k] for k in ['epoch','B3_loss','B4_loss']] for r in historical])
    cora = []
    for m in ['E','B3','B4','B5','A5']:
        _, rows = audit('graph_cora_gcn_'+m)
        cora.extend([m,r['epoch'],r['train_loss']] for r in rows)
    write_csv('revision_cora_loss.csv', ['method','epoch','train_loss'], cora)
    drift = []
    for scale in [2,5,8]:
        for m in ['B3','B5','A5']:
            _, rows = audit(f'mechanism_smooth{scale}_independent_lorentz_3_{m}')
            drift.extend([scale,m,r['epoch'],r['train_loss']] for r in rows)
    write_csv('smooth_drift_loss.csv', ['scale','method','epoch','train_loss'], drift)
    figure_sources = dict(source_records)
    (data_dir/'PROVENANCE.json').write_text(json.dumps({
        'scope':'Every data point used by the three included loss figures. No smoothing, interpolation or synthetic points.',
        'historical_source':{'path':'tables/table1c_kg_trajectory.csv','sha256':digest(original)},
        'run_sources':figure_sources,
        'bundled_csv_sha256':{p.name:digest(p) for p in sorted(data_dir.glob('*.csv'))},
    },indent=2)+'\n')

    paired = json.loads((PAPER/'tables/revision_evidence.json').read_text())
    graph = []
    labels = {'arxiv':'ogbn-arxiv','citeseer':'Citeseer','cora':'Cora','pubmed':'PubMed'}
    encoders = {'sage':'GraphSAGE','gat':'GAT','gcn':'GCN'}
    for r in paired['graph']:
        vals = {}; selected = {}
        for m in ['E','B3','B5','A5']:
            result, _ = audit(f"graph_{r['dataset']}_{r['model']}_{m}")
            vals[m] = result['final']['test_accuracy']
            assert vals[m] == r['final_accuracy'][m]
            selected[m] = result.get('selected_test')
        delta_selected = (100*(selected['B5']['accuracy']-selected['B3']['accuracy'])
                          if selected['B5'] and selected['B3'] else None)
        graph.append({'label':labels[r['dataset']]+' / '+encoders[r['model']],
                      'dataset':r['dataset'],'model':r['model'],'final_accuracy':vals,
                      'delta_final_pp':100*(vals['B5']-vals['B3']),
                      'delta_selected_pp':delta_selected})
    assert len(graph)==12 and sum(r['delta_final_pp']>0 for r in graph)==5

    mech = [c for c in manifest if c['family']=='mechanism' and c['dataset']=='heterogeneous']
    selected = {}
    for m in ['B35','B5','A5']:
        options = [(c,audit(c['id'])[0]) for c in mech if c['method']==m]
        assert len(options)==5
        c,z = min(options, key=lambda cr:cr[1]['final']['val_distortion'])
        selected[m] = z['final']['test_distortion']
    kg = {m:audit('kg_wn18rr_lorentz_translation_'+m)[0]['final']['test_MRR']
          for m in ['B5','A5','shared']}
    continual = {m:audit('continual_rotmnist_mlp_'+m)[0]['final']['average_accuracy']
                 for m in ['B5','A5','B35','shared']}
    narrow = [
        dict(label='Heterogeneous hierarchy',control_name='Selected B35',metric_label=r'Distortion $\downarrow$',
             control=selected['B35'],B5=selected['B5'],A5=selected['A5'],factor=1,digits=6,lower_is_better=True,
             improvement_tex=f"{100*(1-selected['B5']/selected['B35']):.3f}\\%"),
        dict(label='WN18RR / translation',control_name='Shared curvature',metric_label=r'MRR $\uparrow$',
             control=kg['shared'],B5=kg['B5'],A5=kg['A5'],factor=1,digits=6,lower_is_better=False,
             improvement_tex=f"+{kg['B5']-kg['shared']:.6f}"),
    ]
    for m in ['B35','shared']:
        narrow.append(dict(label='Rotated MNIST / MLP',control_name='B35' if m=='B35' else 'Shared curvature',
                           metric_label=r'Mean accuracy (\%) $\uparrow$',control=continual[m],
                           B5=continual['B5'],A5=continual['A5'],factor=100,digits=3,lower_is_better=False,
                           improvement_tex=f"+{100*(continual['B5']-continual[m]):.3f} pp"))
    for r in narrow:
        assert (r['B5'] < r['control']) if r['lower_is_better'] else (r['B5'] > r['control'])
    result = {'scope':'Additional descriptive positive contrasts against explicitly named controls, not new family-level or significance claims. All 12 graph cases retained in this data file; only positive final B5-B3 contrasts displayed. New-task 0.5-percentage-point rule unchanged.',
              'graph':graph,'narrow':narrow,'source_records':source_records,
              'already_present_positive_evidence':[
                  'Historical WN18RR and FB15k-237 curvature evidence with historical-protocol qualification',
                  'Historical synthetic heterogeneity and G8 convergence checkpoints',
                  '48/48 product-factor B5-versus-B3 reconstruction contrasts, with stronger Euclidean control retained',
                  'Three >=0.5% validation-selected mechanism gains (static, smooth2, smooth8)',
                  'Geometric sensitivity and transport consistency operator experiments',
                  'Preliminary short smooth-drift diagnostics, labeled as below full-run requirements',
              ]}
    (evidence/'positive_contrasts.json').write_text(json.dumps(result,indent=2)+'\n')
    complete = []
    for c in manifest:
        p = RUNS/c['id']/'result.json'
        if p.exists():
            z = json.loads(p.read_text())
            assert z['status']=='complete' and z['epochs_completed']==c['epochs']
            complete.append(z)
    assert len(complete)==705
    (evidence/'completed_results.jsonl').write_text(''.join(json.dumps(r, separators=(',',':'))+'\n' for r in complete))
    print(f'Exported {len(cora)} Cora loss points, {len(drift)} drift loss points, 705 result records, and audited positive contrasts.')


if __name__ == '__main__':
    main()
