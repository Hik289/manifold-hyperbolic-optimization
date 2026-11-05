"""Audited full-length endpoints; keep primary metrics and selection rules explicit."""
import hashlib
import json
from pathlib import Path

METHODS = ('E', 'B2', 'B3', 'B4', 'B5', 'A5', 'B35', 'shared')
GROUPS = {
    'text': ('text', '20news', 'mlp'),
    'transformer': ('text', '20news', 'transformer'),
    'kg': ('kg', 'wn18rr', 'lorentz_translation'),
    'kg_distance': ('kg', 'wn18rr', 'lorentz_distance'),
    'recommendation': ('recommendation', 'ml1m', 'mf'),
    'continual': ('continual', 'rotmnist', 'mlp'),
}


def audit_record(config, result, runs_dir):
    folder = Path(runs_dir) / config['id']
    raw = (folder / 'result.json').read_bytes()
    trajectory = (folder / 'trajectory.jsonl').read_bytes()
    rows = [json.loads(line) for line in trajectory.splitlines() if line.strip()]
    assert result['status'] == 'complete'
    assert result['config'] == config
    assert config['seed'] == 42
    assert len(rows) == result['epochs_completed'] == config['epochs']
    assert [row['epoch'] for row in rows] == list(range(config['epochs']))
    final = result['final']
    for key, value in final.items():
        assert rows[-1][key] == value, (config['id'], key)
    selected = None
    if 'selected_test' in result:
        best = max(rows, key=lambda row: row['val_accuracy'])
        assert best['epoch'] == result['selected_epoch']
        assert best['val_accuracy'] == result['best_validation']
        selected = dict(epoch_zero_based=best['epoch'],
                        validation_accuracy=best['val_accuracy'],
                        test=result['selected_test'])
    if config['family'] == 'continual':
        boundaries = [row for row in rows if 'task_accuracies' in row]
        assert len(boundaries) == 5 and len(final['task_accuracies']) == 5
        assert abs(sum(final['task_accuracies']) / 5 - final['average_accuracy']) < 1e-12
        past_best = [max(row['task_accuracies'][j] for row in boundaries
                         if len(row['task_accuracies']) > j) for j in range(4)]
        forgetting = sum(past_best[j] - final['task_accuracies'][j] for j in range(4)) / 4
        assert abs(forgetting - final['forgetting']) < 1e-12
    source_file = folder / 'source_hashes.json'
    return dict(config=config, final=final, selected=selected,
                result_sha256=hashlib.sha256(raw).hexdigest(),
                trajectory_sha256=hashlib.sha256(trajectory).hexdigest(),
                epochs_verified=len(rows),
                source_hashes=json.loads(source_file.read_text()) if source_file.exists() else None)


def table(caption, label, headers, rows, digits, lower_columns=(), plain_columns=()):
    """Bold numeric extrema after formatting, including ties at displayed precision."""
    shown = [[f'{value:.{digits[j]}f}' for j, value in enumerate(values)]
             for _, values in rows]
    best = {j: (min if j in lower_columns else max)(float(row[j]) for row in shown)
            for j in range(len(digits)) if j not in plain_columns}
    lines = [r'% WWW_REFERENCE_STYLE', r'\begin{table}[!htbp]', r'\centering',
             r'\caption{' + caption + '}', r'\label{' + label + '}',
             r'\footnotesize\setlength{\tabcolsep}{5pt}\renewcommand{\arraystretch}{1.12}',
             r'\begin{tabular}{l' + 'r' * len(digits) + '}', r'\toprule',
             ' & '.join(['Method'] + headers) + r' \\', r'\midrule']
    for (method, _), cells in zip(rows, shown):
        if method == 'B5':
            lines.append(r'\rowcolor{oursbg}')
        for j, value in enumerate(cells):
            if j in best and float(value) == best[j]:
                cells[j] = r'\textbf{' + value + '}'
        lines.append(' & '.join([method.capitalize() if method == 'shared' else method] + cells) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    return '\n'.join(lines) + '\n'


def write_long_horizon_evidence(completed, paper, runs_dir):
    tab = Path(paper) / 'tables'
    manifest = json.loads(Path(__file__).with_name('manifest.json').read_text())
    registered = {name: {c['method'] for c in manifest
                        if (c['family'], c['dataset'], c['model']) == target
                        and 'ablation' not in c}
                  for name, target in GROUPS.items()}
    deferred = {name: {c['method'] for c in manifest
                      if (c['family'], c['dataset'], c['model']) == target
                      and 'ablation' not in c
                      and (Path(runs_dir) / c['id'] / 'deferred.json').exists()}
                for name, target in GROUPS.items()}
    groups = {name: {} for name in GROUPS}
    for config, result in completed:
        key = (config['family'], config['dataset'], config['model'])
        for name, target in GROUPS.items():
            if key == target and config['method'] in METHODS and 'ablation' not in config:
                assert config['method'] not in groups[name]
                groups[name][config['method']] = audit_record(config, result, runs_dir)
    evidence = {
        'groups': {name: dict(completed_methods=len(rows), registered_methods=len(registered[name]),
                             deferred_methods=sorted(deferred[name]),
                             retained_methods=len(registered[name]-deferred[name]),
                             retained_block_complete=(registered[name]-deferred[name]) <= rows.keys(),
                             entire_block_complete=registered[name] <= rows.keys(), records=rows)
                   for name, rows in groups.items()},
        'scope': 'Descriptive registered configurations, seed 42. Text reports validation-selected and final checkpoints separately. Other tables report final endpoints. No outcome-based reruns or claim of a new task-family advantage.',
        'source_provenance': 'Per-run source digests are retained; runs are not asserted to have identical complete source snapshots. Missing digests remain null.',
    }
    (tab / 'revision_long_horizon_evidence.json').write_text(json.dumps(evidence, indent=2) + '\n')
    text = groups['text']
    if set(METHODS) <= text.keys():
        rows = [(m, [text[m]['selected']['epoch_zero_based'] + 1,
                     100 * text[m]['selected']['validation_accuracy'],
                     100 * text[m]['selected']['test']['accuracy'],
                     100 * text[m]['selected']['test']['macro_f1'],
                     100 * text[m]['final']['test_accuracy']]) for m in METHODS]
        content = table(
            r'\textbf{Complete 20 Newsgroups classification block.} Eight methods, 1000 epochs each, seed 42. The registry labels the pooled-token encoder as \texttt{mlp}. The first maximum validation-accuracy checkpoint selects each method\textquotesingle s test result; epochs are one-based. All scores are percentages. Final accuracy is a separately labeled endpoint, not an alternative selection rule.',
            'tab:revision-text-audit',
            ['Selected epoch', r'Val. acc.\ $\uparrow$', r'Test acc.\ $\uparrow$',
             r'Test macro-F1\ $\uparrow$', r'Final acc.\ $\uparrow$'], rows, [0, 3, 3, 3, 3], plain_columns=(0,))
        (tab / 'revision_text_audit.tex').write_text(content)
        score = {m: 100 * text[m]['selected']['test']['accuracy'] for m in METHODS}
        summary = (
            '\\paragraph{Completed long-horizon classification control.}\n'
            'The eight-method 20 Newsgroups classification block now completes 1000 epochs per method. '
            'Selecting each checkpoint by validation accuracy gives test accuracy '
            f"${score['B5']:.3f}\\%$ for B5, ${score['B3']:.3f}\\%$ for B3 and B35, "
            f"and ${score['E']:.3f}\\%$ for E. "
            'Thus this block does not add a classification-family advantage. '
            'Appendix~\\ref{app:long-horizon-audit} reports every method, selected epochs, macro-F1, '
            'and final endpoints, together with the available full-length WN18RR, MovieLens-1M, '
            'and Rotated MNIST comparisons. Partial blocks and secondary-metric increases '
            'do not extend the positive family claims.\n')
    else:
        (tab / 'revision_text_audit.tex').write_text('% Awaiting complete eight-method classification block.\n')
        summary = '% Awaiting complete eight-method classification block.\n'

    transformer = groups['transformer']
    if {'B3', 'B5'} <= transformer.keys():
        rows = [(m, [transformer[m]['selected']['epoch_zero_based'] + 1,
                     100 * transformer[m]['selected']['validation_accuracy'],
                     100 * transformer[m]['selected']['test']['accuracy'],
                     100 * transformer[m]['selected']['test']['macro_f1'],
                     100 * transformer[m]['final']['test_accuracy']])
                for m in METHODS if m in transformer]
        caption = (
            r'\textbf{20 Newsgroups Transformer: completed method endpoints.} '
            'Each listed method finishes 1000 epochs, seed 42. The first maximum '
            'validation-accuracy checkpoint selects its test result; epochs are one-based. '
            f"{len(rows)}/{len(registered['transformer'])} registered methods complete; "
            f"{len(deferred['transformer'])} methods deferred by the budget amendment. "
            'Every completed method is retained. All scores are percentages; final accuracy '
            'is separately labeled. Bold compares only the displayed rows; B5 is shaded.')
        (tab / 'revision_transformer_audit.tex').write_text(table(
            caption, 'tab:revision-transformer-audit',
            ['Selected epoch', r'Val. acc.\ $\uparrow$', r'Test acc.\ $\uparrow$',
             r'Test macro-F1\ $\uparrow$', r'Final acc.\ $\uparrow$'],
            rows, [0, 3, 3, 3, 3], plain_columns=(0,)))
        b5 = 100 * transformer['B5']['selected']['test']['accuracy']
        b3 = 100 * transformer['B3']['selected']['test']['accuracy']
        transformer_complete = (registered['transformer']-deferred['transformer']) <= transformer.keys()
        coverage_sentence = (
            f"All {len(transformer)} retained Transformer methods are complete; "
            f"{len(deferred['transformer'])} other registered methods were deferred. "
            if transformer_complete else 'The retained Transformer block remains incomplete. ')
        summary += (
            '\\paragraph{Additional completed controls.}\n'
            'For the separate two-layer Transformer on 20 Newsgroups, the completed '
            f'B5--B3 pair gives validation-selected test accuracy ${b5:.3f}\\%$ versus '
            f'${b3:.3f}\\%$ (B5 minus B3: ${b5-b3:+.3f}$ percentage points). '
            'This is one encoder-specific contrast within text classification. '
            + coverage_sentence +
            'Table~\\ref{tab:revision-transformer-audit} preserves both the selected '
            'and final checkpoints. ')
        rec = groups['recommendation']
        if {'B5', 'A5'} <= rec.keys():
            b5_rec, a5_rec = rec['B5']['final'], rec['A5']['final']
            if all(b5_rec[k] == a5_rec[k] for k in ['test_ndcg20', 'test_recall20']):
                summary += ('The MovieLens-1M A5 control exactly matches '
                            'B5 in the recorded final NDCG@20 and Recall@20, so this '
                            'pair does not isolate a transport-performance advantage. ')
        summary += 'The complete available rows and their primary metrics remain in the appendix.\n'
    else:
        (tab / 'revision_transformer_audit.tex').write_text('% Awaiting completed Transformer B3 and B5.\n')
    distance = groups['kg_distance']
    if registered['kg_distance'] <= distance.keys():
        score = {m: distance[m]['final']['test_MRR'] for m in registered['kg_distance']}
        summary += (
            '\\paragraph{Completed distance-model comparison.}\n'
            'All six WN18RR relation-weighted distance configurations complete 2000 epochs. '
            f"Final filtered MRR is ${score['B5']:.5f}$ for B5, ${score['B3']:.5f}$ for B3, "
            f"${score['E']:.5f}$ for E, and ${score['A5']:.5f}$ for A5. "
            'This complete block does not add a task-family advantage or an isolated transport gain; '
            'Table~\\ref{tab:revision-kg-distance-audit} reports all six methods and all ranking endpoints.\n')
    (tab / 'revision_long_horizon_summary.tex').write_text(summary)

    specs = [
        ('kg', 'revision_kg_audit.tex', 'tab:revision-kg-audit',
         r'WN18RR transported-translation model, 2000 epochs. All-entity filtered head-and-tail ranking at the final checkpoint. MRR is primary; Hits scores are percentages.',
         [r'MRR $\uparrow$', r'Hits@1 (\%) $\uparrow$', r'Hits@3 (\%) $\uparrow$', r'Hits@10 (\%) $\uparrow$'],
         [('test_MRR', 1), ('test_Hits1', 100), ('test_Hits3', 100), ('test_Hits10', 100)], [5, 3, 3, 3], ()),
        ('kg_distance', 'revision_kg_distance_audit.tex', 'tab:revision-kg-distance-audit',
         r'WN18RR relation-weighted distance model, 2000 epochs. All six registered methods are retained under the budget amendment. All-entity filtered head-and-tail ranking at the final checkpoint. MRR is primary; Hits scores are percentages.',
         [r'MRR $\uparrow$', r'Hits@1 (\%) $\uparrow$', r'Hits@3 (\%) $\uparrow$', r'Hits@10 (\%) $\uparrow$'],
         [('test_MRR', 1), ('test_Hits1', 100), ('test_Hits3', 100), ('test_Hits10', 100)], [5, 3, 3, 3], ()),
        ('recommendation', 'revision_recommendation_audit.tex', 'tab:revision-recommendation-audit',
         r'MovieLens-1M product-distance MF, 1000 epochs. Final all-item user-macro scores after chronological splitting and observed-item masking. NDCG@20 is primary; Recall@20 is secondary. Both are percentages.',
         [r'NDCG@20 $\uparrow$', r'Recall@20 $\uparrow$'],
         [('test_ndcg20', 100), ('test_recall20', 100)], [4, 4], ()),
        ('continual', 'revision_retention_audit.tex', 'tab:revision-retention-audit',
         r'Rotated MNIST, five ordered tasks and 1000 total epochs without replay. Mean accuracy across all five tasks is primary; forgetting averages the first four tasks, in percentage points. Last-task accuracy is secondary. All rows use the final checkpoint.',
         [r'Mean acc. (\%) $\uparrow$', r'Forgetting (pp) $\downarrow$', r'Last-task acc. (\%) $\uparrow$'],
         [('average_accuracy', 100), ('forgetting', 100), ('test_accuracy', 100)], [3, 3, 3], (1,)),
    ]
    for name, filename, label, description, headers, keys, digits, lower in specs:
        records = groups[name]
        if not {'B3', 'B5'} <= records.keys():
            (tab / filename).write_text('% Awaiting completed B3 and B5 endpoints.\n')
            continue
        rows = [(m, [records[m]['final'][key] * factor for key, factor in keys])
                for m in METHODS if m in records]
        coverage = f' {len(rows)}/{len(registered[name])} registered methods complete; all completed rows are shown. '
        if deferred[name]:
            noun = 'method is' if len(deferred[name]) == 1 else 'methods are'
            coverage += f'{len(deferred[name])} {noun} deferred by the budget amendment. '
        caption = r'\textbf{Available full-length endpoints.} ' + description + coverage
        caption += 'Bold compares only the displayed completed rows, including rounded ties; B5 is shaded. Seed 42, fixed starting configuration, no equal-budget hyperparameter search.'
        (tab / filename).write_text(table(caption, label, headers, rows, digits, lower))
