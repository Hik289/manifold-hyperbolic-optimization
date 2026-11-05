"""Check the published final-record inventory without training or network access."""
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    configs = json.loads((ROOT/'experiments/revision/manifest.json').read_text())
    scope = json.loads((ROOT/'experiments/revision/execution_scope.json').read_text())
    counts = Counter()
    families = Counter()
    for cfg in configs:
        folder = ROOT/'results/revision'/cfg['id']
        if (folder/'result.json').exists():
            result = json.loads((folder/'result.json').read_text())
            assert result['status'] == 'complete'
            assert result['epochs_completed'] == cfg['epochs']
            # Dispatch priority changed while the queue was running; it does not
            # affect training, initialization, data, epoch count or evaluation.
            assert {k:v for k,v in result['config'].items() if k != 'priority'} == {
                k:v for k,v in cfg.items() if k != 'priority'}
            counts['complete'] += 1
            families[cfg['family']] += 1
        else:
            record = json.loads((folder/'deferred.json').read_text())
            assert record['status'] == 'deferred_scope' and cfg['id'] in scope['deferred_ids']
            counts['deferred'] += 1
    selected = list((ROOT/'results/revision/_extension_selection').glob('*.json'))
    assert counts == {'complete':705, 'deferred':188} and len(selected) == 6
    print(json.dumps({'registered':len(configs), **counts,
                      'completed_by_family':dict(families),
                      'delayed_test_blocks':len(selected)}, indent=2))


if __name__ == '__main__':
    main()
