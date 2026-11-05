"""Run an exact registered configuration, or an explicitly separate CPU fixture."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    actions = ap.add_mutually_exclusive_group(required=True)
    actions.add_argument('--list', action='store_true')
    actions.add_argument('--id')
    actions.add_argument('--smoke', action='store_true')
    actions.add_argument('--all-retained', action='store_true')
    ap.add_argument('--family', help='Filter --list or --all-retained by task family.')
    ap.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    ap.add_argument('--work-dir', type=Path, default=ROOT/'artifacts/reproduction')
    ap.add_argument('--data-root', type=Path,
                    default=Path(os.environ.get('MANIFOLD_DATA', str(ROOT/'data'))))
    args = ap.parse_args()
    manifest = json.loads((ROOT/'experiments/revision/manifest.json').read_text())
    scope = json.loads((ROOT/'experiments/revision/execution_scope.json').read_text())
    deferred = set(scope['deferred_ids'])
    if args.list:
        for cfg in manifest:
            if not args.family or cfg['family'] == args.family:
                print(cfg['id'], 'deferred' if cfg['id'] in deferred else 'retained',
                      f"epochs={cfg['epochs']}", sep='\t')
        return
    if args.smoke:
        cfg = next(c.copy() for c in manifest
                   if c['id'] == 'mechanism_smooth2_independent_lorentz_3_B5')
        cfg.update(id='fixture_mechanism_B5', epochs=3, nodes_per_group=16,
                   dim=2, fixture_only=True)
        configs = [cfg]
    elif args.id:
        configs = [c for c in manifest if c['id'] == args.id]
        if not configs:
            ap.error('Unknown configuration ID; use --list.')
    else:
        configs = [c for c in manifest if c['id'] not in deferred
                   and (not args.family or c['family'] == args.family)]
        if not configs:
            ap.error('No retained configurations match the requested family.')
    work = args.work_dir.resolve()
    data = args.data_root.resolve()
    # Delayed test release resolves the full search manifest from the work tree.
    work_manifest = work/'experiments/revision/manifest.json'
    work_manifest.parent.mkdir(parents=True, exist_ok=True)
    if work_manifest.exists() and json.loads(work_manifest.read_text()) != manifest:
        raise RuntimeError('Work directory belongs to a different experiment manifest.')
    work_manifest.write_text(json.dumps(manifest, indent=2)+'\n')
    for cfg in configs:
        out = work/'runs'/cfg['id']
        out.mkdir(parents=True, exist_ok=True)
        completed = out/'result.json'
        if completed.exists():
            saved = json.loads(completed.read_text())
            if saved.get('config') != cfg or saved.get('epochs_completed') != cfg['epochs']:
                raise RuntimeError('Existing result does not match the requested complete configuration.')
            print('Already complete:', cfg['id'])
            continue
        if (out/'trajectory.jsonl').exists():
            raise RuntimeError('An incomplete trajectory exists; choose a new work directory to avoid mixing runs.')
        config = out/'config.json'
        config.write_text(json.dumps(cfg, indent=2)+'\n')
        print('Fixture only:' if cfg.get('fixture_only') else 'Registered run:', cfg['id'], flush=True)
        subprocess.run([sys.executable, str(ROOT/'experiments/revision/runner.py'),
                        '--config', str(config), '--root', str(data), '--out', str(out),
                        '--device', args.device], check=True)
    print('Results:', work/'runs')


if __name__ == '__main__':
    main()
