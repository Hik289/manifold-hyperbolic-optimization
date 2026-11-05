"""One-off, audited budget amendment; no manifest or training-setting changes.

Historical amendment replay requires an explicit --apply flag. Atomic claim reservations make the amendment compatible with
already loaded workers. Deferred records are separate from numerical failures.
"""
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time

import psutil

ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / 'experiments/revision'
VISION = 'vision_cifar100_resnet18_B5'
KEEP = [
    'continual_rotmnist_mlp_B35', 'continual_rotmnist_mlp_shared',
    'kg_wn18rr_lorentz_translation_B4', 'recommendation_ml1m_mf_B35',
    'text_20news_transformer_A5',
    'kg_wn18rr_lorentz_translation_E', 'recommendation_ml1m_mf_E',
    'text_20news_transformer_E',
] + ['kg_wn18rr_lorentz_distance_' + m for m in ['E', 'B2', 'B3', 'B4', 'B5', 'A5']]


def write_json(path, data):
    tmp = path.with_name(path.name + '.scope-tmp')
    tmp.write_text(json.dumps(data, indent=2) + '\n')
    tmp.replace(path)


def schedulers():
    found = []
    for p in psutil.process_iter(['cmdline']):
        cmd = p.info['cmdline'] or []
        if any(str(HERE / n) in cmd for n in ['supervisor.py', 'worker.py', 'priority_worker.py']):
            assert str(ROOT) in cmd
            found.append(p)
    return found


def main():
    assert ROOT.is_dir()
    record = HERE / 'execution_scope.json'
    if record.exists():
        raise RuntimeError('Scope already recorded; inspect it instead of applying twice.')
    manifest_bytes = (HERE / 'manifest.json').read_bytes()
    manifest = json.loads(manifest_bytes)
    cfgs = {c['id']: c for c in manifest}
    assert len(cfgs) == len(manifest) == 893 and len(KEEP) == len(set(KEEP)) == 14
    assert set(KEEP) <= cfgs.keys()
    paused = []
    stopped_vision_worker = None
    try:
        # Stop dispatch only; retained training children keep computing.
        parents = schedulers()
        parents.sort(key=lambda p: str(HERE / 'supervisor.py') not in p.cmdline())
        for p in parents:
            p.send_signal(signal.SIGSTOP)
            paused.append(p)
        for p in schedulers():
            if p.pid not in {q.pid for q in paused}:
                p.send_signal(signal.SIGSTOP)
                paused.append(p)
        time.sleep(0.2)
        parent_ids = {p.pid for p in paused}
        running = {}
        for p in psutil.process_iter(['cmdline', 'ppid']):
            cmd = p.info['cmdline'] or []
            if str(HERE / 'runner.py') in cmd and p.info['ppid'] in parent_ids:
                rid = Path(cmd[cmd.index('--out') + 1]).name
                assert rid in cfgs and rid not in running
                running[rid] = p
        assert set(running) == set(KEEP[:5]) | {VISION}, sorted(running)
        complete = []
        deferred = []
        for c in manifest:
            out = ROOT / 'runs' / c['id']
            result = out / 'result.json'
            if result.exists():
                r = json.loads(result.read_text())
                assert r['status'] == 'complete' and r['epochs_completed'] == c['epochs']
                complete.append(c['id'])
            elif c['id'] not in KEEP:
                if c['id'] != VISION:
                    assert not (out / 'claim.json').exists(), c['id']
                    assert not (out / 'trajectory.jsonl').exists(), c['id']
                deferred.append(c['id'])
        assert len(complete) == 691 and len(deferred) == 188
        assert not set(complete) & set(KEEP)
        for rid in KEEP[5:]:
            assert not (ROOT / 'runs' / rid / 'claim.json').exists(), rid
        created = time.time()
        amendment = dict(
            schema_version=1, amendment_id='2026-11-04-budget-reduction',
            created_unix=created, host='local-worker',
            authorization='User requested simplifying the remaining work and prioritizing promising experiments.',
            original_registered=893, completed_at_amendment=691,
            original_unfinished=202, retained_remaining_count=14,
            retained_total=705, deferred_count=188,
            manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
            completed_ids=complete, retained_remaining_ids=KEEP, deferred_ids=deferred,
            rationale=[
                'Finish five already running controls and three missing Euclidean controls in the corresponding blocks.',
                'Retain the unopened WN18RR relation-weighted distance block with all six originally registered methods; direct metric fitting tests the hierarchy hypothesis within the existing KG family.',
                'Defer broad dataset/encoder expansion and the expensive partial visual run; no completed result is removed.',
            ],
            inference_scope='Exploratory budget amendment made after observing existing outcomes and compute costs, not a preregistered confirmatory selection or a guarantee of benefit.',
            unchanged=['manifest', 'seed42', 'data splits', 'full registered epoch counts', 'hyperparameters', 'checkpoint selection', 'primary metrics', 'new-task display threshold'],
            termination_rule='Finish all 705 retained full-length configurations and required delayed tests; 188 deferred configurations are not counted as completed or numerical failures.',
            scheduler_pids=[p.pid for p in paused],
            running_before={rid: p.pid for rid, p in running.items()},
        )
        # Validate everything before reserving any omitted job.
        write_json(ROOT / 'analysis/scope_amendment_pre_apply.json', amendment)
        for rid in deferred:
            out = ROOT / 'runs' / rid
            out.mkdir(parents=True, exist_ok=True)
            if not (out / 'config.json').exists():
                write_json(out / 'config.json', cfgs[rid])
            state = dict(status='deferred_scope', amendment_id=amendment['amendment_id'],
                         changed_unix=created, not_started=(rid != VISION),
                         reason='Deferred by user-authorized compute-budget reduction; not a numerical failure or completed benchmark.')
            if rid == VISION:
                rows = [json.loads(x) for x in (out / 'trajectory.jsonl').read_text().splitlines() if x]
                state['last_observed_epoch_zero_based'] = rows[-1]['epoch']
                state['registered_epochs'] = cfgs[rid]['epochs']
                state['original_claim'] = json.loads((out / 'claim.json').read_text())
            write_json(out / 'deferred.json', state)
            if rid != VISION:
                fd = os.open(out / 'claim.json', os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                with os.fdopen(fd, 'w') as stream:
                    json.dump({'deferred_scope': True, 'amendment_id': amendment['amendment_id'],
                               'not_started': True, 'reserved_unix': created}, stream)
            else:
                write_json(out / 'claim.json', {**state['original_claim'], 'deferred_scope': True,
                                              'amendment_id': amendment['amendment_id']})
        # Stop only the verified visual lane, keeping all existing checkpoint files.
        visual = running[VISION]
        worker = psutil.Process(visual.ppid())
        assert '--lane' in worker.cmdline() and worker.cmdline()[-1] == 'vision'
        assert str(HERE / 'priority_worker.py') in worker.cmdline()
        visual.send_signal(signal.SIGSTOP)
        children = visual.children(recursive=True)
        worker.kill()  # Prevent its wait handler from writing a spurious failure.
        stopped_vision_worker = worker.pid
        for child in children:
            try:
                child.terminate()
            except psutil.NoSuchProcess:
                pass
        visual.terminate()
        visual.send_signal(signal.SIGCONT)
        _, alive = psutil.wait_procs([visual, worker] + children, timeout=10)
        for p in alive:
            if p.status() != psutil.STATUS_ZOMBIE:
                p.kill()
        out = ROOT / 'runs' / VISION
        amendment['stopped_visual'] = dict(
            runner_pid=visual.pid, worker_pid=worker.pid,
            files_retained=[dict(name=p.name, bytes=p.stat().st_size,
                                sha256=hashlib.sha256(p.read_bytes()).hexdigest())
                            for p in out.iterdir() if p.name in ['last.pt', 'best.pt', 'trajectory.jsonl']],
        )
        write_json(record, amendment)
        write_json(ROOT / 'analysis/scope_amendment.json', amendment)
        print(json.dumps({k: amendment[k] for k in ['amendment_id', 'completed_at_amendment', 'retained_remaining_count', 'retained_total', 'deferred_count', 'stopped_visual']}, indent=2), flush=True)
    finally:
        for p in paused:
            if p.pid == stopped_vision_worker:
                continue
            try:
                p.send_signal(signal.SIGCONT)
            except psutil.NoSuchProcess:
                pass


if __name__ == '__main__':
    import argparse
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',type=Path,default=ROOT)
    ap.add_argument('--apply',action='store_true',help='Replay the historical amendment; signals matching scheduler processes.')
    args=ap.parse_args();ROOT=args.root.resolve();HERE=ROOT/'experiments/revision'
    if args.apply:main()
    else:print(json.dumps({'mode':'inspect_only','retained_after_amendment':705,'deferred':188,'applied':False},indent=2))
