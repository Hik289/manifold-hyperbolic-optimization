"""Drain a declared lane without interrupting the original long-running workers."""
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time


def eligible(config, lane):
    if lane=='extension':return config.get('suite')=='new_tasks_v1'
    if lane=='extension_graph':return config.get('suite')=='new_tasks_v1' and config['family']=='link_prediction'
    if lane=='extension_retrieval':return config.get('suite')=='new_tasks_v1' and config['family']=='retrieval'
    if lane == 'mechanism':
        return config['family'] == 'mechanism'
    if lane in ['text','kg','recommendation','vision','synthetic']:
        return config['family']==lane and 'ablation' not in config
    return config['family'] == 'graph' and 'ablation' not in config


def priority(config):
    cases = [('citeseer', 'gcn'), ('pubmed', 'gcn'), ('cora', 'gat'),
             ('cora', 'sage'), ('citeseer', 'sage'), ('pubmed', 'sage'),
             ('20news','mlp'),('wn18rr','lorentz_translation'),
             ('ml1m','mf'),('cifar100','resnet18')]
    pair = (config['dataset'], config['model'])
    case = cases.index(pair) if pair in cases else len(cases)
    methods = ['B5', 'B3', 'A5', 'B35', 'shared', 'B4', 'B2', 'E']
    return (config.get('priority', case), config['dataset'], config['model'],
            methods.index(config['method']))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--gpu', required=True)
    ap.add_argument('--lane', choices=['breadth','mechanism','text','kg','recommendation','vision','synthetic','extension','extension_graph','extension_retrieval'], required=True)
    a = ap.parse_args()
    root = Path(a.root)
    here = Path(__file__).resolve().parent
    scope=here/'execution_scope.json'
    deferred=set(json.loads(scope.read_text())['deferred_ids']) if scope.exists() else set()
    while True:
        queue = sorted((c for c in json.loads((here / 'manifest.json').read_text())
                        if c['id'] not in deferred and eligible(c, a.lane)), key=priority)
        selected = None
        for config in queue:
            out = root / 'runs' / config['id']
            out.mkdir(parents=True, exist_ok=True)
            try:
                fd = os.open(out / 'claim.json', os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                continue
            with os.fdopen(fd, 'w') as stream:
                json.dump(dict(host='local-worker', worker_pid=os.getpid(),
                               gpu=a.gpu, lane=a.lane, started_unix=time.time()), stream)
            selected = config
            break
        if selected is None:
            if a.lane=='extension':
                subprocess.run([sys.executable,str(here/'finalize_new_tasks.py'),'--root',str(root),'--gpu',a.gpu],check=True)
                subprocess.run([sys.executable,str(here/'collect.py'),'--root',str(root),'--paper',str(root/'paper')],check=True)
            return
        (out / 'config.json').write_text(json.dumps(selected, indent=2))
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=a.gpu, PYTHONUNBUFFERED='1',
                   HF_HOME=str(root / 'data/hf_cache'))
        with (out / 'console.log').open('w') as stream:
            result = subprocess.run([sys.executable, str(here / 'runner.py'), '--config',
                                     str(out / 'config.json'), '--root', str(root / 'data'),
                                     '--out', str(out)], env=env, stdout=stream,
                                    stderr=subprocess.STDOUT)
        if result.returncode:
            log = (out / 'console.log').read_text(errors='replace')
            status = 'blocked_data' if 'FileNotFoundError' in log else 'failed'
            (out / 'failure.json').write_text(json.dumps(dict(status=status,
                exit_code=result.returncode, tail=log[-3000:], finished_unix=time.time()), indent=2))
        subprocess.run([sys.executable, str(here / 'collect.py'), '--root', str(root),
                        '--paper', str(root / 'paper')], check=True)


if __name__ == '__main__':
    main()
