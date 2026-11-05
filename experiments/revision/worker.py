"""Persistent two-GPU queue with atomic claims and explicit failure states."""
import argparse,json,os,subprocess,sys,time,fcntl,socket
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--gpu',required=True);ap.add_argument('--max-jobs',type=int,default=0);ap.add_argument('--ablations-only',action='store_true');a=ap.parse_args()
root=Path(a.root);here=Path(__file__).resolve().parent;queue=json.loads((here/'manifest.json').read_text());runs=root/'runs';runs.mkdir(exist_ok=True,parents=True)
scope=here/'execution_scope.json'
deferred=set(json.loads(scope.read_text())['deferred_ids']) if scope.exists() else set()
count=0
for cfg in queue:
    if cfg['id'] in deferred:continue
    if a.ablations_only and 'ablation' not in cfg:continue
    out=runs/cfg['id'];out.mkdir(exist_ok=True);claim=out/'claim.json'
    try:
        fd=os.open(claim,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    except FileExistsError:continue
    os.write(fd,json.dumps({'host':'local-worker','worker_pid':os.getpid(),'gpu':a.gpu,'started_unix':time.time()}).encode());os.close(fd)
    (out/'config.json').write_text(json.dumps(cfg,indent=2))
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=a.gpu,PYTHONUNBUFFERED='1',HF_HOME=str(root/'data/hf_cache'))
    with (out/'console.log').open('w') as log:
        result=subprocess.run([sys.executable,str(here/'runner.py'),'--config',str(out/'config.json'),'--root',str(root/'data'),'--out',str(out)],env=env,stdout=log,stderr=subprocess.STDOUT)
    if result.returncode:
        log=(out/'console.log').read_text(errors='replace');status='blocked_data' if ('FileNotFoundError' in log or 'ConnectionError' in log or 'HTTPError' in log) else 'failed'
        (out/'failure.json').write_text(json.dumps({'status':status,'exit_code':result.returncode,'tail':log[-3000:],'finished_unix':time.time()},indent=2))
    subprocess.run([sys.executable,str(here/'collect.py'),'--root',str(root)])
    count+=1
    if a.max_jobs and count>=a.max_jobs:break
