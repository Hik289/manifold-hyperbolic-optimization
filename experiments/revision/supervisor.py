"""Keep paper-generated tables current while the experiment workers run."""
import subprocess,sys,time,json
from pathlib import Path
from priority_worker import eligible
root=Path(sys.argv[1]);here=Path(__file__).resolve().parent
def load_manifest():
    scope=here/'execution_scope.json'
    deferred=set(json.loads(scope.read_text())['deferred_ids']) if scope.exists() else set()
    return [c for c in json.loads((here/'manifest.json').read_text()) if c['id'] not in deferred]
while True:
    acquisition=root/'data/acquisition.json'
    if acquisition.exists():
        acq=json.loads(acquisition.read_text())
        for cfg in load_manifest():
            info=acq.get(cfg['dataset'],{})
            out=root/'runs'/cfg['id']
            if info.get('status')=='available' and (out/'failure.json').exists():
                failure=json.loads((out/'failure.json').read_text())
                claim=json.loads((out/'claim.json').read_text()) if (out/'claim.json').exists() else {}
                if failure.get('not_started') and claim.get('blocked_data') and not (out/'result.json').exists():
                    (out/'previous_data_block.json').write_text(json.dumps(failure,indent=2))
                    (out/'failure.json').unlink();(out/'claim.json').unlink()
            if info.get('status')!='blocked':continue
            out=root/'runs'/cfg['id'];out.mkdir(exist_ok=True,parents=True)
            if (out/'result.json').exists() or (out/'claim.json').exists():continue
            (out/'config.json').write_text(json.dumps(cfg,indent=2))
            (out/'failure.json').write_text(json.dumps({'status':'blocked_data','reason':info.get('reason'),'not_started':True},indent=2))
            (out/'claim.json').write_text(json.dumps({'blocked_data':True,'started_unix':time.time()}))
    # A later manifest expansion is picked up after existing workers exit.
    import psutil
    active=[]
    for proc in psutil.process_iter(['cmdline']):
        try:
            cmd=proc.info['cmdline'] or []
            if any(Path(x).name in ['worker.py','priority_worker.py'] for x in cmd) and str(root) in cmd:active.append(cmd)
        except (psutil.NoSuchProcess,psutil.AccessDenied):pass
    for gpu in ['0','1']:
        if not any(any(Path(x).name=='worker.py' for x in cmd) and '--ablations-only' not in cmd and '--gpu' in cmd and cmd[cmd.index('--gpu')+1]==gpu for cmd in active):
            log=open(root/f'worker_supervised_{gpu}.log','a')
            command=[sys.executable,str(here/'worker.py'),'--root',str(root),'--gpu',gpu]
            subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            active.append(command)
    pending_ablations=any('ablation' in cfg and not (root/'runs'/cfg['id']/'claim.json').exists() for cfg in load_manifest())
    gpu0_workers=sum('--gpu' in cmd and cmd[cmd.index('--gpu')+1]=='0' for cmd in active)
    if pending_ablations and gpu0_workers<4 and not any('--ablations-only' in cmd for cmd in active):
        log=open(root/'worker_supervised_ablations.log','a')
        command=[sys.executable,str(here/'worker.py'),'--root',str(root),'--gpu','0','--ablations-only']
        subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        active.append(command)
    configs=load_manifest()
    lane_specs=[('extension','0'),('extension_graph','1'),('extension_retrieval','0'),('breadth','0'),('mechanism','1'),('text','0'),('kg','1'),('recommendation','1'),('vision','0'),('synthetic','1')]
    # Share only the two devices already assigned to this project. Limit the
    # number of our live workers; never stop or modify unrelated GPU processes.
    worker_counts={gpu:sum('--gpu' in cmd and cmd[cmd.index('--gpu')+1]==gpu for cmd in active) for gpu in ['0','1']}
    for lane,gpu in lane_specs:
        pending=any(eligible(c,lane) and not (root/'runs'/c['id']/'claim.json').exists() for c in configs)
        if lane=='extension':
            datasets={c['dataset'] for c in configs if c.get('suite')=='new_tasks_v1'}
            for dataset in datasets:
                block=[c for c in configs if c.get('suite')=='new_tasks_v1' and c['dataset']==dataset]
                if not (root/'runs/_extension_selection'/f'{dataset}.json').exists() and all((root/'runs'/c['id']/'result.json').exists() for c in block):pending=True
        live=any('--lane' in cmd and cmd[cmd.index('--lane')+1]==lane for cmd in active)
        if pending and not live and worker_counts[gpu]<(5 if lane.startswith('extension') else 4):
            query=subprocess.run(['nvidia-smi','-i',gpu,'--query-gpu=memory.free','--format=csv,noheader,nounits'],capture_output=True,text=True)
            if query.returncode or int(query.stdout.strip())<16384:continue
            log=open(root/f'worker_priority_{lane}.log','a')
            subprocess.Popen([sys.executable,str(here/'priority_worker.py'),'--root',str(root),'--gpu',gpu,'--lane',lane],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            worker_counts[gpu]+=1
    subprocess.run([sys.executable,str(here/'collect.py'),'--root',str(root),'--paper',str(root/'paper')])
    s=json.loads((root/'analysis/status.json').read_text())
    test_complete=all((root/'runs/_extension_selection'/f"{c['dataset']}.json").exists() for c in configs if c.get('suite')=='new_tasks_v1')
    if s['counts'].get('pending',0)==0 and s['counts'].get('claimed',0)==0 and test_complete:break
    time.sleep(60)
