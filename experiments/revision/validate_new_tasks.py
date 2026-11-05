"""Small device-portable path/leakage checks, never counted as formal experiment results."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import torch
from extension_data import edge_split,assert_split
from extension_manifest import generate
import new_tasks

ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);a=ap.parse_args()
root=Path(a.root);cache=root/'new_tasks_v1';cache.mkdir(exist_ok=True,parents=True)
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');torch.set_num_threads(2)
checks=[];rng=np.random.default_rng(42)
for family in ['link_prediction','taxonomy','retrieval']:
    name='fixture_'+family
    if family=='retrieval':
        data={}
        for split,n in [('train',30),('validation',10),('test',10)]:
            data[split+'_x']=rng.normal(size=(n,6)).astype(np.float32);data[split+'_y']=np.arange(n,dtype=np.int64)%2
    else:
        edges=[(i,j) for i in range(24) for j in range(i) if j==i//2 or (i+j)%7==0]
        data=edge_split(24,edges,family=='taxonomy');assert_split(data,family=='taxonomy')
        checks.append({'check':'disjoint_'+family,'passed':True})
    path=cache/(name+'.npz');np.savez_compressed(path,**data)
    (cache/(name+'.json')).write_text(json.dumps({'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'fixture':True}))
    for method in ['E','B3','B35','B5','A5']:
        torch.manual_seed(42)
        cfg=next(c.copy() for c in generate() if c['family']==family and c['method']==method)
        cfg.update(dataset=name,epochs=2,eval_every=1,id='fixture_'+family+'_'+method)
        calls=[];original=new_tasks.evaluate
        def checked_evaluate(model,data,split):
            calls.append(split);return original(model,data,split)
        new_tasks.evaluate=checked_evaluate
        out=root/'checks'/cfg['id'];result=new_tasks.fit(cfg,root,device,out,allow_selection=False)
        new_tasks.evaluate=original
        assert calls==['validation','validation'],calls
        assert result['epochs_completed']==2 and not any(k.startswith('test_') for k in result['final'])
        raw,_=new_tasks.load(name,root);td=new_tasks.tensor_data(raw,device);model=new_tasks.Model(cfg,td,device)
        model.load_state_dict(torch.load(out/'last.pt',map_location=device,weights_only=False)['model'])
        measured=original(model,td,'test')
        assert all(np.isfinite(v) and 0<=v<=1 for v in measured.values()),measured
        checks.append({'check':family+'_'+method,'training':True,'test_withheld_during_fit':True,'evaluation_range':True})
(root/'validation.json').write_text(json.dumps(checks,indent=2)+'\n')
print(json.dumps(checks,indent=2))
