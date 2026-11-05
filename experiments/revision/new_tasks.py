"""Three prospective task families with validation-only selection and delayed tests."""
import fcntl,json,math,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from geometry import Geometry,origin_embed,L
from extension_data import load
from extension_manifest import METHODS


class Model(nn.Module):
    def __init__(self,cfg,data,device):
        super().__init__();self.cfg=cfg;self.retrieval=cfg['family']=='retrieval';self.directed=cfg['family']=='taxonomy'
        count=int(data['train_y'].max()+1) if self.retrieval else int(data['n'])
        self.geo=Geometry([(count,2)]*4,cfg['method'],device,lr=cfg['lr'],rho=cfg['rho'],beta=cfg['beta'],init_K=cfg['init_K'],k_min=.001,radius=4.)
        self.euclidean=nn.Parameter(torch.stack([p.detach()[:,1:].clone() for p in self.geo.points])) if cfg['method']=='E' else None
        self.em=None if self.euclidean is None else torch.zeros_like(self.euclidean)
        self.bias=nn.Parameter(torch.zeros((),device=device));self.log_temperature=nn.Parameter(torch.zeros((),device=device))
        self.encoder=nn.Sequential(nn.Linear(data['train_x'].shape[1],128),nn.Tanh(),nn.Linear(128,8)).to(device) if self.retrieval else None

    def points(self):return list(self.euclidean) if self.euclidean is not None else self.geo.chart_points()

    def feature_points(self,x):
        parts=self.encoder(x).reshape(len(x),4,2)
        if self.euclidean is not None:return [v/(1+v.norm(dim=-1,keepdim=True)) for v in parts.unbind(1)]
        return [origin_embed(v,k) for v,k in zip(parts.unbind(1),self.geo.K())]

    def distance2(self,a,b):
        if self.euclidean is not None:return sum((x-y).square().sum(-1) for x,y in zip(a,b))
        return sum(L.dist(x,y,k).square() for x,y,k in zip(a,b,self.geo.K()))

    def logits(self,x):
        q=self.feature_points(x);p=self.points()
        return -self.distance2([v[:,None,:] for v in q],[v[None,:,:] for v in p])/self.log_temperature.exp()

    def pair_score(self,pairs,points=None):
        points=self.points() if points is None else points
        a,b=pairs[...,0],pairs[...,1]
        x=[p[a] for p in points];y=[p[b] for p in points]
        distance=self.distance2(x,y)
        if self.directed:
            # Edges point from descendant to ancestor; penalize reversed radial order.
            if self.euclidean is not None:
                ra=sum(p.square().sum(-1) for p in x).clamp_min(1e-12).sqrt()
                rb=sum(p.square().sum(-1) for p in y).clamp_min(1e-12).sqrt()
            else:
                ra=sum(L.dist(p,L.origin(2,k,device=p.device).expand_as(p),k).square() for p,k in zip(x,self.geo.K())).clamp_min(1e-12).sqrt()
                rb=sum(L.dist(p,L.origin(2,k,device=p.device).expand_as(p),k).square() for p,k in zip(y,self.geo.K())).clamp_min(1e-12).sqrt()
            distance=distance+F.relu(rb-ra)
        return self.bias-distance/self.log_temperature.exp()

    def auxiliary_parameters(self):
        return [self.bias,self.log_temperature]+(list(self.encoder.parameters()) if self.encoder is not None else [])

    @torch.no_grad()
    def step(self):
        if self.euclidean is None:self.geo.step()
        else:
            grad=self.euclidean.grad+self.cfg.get('point_decay',0.)*self.euclidean
            grad=grad*(10/grad.norm(dim=-1,keepdim=True).clamp_min(1e-12)).clamp(max=1)
            self.em.mul_(self.cfg['beta']).add_(grad,alpha=1-self.cfg['beta'])
            delta=-self.cfg['lr']*self.em
            delta*= (.25/delta.norm(dim=-1,keepdim=True).clamp_min(1e-12)).clamp(max=1)
            self.euclidean.add_(delta)
            self.euclidean.mul_((4/self.euclidean.norm(dim=-1,keepdim=True).clamp_min(1e-12)).clamp(max=1))
        self.log_temperature.clamp_(-3,3);self.bias.clamp_(-10,10)


def tensor_data(data,device):
    return {k:torch.tensor(v,device=device) for k,v in data.items() if k!='names'}


@torch.no_grad()
def evaluate(model,data,split):
    from sklearn.metrics import average_precision_score,roc_auc_score
    model.eval()
    if model.retrieval:
        gallery=model.feature_points(data['train_x']);labels=data['train_y'];q=data[split+'_x'];targets=data[split+'_y']
        hits=[];ndcgs=[]
        discounts=1/torch.log2(torch.arange(2,12,device=q.device).float())
        counts=torch.bincount(labels)
        for offset in range(0,len(q),64):
            qp=model.feature_points(q[offset:offset+64]);target=targets[offset:offset+64]
            scores=-model.distance2([v[:,None,:] for v in qp],[v[None,:,:] for v in gallery])
            ids=scores.topk(10,dim=-1).indices;relevant=(labels[ids]==target[:,None])
            hits.extend(relevant[:,0].float().cpu().tolist())
            ideal=torch.stack([discounts[:min(10,int(counts[t]))].sum() for t in target])
            ndcgs.extend(((relevant*discounts).sum(-1)/ideal).cpu().tolist())
        return dict(recall1=float(np.mean(hits)),ndcg10=float(np.mean(ndcgs)))
    p=data[split+'_positive'];n=data[split+'_negative'];points=model.points()
    scores=model.pair_score(torch.cat([p,n]),points).cpu().numpy();labels=np.r_[np.ones(len(p)),np.zeros(len(n))]
    result=dict(ap=float(average_precision_score(labels,scores)),auc=float(roc_auc_score(labels,scores)))
    if model.directed:
        known={};targets={}
        for a,b in data['all_positive'].cpu().tolist():known.setdefault(a,set()).add(b)
        for a,b in p.cpu().tolist():targets.setdefault(a,set()).add(b)
        candidates=torch.arange(int(data['n']),device=p.device);aps=[]
        queries=sorted(targets)
        for offset in range(0,len(queries),64):
            batch=queries[offset:offset+64];heads=torch.tensor(batch,device=p.device)
            edges=torch.stack([heads[:,None].expand(-1,len(candidates)),candidates[None,:].expand(len(heads),-1)],-1)
            scores=model.pair_score(edges,points).cpu().numpy()
            for a,score in zip(batch,scores):
                relevant=targets[a];keep=np.ones(len(score),dtype=bool);keep[a]=False
                keep[list(known[a]-relevant)]=False
                truth=np.zeros(len(score),dtype=np.int8);truth[list(relevant)]=1
                aps.append(average_precision_score(truth[keep],score[keep]))
        result['map']=float(np.mean(aps))
    return result


def write_json(path,value):
    path=Path(path);temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temporary.replace(path)


def fit(cfg,root,device,out,allow_selection=True):
    out=Path(out);out.mkdir(exist_ok=True,parents=True)
    raw,provenance=load(cfg['dataset'],root);data=tensor_data(raw,device)
    model=Model(cfg,data,device)
    optimizer=torch.optim.AdamW(model.auxiliary_parameters(),lr=.001,weight_decay=.0001)
    protocol=dict(suite='new_tasks_v1',data=provenance,search_runs_per_method=6,
        primary_metric=cfg['selection_metric'],selection='Complete all 30 configs for the dataset, select each method by final validation metric, then evaluate only the selected checkpoints on test.',
        initialization='Identical seed42; K-dependent Lorentz homothety as in the optimizer.',
        dimensionality='4 factors x 2 intrinsic dimensions; Euclidean control has 8 dimensions.',
        calibration='Every method learns the same score bias and temperature; temperature constrained to exp([-3,3]).',
        known_limitations='Single seed. These are compact custom protocols, not reproductions of published benchmark scores; common canonical cap permits curvature-dependent physical radius.')
    write_json(out/'protocol.json',protocol)
    start=time.monotonic();history=[]
    if not model.retrieval:
        pairs=torch.cat([data['train_positive'],data['train_negative']])
        labels=torch.cat([torch.ones(len(data['train_positive']),device=device),torch.zeros(len(data['train_negative']),device=device)])
    for epoch in range(cfg['epochs']):
        model.train();model.zero_grad(set_to_none=True);model.geo.epoch=epoch
        if model.retrieval:loss=F.cross_entropy(model.logits(data['train_x']),data['train_y'])
        else:loss=F.binary_cross_entropy_with_logits(model.pair_score(pairs),labels)
        if not torch.isfinite(loss):raise FloatingPointError(f'nonfinite loss at {epoch}')
        loss.backward();torch.nn.utils.clip_grad_norm_(model.auxiliary_parameters(),10.)
        optimizer.step();model.step()
        row=dict(epoch=epoch,train_loss=float(loss.detach()),seconds=time.monotonic()-start,
            K=model.geo.K().detach().cpu().tolist() if model.euclidean is None else [],
            temperature=float(model.log_temperature.exp().detach()),peak_memory_mb=torch.cuda.max_memory_allocated()/1e6,**model.geo.last)
        # Cheap held-out loss each epoch; task ranking every 25 epochs and at the endpoint.
        with torch.no_grad():
            if model.retrieval:vl=F.cross_entropy(model.logits(data['validation_x']),data['validation_y'])
            else:
                vp,vn=data['validation_positive'],data['validation_negative']
                vl=F.binary_cross_entropy_with_logits(model.pair_score(torch.cat([vp,vn])),torch.cat([torch.ones(len(vp),device=device),torch.zeros(len(vn),device=device)]))
            row['val_loss']=float(vl)
        if epoch%cfg.get('eval_every',25)==0 or epoch==cfg['epochs']-1:
            met=evaluate(model,data,'validation');row.update({'val_'+k:v for k,v in met.items()})
            if not all(math.isfinite(v) for v in met.values()):raise FloatingPointError('nonfinite validation metric')
            print(json.dumps(row),flush=True)
        with (out/'trajectory.jsonl').open('a') as stream:stream.write(json.dumps(row,allow_nan=False)+'\n')
        history.append(row)
        if (epoch+1)%100==0 or epoch in [0,1,2,5,10,20,50]:
            torch.save(dict(model=model.state_dict(),epoch=epoch,config=cfg,momentum=model.geo.m,euclidean_momentum=model.em),out/'last.pt')
    result=dict(status='complete',config=cfg,final=history[-1],epochs_completed=len(history),
        test_status='withheld_until_all_30_configs_complete',protocol=protocol)
    write_json(out/'result.json',result)
    if allow_selection:finalize_ready(out.parent.parent,root,device,cfg['dataset'])
    return result


def finalize_ready(project,data_root,device,dataset):
    """Release test only after every registered comparator has completed its budget."""
    project=Path(project);runs=project/'runs';dest=runs/'_extension_selection';dest.mkdir(exist_ok=True)
    output=dest/(dataset+'.json')
    with (dest/(dataset+'.lock')).open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if output.exists():return json.loads(output.read_text())
        manifest=json.loads((project/'experiments/revision/manifest.json').read_text())
        configs=[c for c in manifest if c.get('suite')=='new_tasks_v1' and c['dataset']==dataset]
        if len(configs)!=30:raise ValueError('Expected the complete registered 30-run matrix')
        records={}
        for c in configs:
            p=runs/c['id']/'result.json'
            if not p.exists():return None
            r=json.loads(p.read_text())
            if r['epochs_completed']!=c['epochs']:return None
            records[c['id']]=r
        raw,provenance=load(dataset,data_root);data=tensor_data(raw,device);selected={};test={}
        for method in METHODS:
            pool=[c for c in configs if c['method']==method]
            key='val_'+pool[0]['selection_metric']
            chosen=min(pool,key=lambda c:(-records[c['id']]['final'][key],c['id']))
            selected[method]=dict(id=chosen['id'],val_score=records[chosen['id']]['final'][key],config=chosen)
            model=Model(chosen,data,device)
            checkpoint=torch.load(runs/chosen['id']/'last.pt',map_location=device,weights_only=False)
            assert checkpoint['epoch']==chosen['epochs']-1
            model.load_state_dict(checkpoint['model']);test[method]=evaluate(model,data,'test')
        full=selected['B5']['config']
        matched=next(c for c in configs if c['method']=='A5' and c['search_slot']==full['search_slot'])
        model=Model(matched,data,device);model.load_state_dict(torch.load(runs/matched['id']/'last.pt',map_location=device,weights_only=False)['model'])
        matched_test=evaluate(model,data,'test')
        result=dict(dataset=dataset,family=full['family'],metric=full['selection_metric'],
            configs_completed=len(records),created_unix=time.time(),selected=selected,test=test,
            matched_A5=dict(id=matched['id'],test=matched_test),data_sha256=provenance['sha256'],
            selection_scope='Entire 30-run matrix completed before test release. No test-based hyperparameter choice. Exploratory single-seed extension; all task outcomes retained.')
        write_json(output,result)
        print('Released complete selected test matrix:',dataset,flush=True)
        return result


def run(cfg,root,device,out):return fit(cfg,root,device,out)
