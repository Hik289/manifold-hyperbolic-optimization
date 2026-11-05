"""Synthetic, filtered KG, and implicit recommendation tasks with real manifold states."""
import json,math,time,zipfile,urllib.request,gzip
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from geometry import Geometry,L

def fetch(url,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if not path.exists():
        tmp=path.with_suffix(path.suffix+'.partial');urllib.request.urlretrieve(url,tmp);tmp.rename(path)
    return path

def sorted_contains(sorted_values, queries):
    """Exact membership without re-sorting a million observed pairs per batch."""
    if not len(sorted_values):return torch.zeros_like(queries,dtype=torch.bool)
    positions=torch.searchsorted(sorted_values,queries)
    return (positions<len(sorted_values)) & (sorted_values[positions.clamp_max(len(sorted_values)-1)]==queries)

def triples(root,name):
    from train_kg import load_triples
    p=root/name
    if not (p/'train.txt').exists():raise FileNotFoundError(f'KG data missing: {p}')
    e,r,s=load_triples(str(p));return len(e),len(r),s

def split_recommendation(root,name):
    if name=='yelp' and (root/'yelp2018/train.txt').exists():
        def read(path):
            result={}
            for line in path.read_text().splitlines():
                values=list(map(int,line.split()))
                if values:result[values[0]]=sorted(set(values[1:]))
            return result
        original=read(root/'yelp2018/train.txt');tests=read(root/'yelp2018/test.txt')
        rng=np.random.default_rng(42);train=[];val=[];test=[];seen={}
        for u,items in sorted(original.items()):
            if len(items)<2:raise ValueError('Yelp2018 user has insufficient training interactions')
            if set(items)&set(tests.get(u,[])):raise ValueError('Yelp2018 train/test overlap')
            held=items[int(rng.integers(len(items)))];remaining=set(items)-{held}
            train.extend((u,i) for i in items if i!=held);val.append((u,held));seen[u]=remaining
            test.extend((u,i) for i in tests.get(u,[]))
        nu=max(set(original)|set(tests))+1;ni=max(i for by in [original,tests] for items in by.values() for i in items)+1
        return nu,ni,train,val,test,seen
    if name=='ml1m':
        p=fetch('https://files.grouplens.org/datasets/movielens/ml-1m.zip',root/'ml-1m.zip')
        if not (root/'ml-1m/ratings.dat').exists():
            with zipfile.ZipFile(p) as z:z.extractall(root)
        rows=[]
        for line in (root/'ml-1m/ratings.dat').open():
            u,i,r,t=line.strip().split('::');rows.append((u,i,int(t)))
    else:
        p=root/(name+'.jsonl')
        if not p.exists():raise FileNotFoundError(f'{p}: use the original subset or prepare_recommendation.py; no silent substitute')
        rows=[]
        for line in p.open():
            r=json.loads(line);rows.append((str(r['user']),str(r['item']),int(r.get('timestamp',0))))
    users={};items={};by={}
    for u,i,t in rows:
        ui=users.setdefault(u,len(users));ii=items.setdefault(i,len(items));by.setdefault(ui,{})[ii]=t
    train=[];val=[];test=[];seen={}
    for u,ir in by.items():
        ordered=sorted(ir,key=lambda i:(ir[i],i))
        if len(ordered)<5:continue
        train.extend((u,i) for i in ordered[:-2]);val.append((u,ordered[-2]));test.append((u,ordered[-1]));seen[u]=set(ordered[:-2])
    return len(users),len(items),train,val,test,seen

class ProductModel(torch.nn.Module):
    def __init__(self,counts,method,device,model='mf',edges=None,lr=.1,rho=.001):
        super().__init__();self.counts=counts;self.groups=4;self.model=model;self.edges=edges
        self.geo=Geometry([(sum(counts),8)]*4,method,device,lr=lr,rho=rho)
        self.euclid=torch.nn.Parameter(torch.randn(sum(counts),32,device=device)*.03) if method=='E' else None
    def representations(self):
        if self.euclid is not None:
            if self.model!='lightgcn':return self.euclid
            sp=self.euclid;embs=[sp]
            for _ in range(2):sp=torch.sparse.mm(self.edges,sp);embs.append(sp)
            return torch.stack(embs).mean(0)
        ps=self.geo.chart_points()
        if self.model!='lightgcn':return ps
        # Parameter-free graph propagation in the canonical origin tangent chart.
        out=[];ks=self.geo.K()
        for g,p in enumerate(ps):
            k=ks[g];o=L.origin(8,k,device=p.device).expand_as(p);v=L.log_map(o,p,k);sp=v[:,1:];embs=[sp]
            for _ in range(2):sp=torch.sparse.mm(self.edges,sp);embs.append(sp)
            mean=torch.stack(embs).mean(0);out.append(L.exp_map(o,torch.cat([torch.zeros_like(mean[:,:1]),mean],-1),k))
        return out
    def score(self,a,b,ps=None):
        if self.euclid is not None:
            representation=self.representations() if ps is None else ps
            return -(representation[a]-representation[b]).square().sum(-1)
        ps=self.representations() if ps is None else ps;ks=self.geo.K()
        return -sum(L.dist(p[a],p[b],ks[g]).square() for g,p in enumerate(ps))

@torch.no_grad()
def rec_eval(model,pairs,seen,nuser,nitem,device):
    ps=model.representations();rec=[];nd=[]
    allitems=torch.arange(nitem,device=device)+nuser
    # Macro-average per user; supports both leave-one-out and multi-positive tests.
    targets={}
    for uid,target in pairs:targets.setdefault(uid,set()).add(target)
    users=list(targets)
    for start in range(0,len(users),64):
        batch=users[start:start+64];u=torch.tensor(batch,device=device)
        scores=model.score(u[:,None],allitems[None,:],ps)
        for j,uid in enumerate(batch):
            mask=list(seen[uid]-targets[uid]);scores[j,mask]=-torch.inf
        top=scores.topk(min(20,nitem),dim=-1).indices.cpu().tolist()
        for uid,ranked in zip(batch,top):
            relevant=targets[uid];hits=[rank for rank,item in enumerate(ranked) if item in relevant]
            rec.append(len(hits)/len(relevant))
            dcg=sum(1/math.log2(rank+2) for rank in hits)
            ideal=sum(1/math.log2(rank+2) for rank in range(min(20,len(relevant))))
            nd.append(dcg/ideal)
    return {'recall20':float(np.mean(rec)),'ndcg20':float(np.mean(nd))}

def recommend(cfg,root,device,out):
    nu,ni,tr,va,te,seen=split_recommendation(root,cfg['dataset']);train=torch.tensor(tr,device=device);edges=None
    if cfg['dataset']=='yelp' and (root/'yelp2018/provenance.json').exists():
        (out/'data_provenance.json').write_text((root/'yelp2018/provenance.json').read_text())
    if cfg['model']=='lightgcn':
        a=train[:,0];b=train[:,1]+nu;idx=torch.stack([torch.cat([a,b]),torch.cat([b,a])]);deg=torch.bincount(idx[0],minlength=nu+ni).float().clamp_min(1)
        vals=(deg[idx[0]]*deg[idx[1]]).rsqrt();edges=torch.sparse_coo_tensor(idx,vals,(nu+ni,nu+ni)).coalesce()
    m=ProductModel([nu,ni],cfg['method'],device,cfg['model'],edges,lr=cfg.get('lr',.1));opt=torch.optim.AdamW([m.euclid],lr=.001) if m.euclid is not None else None
    # Known positives are excluded from negative draws (train only, never target labels).
    seenkeys=torch.tensor(sorted(u*ni+i for u,i in tr),device=device)
    start=time.time();best=-1
    for epoch in range(cfg['epochs']):
        perm=torch.randperm(len(train),device=device);total=0.
        for ids in perm.split(cfg.get('batch_size',2048)):
            u=train[ids,0];pos=train[ids,1];neg=torch.randint(ni,u.shape,device=device)
            bad=sorted_contains(seenkeys,u*ni+neg)
            while bad.any():neg[bad]=torch.randint(ni,(int(bad.sum()),),device=device);bad=sorted_contains(seenkeys,u*ni+neg)
            m.geo.zero_grad()
            if opt:opt.zero_grad()
            ps=m.representations();loss=F.softplus(m.score(u,neg+nu,ps)-m.score(u,pos+nu,ps)).mean()
            if not torch.isfinite(loss):raise FloatingPointError(f'nonfinite epoch {epoch}')
            loss.backward()
            if opt:opt.step()
            else:m.geo.step()
            total+=float(loss)*len(ids)
        row={'epoch':epoch,'train_loss':total/len(train),'seconds':time.time()-start,'K':m.geo.K().detach().cpu().tolist(),**m.geo.last}
        if epoch%25==0 or epoch==cfg['epochs']-1:
            met=rec_eval(m,va,seen,nu,ni,device);row.update({'val_'+k:v for k,v in met.items()});print(json.dumps(row),flush=True)
            if met['ndcg20']>best:best=met['ndcg20'];torch.save(m.state_dict(),out/'best.pt')
        if epoch==cfg['epochs']-1:
            testseen={u:seen[u]|{i} for u,i in va};row.update({'test_'+k:v for k,v in rec_eval(m,te,testseen,nu,ni,device).items()})
        with (out/'trajectory.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
    (out/'result.json').write_text(json.dumps({'status':'complete','config':cfg,'final':row,'epochs_completed':cfg['epochs']},indent=2));torch.save(m.state_dict(),out/'last.pt')

def synthetic(cfg,root,device,out):
    from train_formal import gen_tree
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import shortest_path
    n=cfg['nodes'];edges,par,n=gen_tree(cfg['dataset'],n)
    a=np.array([i for i,j in edges]+[j for i,j in edges]);b=np.array([j for i,j in edges]+[i for i,j in edges]);D=shortest_path(csr_matrix((np.ones(len(a)),(a,b)),shape=(n,n)),unweighted=True)
    D=torch.tensor(D,dtype=torch.float32,device=device)
    # Product factors partition the requested total intrinsic dimension.
    # Factors share tree observations, not parameter states; this differs from the historical independent-tree groups.
    groups=min(4,cfg['dim']);dims=[cfg['dim']//groups+(g<cfg['dim']%groups) for g in range(groups)]
    m=Geometry([(n,dd) for dd in dims],cfg['method'],device,lr=cfg.get('lr',10.),rho=cfg.get('rho',.001))
    ep=torch.nn.Parameter(torch.randn(n,cfg['dim'],device=device)*.03) if cfg['method']=='E' else None
    opt=torch.optim.SGD([ep],lr=10.,momentum=.9) if ep is not None else None
    evgen=torch.Generator(device=device).manual_seed(1042);ei=torch.randint(n,(min(100000,n*n),),device=device,generator=evgen);ej=torch.randint(n,ei.shape,device=device,generator=evgen);keep=ei!=ej;ei,ej=ei[keep],ej[keep]
    def distances(i,j):
        if ep is not None:return (ep[i]-ep[j]).norm(dim=-1)
        ps=m.chart_points();k=m.K();return torch.sqrt(sum(L.dist(p[i],p[j],k[g]).square() for g,p in enumerate(ps)).clamp_min(1e-12))
    start=time.time()
    for epoch in range(cfg['epochs']):
        m.zero_grad()
        if opt:opt.zero_grad()
        i=torch.randint(n,(65536,),device=device);j=torch.randint(n,i.shape,device=device);keep=i!=j;i,j=i[keep],j[keep]
        loss=(distances(i,j)/D[i,j]-1).square().mean()
        if not torch.isfinite(loss):raise FloatingPointError('nonfinite synthetic loss')
        loss.backward()
        if opt:opt.step()
        else:m.step()
        with torch.no_grad():dist=float((distances(ei,ej)/D[ei,ej]-1).abs().mean())
        row={'epoch':epoch,'train_loss':float(loss),'distortion_fixed_pairs':dist,'seconds':time.time()-start,'K':m.K().detach().cpu().tolist(),**m.last}
        if epoch%100==0:print(json.dumps(row),flush=True)
        with (out/'trajectory.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
    # Adjacency reconstruction MAP and mean rank, all candidates for every node.
    with torch.no_grad():
        adj=[set() for _ in range(n)]
        for a,b in edges:adj[a].add(b);adj[b].add(a)
        maps=[];ranks=[]
        for a in range(n):
            cand=torch.arange(n,device=device);ds=distances(torch.full_like(cand,a),cand);ds[a]=torch.inf;order=ds.argsort().cpu().tolist();rr=sorted(order.index(b)+1 for b in adj[a]);maps.append(sum((j+1)/r for j,r in enumerate(rr))/len(rr));ranks.extend(rr)
        row.update(MAP=float(np.mean(maps)),mean_rank=float(np.mean(ranks)))
    (out/'result.json').write_text(json.dumps({'status':'complete','config':cfg,'final':row,'epochs_completed':cfg['epochs']},indent=2));torch.save(m.state_dict(),out/'last.pt')

class KG(torch.nn.Module):
    def __init__(self,ne,nr,method,device,kind='lorentz_translation'):
        super().__init__();self.ne=ne;self.nr=nr;self.kind=kind
        self.geo=Geometry([(ne+nr,16)]*4,method,device,lr=.1,rho=.001)
        self.e=torch.nn.Parameter(torch.randn(ne+nr,64,device=device)*.03) if method=='E' else None
    def score(self,h,r,t,ps=None):
        if self.e is not None:
            if self.kind=='lorentz_distance':
                # Match the four-factor relation-weighted distance architecture.
                hparts=self.e[h].reshape(*h.shape,4,16);tparts=self.e[t].reshape(*t.shape,4,16)
                rparts=self.e[self.ne+r].reshape(*r.shape,4,16)
                return -((hparts-tparts).square().sum(-1)*(1+rparts.square().sum(-1))).sum(-1)
            return -(self.e[h]+self.e[self.ne+r]-self.e[t]).square().sum(-1)
        ps=self.geo.chart_points() if ps is None else ps;ks=self.geo.K();scores=[]
        for g,p in enumerate(ps):
            k=ks[g];hh=p[h];tt=p[t];rv=L.log_map(L.origin(16,k,device=p.device).expand_as(p[self.ne+r]),p[self.ne+r],k)
            rv=L.parallel_transport(L.origin(16,k,device=p.device).expand_as(hh),hh,rv,k);ht=L.exp_map(hh,rv,k)
            scores.append(-L.dist(hh if self.kind=='lorentz_distance' else ht,tt,k).square()*(1+L.lorentz_norm(rv).square()) if self.kind=='lorentz_distance' else -L.dist(ht,tt,k).square())
        return sum(scores)

@torch.no_grad()
def kg_eval(m,trip,known,ne,device):
    ranks=[]
    ps=m.geo.chart_points()
    for h,r,t in trip:
        for side in [0,2]:
            scores=[]
            for candidates in torch.arange(ne,device=device).split(4096):
                hh=candidates if side==0 else torch.full_like(candidates,h);tt=candidates if side==2 else torch.full_like(candidates,t);rr=torch.full_like(candidates,r);scores.append(m.score(hh,rr,tt,ps))
            s=torch.cat(scores);target=h if side==0 else t;val=s[target].clone()
            filt=known[side].get((r,t) if side==0 else (h,r),set())-{target}
            if filt:s[list(filt)]=-torch.inf
            # Average rank for tied scores, explicit in protocol.
            rank=1+(s>val).sum().item()+.5*((s==val).sum().item()-1);ranks.append(rank)
    r=np.array(ranks);return {'MRR':float(np.mean(1/r)),'Hits1':float(np.mean(r<=1)),'Hits3':float(np.mean(r<=3)),'Hits10':float(np.mean(r<=10))}

def knowledge(cfg,root,device,out):
    ne,nr,s=triples(root,cfg['dataset']);train=torch.tensor(s['train'],device=device);known={0:{},2:{}}
    for split in s.values():
        for h,r,t in split:known[0].setdefault((r,t),set()).add(h);known[2].setdefault((h,r),set()).add(t)
    m=KG(ne,nr,cfg['method'],device,cfg['model']);opt=torch.optim.AdamW([m.e],lr=.001) if m.e is not None else None;start=time.time();best=-1
    for epoch in range(cfg['epochs']):
        total=0.
        for ids in torch.randperm(len(train),device=device).split(1024):
            batch=train[ids];h,r,t=batch.T;m.geo.zero_grad()
            if opt:opt.zero_grad()
            nt=torch.randint(ne,(len(ids),16),device=device);nh=torch.randint(ne,nt.shape,device=device)
            pos=m.score(h,r,t);neg=torch.cat([m.score(h[:,None],r[:,None],nt),m.score(nh,r[:,None],t[:,None])],-1)
            loss=F.cross_entropy(torch.cat([pos[:,None],neg],-1),torch.zeros(len(ids),dtype=torch.long,device=device))
            if not torch.isfinite(loss):raise FloatingPointError('nonfinite KG loss')
            loss.backward()
            if opt:opt.step()
            else:m.geo.step()
            total+=float(loss)*len(ids)
        row={'epoch':epoch,'train_loss':total/len(train),'seconds':time.time()-start,'K':m.geo.K().detach().cpu().tolist(),**m.geo.last}
        # Full validation/test, not changing random subsets. Cost is intentionally explicit.
        if (epoch+1)%100==0 or epoch==cfg['epochs']-1:
            met=kg_eval(m,s['valid'],known,ne,device);row.update({'val_'+k:v for k,v in met.items()})
            if met['MRR']>best:best=met['MRR'];torch.save(m.state_dict(),out/'best.pt')
        if epoch==cfg['epochs']-1:row.update({'test_'+k:v for k,v in kg_eval(m,s['test'],known,ne,device).items()})
        if epoch%25==0:print(json.dumps(row),flush=True)
        with (out/'trajectory.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
    (out/'result.json').write_text(json.dumps({'status':'complete','config':cfg,'final':row,'epochs_completed':cfg['epochs']},indent=2));torch.save(m.state_dict(),out/'last.pt')

def run(cfg,root,device,out):
    {'synthetic':synthetic,'kg':knowledge,'recommendation':recommend}[cfg['family']](cfg,root,device,out)
