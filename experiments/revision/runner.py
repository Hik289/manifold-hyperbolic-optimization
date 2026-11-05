"""Full-data, seed-42 benchmark execution. No automatic outcome-based task filtering."""
import argparse,csv,json,math,os,random,time,hashlib
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from geometry import Head,Geometry,origin_embed,L

class MLP(nn.Module):
    def __init__(self,inp,d=32):
        super().__init__(); self.net=nn.Sequential(nn.Flatten(),nn.Linear(inp,256),nn.ReLU(),nn.Linear(256,d))
    def forward(self,x):return self.net(x)

class TextNet(nn.Module):
    def __init__(self,kind,vocab=30000,d=32):
        super().__init__(); self.emb=nn.Embedding(vocab,128,padding_idx=0); self.kind=kind
        if kind=='transformer':
            self.pos=nn.Parameter(torch.randn(256,128)*.01)
            self.encoder=nn.TransformerEncoder(nn.TransformerEncoderLayer(128,4,256,batch_first=True,dropout=0.),2)
        self.proj=nn.Linear(128,d)
    def forward(self,x):
        z=self.emb(x); mask=x==0
        if self.kind=='transformer':z=self.encoder(z+self.pos[:x.shape[1]],src_key_padding_mask=mask)
        return self.proj((z*(~mask).unsqueeze(-1)).sum(1)/(~mask).sum(1,keepdim=True).clamp_min(1))

class GraphNet(nn.Module):
    def __init__(self,inp,kind,d=32):
        super().__init__(); from torch_geometric.nn import GCNConv,GATConv,SAGEConv
        cls={'gcn':GCNConv,'gat':GATConv,'sage':SAGEConv}[kind]
        self.a=cls(inp,64);self.b=cls(64,d)
    def forward(self,x,edge):return self.b(F.relu(self.a(x,edge)),edge)

def hash_tokens(text):
    # Stable vocabulary-free tokenizer, no test corpus fitting.
    import re
    ts=re.findall(r'\w+',text.lower())[:256]
    return [1+int.from_bytes(hashlib.blake2b(s.encode(),digest_size=4).digest(),'little')%29999 for s in ts] or [1]

def text_data(name,root):
    from sklearn.model_selection import train_test_split
    if name in ['ag_news','dbpedia']:
        from datasets import load_dataset
        ds=load_dataset('fancyzhx/ag_news' if name=='ag_news' else 'fancyzhx/dbpedia_14',cache_dir=str(root/'hf'))
        key='text' if name=='ag_news' else 'content'
        tr=list(ds['train'][key]);y=list(ds['train']['label']);te=list(ds['test'][key]);yt=list(ds['test']['label'])
    elif name=='20news':
        from sklearn.datasets import fetch_20newsgroups
        a=fetch_20newsgroups(subset='train',data_home=str(root),remove=('headers','footers','quotes'));b=fetch_20newsgroups(subset='test',data_home=str(root),remove=('headers','footers','quotes'))
        tr,y,te,yt=a.data,a.target,b.data,b.target
    else:
        import nltk
        nltk.download('reuters',download_dir=str(root/'nltk'),quiet=True); nltk.data.path.insert(0,str(root/'nltk'))
        from nltk.corpus import reuters
        cats=sorted(reuters.categories()); ci={c:i for i,c in enumerate(cats)}
        tr=[];y=[];te=[];yt=[]
        for fid in reuters.fileids():
            # Reuters is multi-label; retain all labels and use BCE/micro+macro F1.
            label=np.zeros(len(cats),dtype=np.float32)
            for c in reuters.categories(fid):label[ci[c]]=1
            if fid.startswith('training/'):tr.append(reuters.raw(fid));y.append(label)
            else:te.append(reuters.raw(fid));yt.append(label)
    def encode(ts):
        a=torch.zeros(len(ts),256,dtype=torch.long)
        for i,t in enumerate(ts):
            tok=hash_tokens(t);a[i,:len(tok)]=torch.tensor(tok)
        return a
    x=encode(tr);xt=encode(te);y=torch.tensor(np.array(y));yt=torch.tensor(np.array(yt))
    ids=np.arange(len(y));tri,vi=train_test_split(ids,test_size=.1,random_state=42,stratify=y.numpy() if y.ndim==1 else None)
    return torch.utils.data.TensorDataset(x[tri],y[tri]),torch.utils.data.TensorDataset(x[vi],y[vi]),torch.utils.data.TensorDataset(xt,yt),int(y.max()+1) if y.ndim==1 else y.shape[1]

def vision_data(name,root):
    from torchvision import datasets,transforms
    size=64 if name=='tinyimagenet' else 224 if name=='imagenet100' else 32
    tf=transforms.Compose([transforms.Resize((size,size)),transforms.ToTensor(),transforms.Normalize((.5,)*3,(.5,)*3)])
    if name=='imagenet100' and (root/name/'ready.json').exists():
        from manifest_images import load_manifest_images
        return load_manifest_images(root/name,tf)
    if name in ['cifar10','cifar100','split_cifar100']:
        cls=datasets.CIFAR10 if name=='cifar10' else datasets.CIFAR100
        tr=cls(str(root),train=True,download=True,transform=tf);te=cls(str(root),train=False,download=True,transform=tf)
    elif name=='rotmnist':
        tf=transforms.Compose([transforms.Resize((32,32)),transforms.Grayscale(3),transforms.ToTensor()])
        tr=datasets.MNIST(str(root),train=True,download=True,transform=tf);te=datasets.MNIST(str(root),train=False,download=True,transform=tf)
    else:
        p=root/name
        if not (p/'train').is_dir():raise FileNotFoundError(f'{p}/train: ImageFolder data required; dataset acquisition is tracked separately')
        tr=datasets.ImageFolder(str(p/'train'),tf);te=datasets.ImageFolder(str(p/'val'),tf)
        if name=='imagenet100' and len(tr.classes)!=100:raise ValueError('ImageNet-100 must have exactly 100 classes')
    gen=torch.Generator().manual_seed(42);n=len(tr);ids=torch.randperm(n,generator=gen).tolist();cut=int(.9*n)
    return torch.utils.data.Subset(tr,ids[:cut]),torch.utils.data.Subset(tr,ids[cut:]),te,len(tr.classes)

def backbone(kind,inp=3072,size=32):
    if kind=='mlp':return MLP(3*size*size)
    if kind in ['resnet18','resnet50']:
        from torchvision import models
        net=getattr(models,kind)(weights=None);net.conv1=nn.Conv2d(3,64,3,1,1,bias=False);net.maxpool=nn.Identity();net.fc=nn.Linear(net.fc.in_features,32);return net
    import timm
    return timm.create_model('vit_small_patch16_224',pretrained=False,num_classes=32,img_size=size)

def metrics(logits,y):
    from sklearn.metrics import f1_score
    if y.ndim==2:
        p=logits.sigmoid().cpu().numpy()>=.5;t=y.cpu().numpy()
        return {'accuracy':float((p==t).all(1).mean()),'macro_f1':float(f1_score(t,p,average='macro',zero_division=0)),'micro_f1':float(f1_score(t,p,average='micro',zero_division=0))}
    p=logits.argmax(-1);return {'accuracy':float((p==y).float().mean()),'macro_f1':float(f1_score(y.cpu(),p.cpu(),average='macro',zero_division=0))}

def lossfn(logits,y):return F.binary_cross_entropy_with_logits(logits,y.float()) if y.ndim==2 else F.cross_entropy(logits,y.long())

def classification(cfg,root,device,out):
    family=cfg['family'];name=cfg['dataset'];kind=cfg['model'];method=cfg['method']
    graph=family=='graph';cl=family=='continual'
    if graph:
        if name=='arxiv':
            from ogb.nodeproppred import PygNodePropPredDataset
            ds=PygNodePropPredDataset(name='ogbn-arxiv',root=str(root));data=ds[0];sp=ds.get_idx_split(); masks=[sp['train'],sp['valid'],sp['test']];data.y=data.y.flatten()
        else:
            from torch_geometric.datasets import Planetoid
            ds=Planetoid(str(root/name),{'cora':'Cora','citeseer':'CiteSeer','pubmed':'PubMed'}[name]);data=ds[0];masks=[data.train_mask,data.val_mask,data.test_mask]
        from torch_geometric.utils import to_undirected
        data.edge_index=to_undirected(data.edge_index);data=data.to(device);masks=[m.to(device) for m in masks]
        net=GraphNet(data.num_node_features,kind).to(device);classes=int(data.y.max()+1);tr=va=te=None
    else:
        tr,va,te,classes=text_data(name,root) if family=='text' else vision_data(name,root)
        if name=='imagenet100' and (root/name/'ready.json').exists():
            (out/'data_provenance.json').write_text((root/name/'ready.json').read_text())
        net=(TextNet(kind) if family=='text' else backbone(kind,size=64 if name=='tinyimagenet' else 224 if name=='imagenet100' else 32)).to(device)
    head=Head(32,classes,method,device,lr=cfg.get('lr',.1),rho=cfg.get('rho',.001),beta=cfg.get('beta',.9),gamma=cfg.get('gamma',1.),init_K=cfg.get('init_K',1.),project_tangent=cfg.get('project_tangent',True),log_clip=cfg.get('log_clip',.01),curvature_every=cfg.get('curvature_every',1)).to(device)
    params=list(net.parameters())+([] if method!='E' else list(head.euclidean.parameters()))
    opt=torch.optim.AdamW(params,lr=.001) if family in ['graph','text'] else torch.optim.SGD(params,lr=.01,momentum=.9,weight_decay=.0005)
    loaders=None if graph else [torch.utils.data.DataLoader(d,batch_size=cfg.get('batch_size',128),shuffle=i==0,num_workers=2,pin_memory=True,generator=torch.Generator().manual_seed(42)) for i,d in enumerate([tr,va,te])]
    history=[];best=-1;start=time.time();taskbest=[0.]*5
    def transform(x,task):
        if not cl:return x
        if name=='rotmnist':
            from torchvision.transforms.functional import rotate
            return rotate(x,float(task*36))
        return x
    def restrict(loader,task):
        for x,y in loader:
            if cl and name=='split_cifar100':
                keep=(y//20==task);x=x[keep];y=y[keep]
                if not len(y):continue
            yield transform(x.to(device),task),y.to(device)
    def evaluate(loader,task):
        logits=[];ys=[]
        with torch.no_grad():
            if graph:return metrics(head(net(data.x,data.edge_index))[masks[task]],data.y[masks[task]])
            for x,y in restrict(loader,task):logits.append(head(net(x)));ys.append(y)
        return metrics(torch.cat(logits),torch.cat(ys))
    for epoch in range(cfg['epochs']):
        head.geo.epoch=epoch
        task=min(4,epoch//max(1,cfg['epochs']//5)) if cl else 0
        net.train();head.train();total=0.;count=0
        batches=[(None,None)] if graph else restrict(loaders[0],task)
        for x,y in batches:
            opt.zero_grad();head.geo.zero_grad()
            logits=head(net(data.x,data.edge_index))[masks[0]] if graph else head(net(x));target=data.y[masks[0]] if graph else y
            loss=lossfn(logits,target)
            if method in ['B4','B5','A5','shared']:loss=loss+.0001*head.geo.K().log().square().sum()
            if not torch.isfinite(loss):raise FloatingPointError(f'nonfinite loss epoch {epoch}')
            loss.backward();torch.nn.utils.clip_grad_norm_(params,10.);opt.step()
            if method!='E':head.geo.step()
            total+=float(loss)*len(target);count+=len(target)
        net.eval();head.eval()
        row={'epoch':epoch,'task':task,'train_loss':total/max(count,1),'seconds':time.time()-start,'K':head.geo.K().detach().cpu().tolist(),'learning_rate':head.geo.lr,'curvature_learning_rate':head.geo.rho,'peak_memory_mb':torch.cuda.max_memory_allocated()/1e6,**head.geo.last}
        # Validation is logged each epoch; test is evaluated only at registered checkpoints.
        v=evaluate(None if graph else loaders[1],1 if graph else task);row.update({'val_'+k:v for k,v in v.items()})
        score=v.get('micro_f1',v['accuracy'])
        if score>best:best=score;torch.save({'net':net.state_dict(),'head':head.state_dict(),'epoch':epoch},out/'best.pt')
        if epoch==cfg['epochs']-1 or (cl and (epoch+1)%max(1,cfg['epochs']//5)==0):
            t=evaluate(None if graph else loaders[2],2 if graph else task);row.update({'test_'+k:v for k,v in t.items()})
            if cl:
                accs=[evaluate(loaders[2],j)['accuracy'] for j in range(task+1)]
                for j,a in enumerate(accs):taskbest[j]=max(taskbest[j],a)
                row.update(task_accuracies=accs,average_accuracy=float(np.mean(accs)),forgetting=float(np.mean([taskbest[j]-accs[j] for j in range(task)])) if task else 0.)
                # Recovery: epochs since task boundary to 90% of preceding task's best validation.
        with open(out/'trajectory.jsonl','a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
        if epoch%25==0:print(json.dumps(row),flush=True)
        history.append(row)
        if (epoch+1)%100==0 or epoch==cfg['epochs']-1:torch.save({'net':net.state_dict(),'head':head.state_dict(),'epoch':epoch,'config':cfg},out/'last.pt')
    selected={}
    if not cl:
        ckpt=torch.load(out/'best.pt',map_location=device,weights_only=False);net.load_state_dict(ckpt['net']);head.load_state_dict(ckpt['head']);net.eval();head.eval()
        selected={'selected_epoch':ckpt['epoch'],'selected_test':evaluate(None if graph else loaders[2],2 if graph else 0)}
    result={**selected,'status':'complete','config':cfg,'final':history[-1],'best_validation':best,'epochs_completed':len(history)}
    (out/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False));return result

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',required=True);ap.add_argument('--root',required=True);ap.add_argument('--out',required=True);ap.add_argument('--device',choices=['auto','cpu','cuda'],default='auto');a=ap.parse_args()
    cfg=json.loads(Path(a.config).read_text());out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    import hashlib
    (out/'source_hashes.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')},indent=2))
    torch.manual_seed(42);np.random.seed(42);random.seed(42);torch.set_num_threads(4)
    torch.backends.cudnn.benchmark=False
    if a.device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA requested but unavailable')
    device=torch.device('cuda:0' if a.device=='cuda' or (a.device=='auto' and torch.cuda.is_available()) else 'cpu');root=Path(a.root);root.mkdir(parents=True,exist_ok=True)
    if cfg.get('suite')=='new_tasks_v1':
        from new_tasks import run
        run(cfg,root,device,out)
    elif cfg['family']=='mechanism':
        from mechanism import run
        run(cfg,root,device,out)
    elif cfg['family'] in ['vision','graph','text','continual']:classification(cfg,root,device,out)
    else:
        from structured import run
        run(cfg,root,device,out)
if __name__=='__main__':main()
