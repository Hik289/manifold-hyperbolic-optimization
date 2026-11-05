"""Expand every dataset/model/scale explicitly named in proposal section 6."""
import itertools,json
from pathlib import Path
METHODS=['E','B2','B3','B4','B5','A5']
def generate():
    tasks=[]
    def add(family,datasets,models,epochs,**extra):
        for ds,model,method in itertools.product(datasets,models,METHODS):
            tasks.append(dict(family=family,dataset=ds,model=model,method=method,epochs=epochs,seed=42,**extra))
    add('graph',['cora','citeseer','pubmed','arxiv'],['gcn','gat','sage'],1000)
    add('text',['ag_news','dbpedia','20news','reuters'],['mlp','transformer'],1000)
    add('vision',['cifar10','cifar100','tinyimagenet','imagenet100'],['resnet18','resnet50','vit_s16'],1000)
    add('recommendation',['ml1m','amazon_books','yelp'],['mf','lightgcn'],1000)
    add('continual',['split_cifar100','rotmnist'],['mlp','resnet18'],1000)
    add('kg',['wn18rr','fb15k237','yago310'],['lorentz_distance','lorentz_translation'],2000)
    for n,d in itertools.product([1000,5000,10000],[2,5,10,20]):
        add('synthetic',['binary','4ary','8ary','variable'],['product_lorentz'],2000,nodes=n,dim=d,lr=10.)
    # Mechanism ablations on a representative configuration in each family.
    representatives=[('graph','cora','gcn',1000),('text','20news','mlp',1000),('vision','cifar100','resnet18',1000),('recommendation','ml1m','mf',1000),('continual','rotmnist','mlp',1000),('kg','wn18rr','lorentz_translation',2000)]
    for f,d,m,e in representatives:
        for method in ['B35','shared']:tasks.append(dict(family=f,dataset=d,model=m,method=method,epochs=e,seed=42))
    # Full A1--A12 mechanism checks on Cora/GCN. A3 and A11 coincide
    # for a class-prototype head with a shared scalar curvature and remain labeled.
    ablations=[('A1',{'beta':0.}),('A2',{'rho':0.}),('A3',{'method':'shared'}),('A4',{'gamma':0.}),('A5',{'method':'A5'}),('A6',{'project_tangent':False}),('A7',{'rho':.1}),('A8',{'log_clip':None}),('A9',{'curvature_every':10}),('A10',{'curvature_every':100}),('A11',{'method':'shared'})]
    for label,opts in ablations:
        tasks.append(dict(family='graph',dataset='cora',model='gcn',method='B5',epochs=1000,seed=42,ablation=label,**{k:v for k,v in opts.items() if k!='method'}))
        if 'method' in opts:tasks[-1]['method']=opts['method']
    for initial in [.01,.1,.1001,.5,2.,5.]:tasks.append(dict(family='graph',dataset='cora',model='gcn',method='B5',epochs=1000,seed=42,ablation='A12_K'+str(initial),init_K=initial))
    # Predeclared long-horizon mechanism matrix. Fixed-K normalization controls
    # get a five-value K sweep, selected using validation only, before test comparison.
    mechanism_cases=[('static','static',1.,[2,4,8,16]),
                     ('smooth2','smooth',2.,[2,4,8,16]),
                     ('smooth5','smooth',5.,[2,4,8,16]),
                     ('smooth8','smooth',8.,[2,4,8,16]),
                     ('abrupt8','abrupt',8.,[2,4,8,16]),
                     ('heterogeneous','static',1.,[2,8,32,128])]
    for priority,(case,schedule,scale,branchings) in enumerate(mechanism_cases):
        base=dict(family='mechanism',dataset=case,model='independent_lorentz',
                  epochs=2000,seed=42,nodes_per_group=250,dim=3,
                  branchings=branchings,schedule=schedule,scale_end=scale,
                  lr=100.,rho=.01,beta=.98,priority=priority)
        for method in ['B3','B35','B4','B5','A5','shared']:
            tasks.append(dict(base,method=method,priority=priority if method in ['B3','B5','A5'] else 100+priority))
        for initial in [.03,.1,.3,3.]:
            tasks.append(dict(base,method='B35',init_K=initial,ablation='fixed_K'+str(initial),priority=200+priority))
            for method in ['B5','A5']:
                tasks.append(dict(base,method=method,init_K=initial,ablation='initial_K'+str(initial),priority=200+priority))
    # Finish small representative graph comparisons first, then cover one config per family.
    def priority(t):
        first=t['dataset']=='cora' and t['model']=='gcn' and t['method'] in ['B3','B5','B4','A5','E','B2']
        representative=(t['family'],t['dataset'],t['model']) in [(x[0],x[1],x[2]) for x in representatives]
        return (0 if first else 1 if representative else 2,t['family'],t['dataset'],t['model'],METHODS.index(t['method']) if t['method'] in METHODS else 6)
    tasks.sort(key=priority)
    for t in tasks:
        t['id']='_'.join(str(t.get(k,'')) for k in ['family','dataset','model','nodes','dim','method']).replace('__','_').replace('__','_')+(('_'+t['ablation']) if 'ablation' in t else '')
    from extension_manifest import generate as extension_configs
    tasks.extend(extension_configs())
    return tasks
if __name__=='__main__':
    tasks=generate();Path(__file__).with_name('manifest.json').write_text(json.dumps(tasks,indent=2));print(len(tasks),'registered full runs')
