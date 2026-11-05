"""Small fixture tests of executable task paths; never reported as benchmark results."""
import tempfile,json,torch,numpy as np,random
from pathlib import Path
from geometry import Head
from runner import MLP,TextNet,GraphNet
from structured import synthetic,knowledge,recommend

def main():
    torch.manual_seed(42);np.random.seed(42);random.seed(42);torch.set_num_threads(2);device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu');root=Path(tempfile.mkdtemp(prefix='manifold_smoke_'))
    passed=[]
    for name,net,x in [('vision',MLP(3*32*32),torch.randn(4,3,32,32)),('text',TextNet('transformer'),torch.randint(1,30000,(4,12)))]:
        net=net.to(device);head=Head(32,3,'B5',device);loss=torch.nn.functional.cross_entropy(head(net(x.to(device))),torch.tensor([0,1,2,0],device=device));loss.backward();head.geo.step();passed.append(name)
    net=GraphNet(5,'gcn').to(device);head=Head(32,3,'B5',device);x=torch.randn(4,5,device=device);edge=torch.tensor([[0,1,2,3],[1,2,3,0]],device=device);loss=head(net(x,edge)).square().mean();loss.backward();head.geo.step();passed.append('graph')
    out=root/'synthetic';out.mkdir();synthetic(dict(family='synthetic',dataset='binary',model='product_lorentz',method='B5',epochs=2,nodes=32,dim=3),root,device,out);passed.append('synthetic')
    p=root/'toykg';p.mkdir()
    for split,rows in [('train','a\tr\tb\nb\tr\tc\nc\tr\td\n'),('valid','a\tr\tc\n'),('test','b\tr\td\n')]: (p/(split+'.txt')).write_text(rows)
    out=root/'kg';out.mkdir();knowledge(dict(family='kg',dataset='toykg',model='lorentz_translation',method='B5',epochs=2),root,device,out);passed.append('kg')
    with (root/'toyrec.jsonl').open('w') as f:
        for u in range(3):
            for j in range(7):f.write(json.dumps(dict(user=u,item=(u*3+j)%12,timestamp=j))+'\n')
    for model in ['mf','lightgcn']:
        out=root/model;out.mkdir();recommend(dict(family='recommendation',dataset='toyrec',model=model,method='B5',epochs=2,batch_size=16),root,device,out);passed.append('recommendation_'+model)
    print(json.dumps({'status':'passed','paths':passed,'fixture_only':True,'directory':str(root)},indent=2))
if __name__=='__main__':main()
