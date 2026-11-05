"""Versioned real-data splits. No test outcome participates in task construction."""
import fcntl,hashlib,json
from pathlib import Path
import numpy as np


def edge_split(n,edges,directed,seed=42):
    """Disjoint positive and negative splits, including reverse-edge protection."""
    rng=np.random.default_rng(seed)
    pos=sorted({(int(a),int(b)) if directed else tuple(sorted((int(a),int(b))))
                for a,b in edges if a!=b})
    ordered=[pos[i] for i in rng.permutation(len(pos))]
    # Keep at least one observed relation incident to every nonisolated vertex.
    cover=[];rest=[];seen=set()
    for a,b in ordered:
        if a not in seen or b not in seen:cover.append((a,b));seen.update([a,b])
        else:rest.append((a,b))
    nval=ntest=min(int(.1*len(pos)),len(rest)//3)
    valid=rest[:nval];test=rest[nval:nval+ntest]
    train=cover+rest[nval+ntest:]
    assert nval>0 and ntest>0
    known=set(pos);negative=set();need=len(pos)
    assert need <= (n*(n-1) if directed else n*(n-1)//2)-len(pos)
    while len(negative)<need:
        for a,b in rng.integers(n,size=(max(1000,need-len(negative)),2)):
            pair=(int(a),int(b)) if directed else tuple(sorted((int(a),int(b))))
            if a!=b and pair not in known:negative.add(pair)
            if len(negative)==need:break
    neg=np.array(sorted(negative),dtype=np.int64);neg=neg[rng.permutation(len(neg))]
    out={'n':np.array(n),'all_positive':np.array(pos,dtype=np.int64)}
    offset=0
    for key,positive in [('train',train),('validation',valid),('test',test)]:
        out[key+'_positive']=np.array(positive,dtype=np.int64)
        out[key+'_negative']=neg[offset:offset+len(positive)];offset+=len(positive)
    assert_split(out,directed)
    return out


def assert_split(data,directed):
    sets={}
    for kind in ['positive','negative']:
        for split in ['train','validation','test']:
            arr=data[split+'_'+kind]
            pairs={tuple(map(int,row)) for row in arr}
            assert len(pairs)==len(arr) and all(a!=b for a,b in pairs)
            if not directed:assert all(a<b for a,b in pairs)
            sets[split+'_'+kind]=pairs
    keys=list(sets)
    for i,a in enumerate(keys):
        for b in keys[i+1:]:assert not sets[a]&sets[b],(a,b)
    covered=set(data['train_positive'].flatten().tolist())
    assert set(data['all_positive'].flatten().tolist())<=covered


def graph_data(name,root):
    from torch_geometric.datasets import Planetoid
    base='cora' if name.startswith('cora') else 'citeseer'
    graph=Planetoid(str(root/base),{'cora':'Cora','citeseer':'CiteSeer'}[base])[0]
    data=edge_split(graph.num_nodes,graph.edge_index.T.numpy(),False)
    meta=dict(source='PyG Planetoid '+base,source_url='https://github.com/kimiyoung/planetoid',
        directed=False,features_used=False,protocol='Transductive held-out-edge prediction; no class labels or node features used. Canonical undirected pairs and negatives are disjoint across splits; all known positive pairs are excluded from negative draws.')
    return data,meta


def taxonomy_data(name,root):
    import nltk
    dest=root/'nltk';dest.mkdir(exist_ok=True)
    nltk.data.path.insert(0,str(dest))
    from nltk.corpus import wordnet as wn
    try:version=wn.get_version()
    except LookupError:
        if not nltk.download('wordnet',download_dir=str(dest),quiet=True,raise_on_error=True):raise RuntimeError('WordNet acquisition failed')
        version=wn.get_version()
    start=wn.synset('mammal.n.01' if name=='wordnet_mammal' else 'vehicle.n.01')
    nodes={start.name():start};queue=[start]
    while queue:
        node=queue.pop()
        for child in node.hyponyms()+node.instance_hyponyms():
            if child.name() not in nodes:nodes[child.name()]=child;queue.append(child)
    names=sorted(nodes);index={name:i for i,name in enumerate(names)};edges=[]
    for name,node in nodes.items():
        for parent in node.closure(lambda x:x.hypernyms()+x.instance_hypernyms()):
            if parent.name() in index:edges.append((index[name],index[parent.name()]))
    data=edge_split(len(names),edges,True);data['names']=np.array(names)
    archive=dest/'corpora/wordnet.zip'
    meta=dict(source='Princeton WordNet via NLTK',version=version,
        source_url='https://www.nltk.org/howto/wordnet.html',root_synset=start.name(),directed=True,
        archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest() if archive.exists() else None,
        protocol='All descendants and internal transitive hypernym relations, including instances. Unfiltered subtree: differs from the original Poincare mammals reconstruction benchmark. Hold out 10% validation and 10% test relations with training vertex coverage. Rank every other vertex with other known positives filtered; report macro query MAP.')
    return data,meta


def retrieval_data(name,root):
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    if name=='digits_retrieval':
        from sklearn.datasets import load_digits
        d=load_digits();allids=np.arange(len(d.target))
        train,other=train_test_split(allids,test_size=.4,random_state=42,stratify=d.target)
        val,test=train_test_split(other,test_size=.5,random_state=42,stratify=d.target[other])
        scaler=StandardScaler().fit(d.data[train]);xs=[scaler.transform(d.data[s]) for s in [train,val,test]]
        ys=[d.target[s] for s in [train,val,test]]
        meta=dict(source='scikit-learn load_digits (UCI optical digits subset)',source_url='https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html',split='Stratified 60/20/20; seed42; scaler fit on training only',preprocessing='Train-only StandardScaler',sample_ids={k:s.tolist() for k,s in zip(['train','validation','test'],[train,val,test])})
    else:
        from sklearn.datasets import fetch_20newsgroups
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import TruncatedSVD
        tr=fetch_20newsgroups(subset='train',data_home=str(root),remove=('headers','footers','quotes'))
        te=fetch_20newsgroups(subset='test',data_home=str(root),remove=('headers','footers','quotes'))
        ids=np.arange(len(tr.target));train,val=train_test_split(ids,test_size=.1,random_state=42,stratify=tr.target)
        vector=TfidfVectorizer(max_features=6000,min_df=2,sublinear_tf=True)
        a=vector.fit_transform([tr.data[i] for i in train]);b=vector.transform([tr.data[i] for i in val]);c=vector.transform(te.data)
        svd=TruncatedSVD(n_components=128,random_state=42)
        xs=[svd.fit_transform(a),svd.transform(b),svd.transform(c)]
        scaler=StandardScaler().fit(xs[0]);xs=[scaler.transform(x) for x in xs]
        ys=[tr.target[train],tr.target[val],te.target]
        meta=dict(source='scikit-learn 20 Newsgroups; official train/test split',source_url='https://scikit-learn.org/stable/datasets/real_world.html#the-20-newsgroups-text-dataset',split='90/10 stratified official training split; original test unchanged; seed42',preprocessing='Headers/footers/quotes removed. TF-IDF vocabulary, 128D SVD and scaler fit on training only.',sample_ids={'train':train.tolist(),'validation':val.tolist(),'test':list(range(len(te.target)))})
    data={}
    for key,x,y in zip(['train','validation','test'],xs,ys):
        data[key+'_x']=np.asarray(x,dtype=np.float32);data[key+'_y']=np.asarray(y,dtype=np.int64)
    meta['protocol']='Inductive class-relevance retrieval. Training observations alone form the gallery; validation/test queries are disjoint. Primary Recall@1 (same-label hit); secondary class-relevance NDCG@10. Training uses class proxies, evaluation retrieves observations.'
    return data,meta


def load(name,root):
    root=Path(root);folder=root/'new_tasks_v1';folder.mkdir(exist_ok=True,parents=True)
    with (folder/(name+'.lock')).open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        cache=folder/(name+'.npz');manifest=folder/(name+'.json')
        if not (cache.exists() and manifest.exists()):
            if name.startswith('wordnet'):data,meta=taxonomy_data(name,root)
            elif name.endswith('_links'):data,meta=graph_data(name,root)
            else:data,meta=retrieval_data(name,root)
            np.savez_compressed(cache,**data)
            meta.update(dataset=name,split_seed=42,sha256=hashlib.sha256(cache.read_bytes()).hexdigest(),shapes={k:list(v.shape) for k,v in data.items()})
            manifest.write_text(json.dumps(meta,indent=2)+'\n')
        meta=json.loads(manifest.read_text())
        if hashlib.sha256(cache.read_bytes()).hexdigest()!=meta['sha256']:raise ValueError('Dataset cache hash mismatch')
        with np.load(cache,allow_pickle=False) as archive:data={k:archive[k] for k in archive.files}
        return data,meta

if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);a=ap.parse_args()
    from extension_manifest import CASES
    for _,name in CASES:
        data,meta=load(name,Path(a.root));print(name,meta['shapes'],flush=True)
