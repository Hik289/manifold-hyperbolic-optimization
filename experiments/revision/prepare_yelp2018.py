"""Acquire the original LightGCN authors' processed Yelp2018 benchmark."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request
import subprocess
import re


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);a=ap.parse_args()
    root=Path(a.root);dest=root/'yelp2018';dest.mkdir(parents=True,exist_ok=True)
    remote='https://github.com/kuandeng/LightGCN.git'
    refs=subprocess.check_output(['git','ls-remote',remote,'refs/heads/master'],text=True)
    commit=refs.split()[0]
    if not re.fullmatch(r'[0-9a-f]{40}',commit):raise ValueError('Invalid upstream revision')
    records={}
    for split in ['train','test']:
        url=f'https://raw.githubusercontent.com/kuandeng/LightGCN/{commit}/Data/yelp2018/{split}.txt'
        path=dest/f'{split}.txt'
        if not path.exists():
            temp=path.with_suffix('.partial');urllib.request.urlretrieve(url,temp);temp.rename(path)
        rows=[list(map(int,line.split())) for line in path.read_text().splitlines() if line.strip()]
        records[split]=dict(url=url,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                           users=len(rows),interactions=sum(len(row)-1 for row in rows))
    from structured import split_recommendation
    nu,ni,tr,va,te,seen=split_recommendation(root,'yelp')
    if (nu,ni,records['train']['interactions'],records['test']['interactions'])!=(31668,38048,1237259,324147):
        raise ValueError('Unexpected official Yelp2018 benchmark counts')
    if len(tr)+len(va)!=records['train']['interactions'] or len(te)!=records['test']['interactions']:
        raise ValueError('Interaction split does not preserve the original benchmark')
    provenance=dict(version='Yelp2018 processed interaction benchmark',source='https://github.com/kuandeng/LightGCN',
        commit=commit,files=records,users=nu,items=ni,train_interactions=len(tr),
        validation_interactions=len(va),test_interactions=len(te),
        split='Original test unchanged; one seed-42 random original training interaction per user held out for validation. Not a chronological split.',
        evaluation='All-item macro user Recall@20 and NDCG@20, retaining all test positives per user; train and validation positives masked for test.')
    (dest/'provenance.json').write_text(json.dumps(provenance,indent=2))
    acquisition=root/'acquisition.json';report=json.loads(acquisition.read_text()) if acquisition.exists() else {}
    report['yelp']=dict(status='available',**provenance)
    temporary=acquisition.with_suffix('.tmp');temporary.write_text(json.dumps(report,indent=2));temporary.replace(acquisition)
    print(json.dumps(provenance,indent=2))


if __name__=='__main__':main()
