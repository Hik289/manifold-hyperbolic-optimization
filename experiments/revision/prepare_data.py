"""Acquire public data once; record provenance and missing licensed ImageNet data."""
import csv,gzip,json,os,tarfile,zipfile,urllib.request,shutil
from pathlib import Path
from structured import fetch
root=Path(os.environ.get('MANIFOLD_DATA',str(Path(__file__).resolve().parents[2]/'data')));root.mkdir(parents=True,exist_ok=True)
report={}
def record(name,fn):
    try:fn();report[name]={'status':'available'}
    except Exception as e:report[name]={'status':'blocked','reason':repr(e)}
    (root/'acquisition.json').write_text(json.dumps(report,indent=2))
def tiny():
    p=fetch('https://cs231n.stanford.edu/tiny-imagenet-200.zip',root/'tiny-imagenet-200.zip')
    if not (root/'tiny-imagenet-200').exists():
        with zipfile.ZipFile(p) as z:z.extractall(root)
    src=root/'tiny-imagenet-200';dest=root/'tinyimagenet';dest.mkdir(exist_ok=True)
    if not (dest/'train').exists():(dest/'train').symlink_to(src/'train',target_is_directory=True)
    val=dest/'val';val.mkdir(exist_ok=True)
    for line in (src/'val/val_annotations.txt').read_text().splitlines():
        name,cls,*_=line.split('\t');d=val/cls;d.mkdir(exist_ok=True);target=d/name
        if not target.exists():target.symlink_to(src/'val/images'/name)
    (dest/'provenance.json').write_text(json.dumps({'url':'https://cs231n.stanford.edu/tiny-imagenet-200.zip','evaluation':'official validation images, full training split with fixed 10% held out for validation'}))
def yago():
    p=fetch('https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding/raw/master/data/YAGO3-10/train.txt',root/'yago310/train.txt')
    for s in ['valid','test']:fetch(f'https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding/raw/master/data/YAGO3-10/{s}.txt',root/f'yago310/{s}.txt')
def amazon():
    # Official 2018 5-core ratings. Deterministic first 20k users, retain their full histories.
    url='https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2/categoryFilesSmall/Books_5.json.gz'
    p=fetch(url,root/'Books_5.json.gz');users=set();dest=root/'amazon_books.jsonl'
    if dest.exists():return
    tmp=dest.with_suffix('.partial')
    with gzip.open(p,'rt') as f,tmp.open('w') as out:
        for line in f:
            r=json.loads(line);u=r['reviewerID']
            if u not in users and len(users)>=20000:continue
            users.add(u);out.write(json.dumps({'user':u,'item':r['asin'],'timestamp':r['unixReviewTime']})+'\n')
    tmp.rename(dest);(root/'amazon_books_provenance.json').write_text(json.dumps({'source':url,'subset':'first 20000 distinct users in source order, complete retained histories','split':'per-user chronological leave-last-two-out'}))
def yelp():
    if (root/'yelp2018/provenance.json').exists():
        from structured import split_recommendation
        split_recommendation(root,'yelp')
        return
    # No guessed mirrors: official Yelp Open Dataset must already have been acquired.
    candidates=[root/'yelp_academic_dataset_review.json']
    source=next((p for p in candidates if p.exists()),None)
    if source is None:raise FileNotFoundError('No original Yelp review dataset found; official acquisition: https://business.yelp.com/data/resources/open-dataset/ . Need review JSON, not a different benchmark substituted under same name.')
    from datetime import datetime
    users=set();dest=root/'yelp.jsonl'
    with source.open() as f,dest.open('w') as out:
        for line in f:
            r=json.loads(line);u=r['user_id']
            if u not in users and len(users)>=20000:continue
            users.add(u);stamp=int(datetime.strptime(r['date'],'%Y-%m-%d %H:%M:%S').timestamp());out.write(json.dumps({'user':u,'item':r['business_id'],'timestamp':stamp})+'\n')
def main():
    import argparse
    global root
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',type=Path,default=root)
    ap.add_argument('--datasets',nargs='+',choices=['tinyimagenet','yago310','amazon_books','yelp','imagenet100'],required=True)
    args=ap.parse_args();root=args.root;root.mkdir(parents=True,exist_ok=True)
    def imagenet():
        if not ((root/'imagenet100/train').exists() or (root/'imagenet100/ready.json').exists()):
            raise FileNotFoundError('Provide licensed ImageNet-100 data as ImageFolder or a validated manifest.')
    tasks={'tinyimagenet':tiny,'yago310':yago,'amazon_books':amazon,'yelp':yelp,'imagenet100':imagenet}
    for name in args.datasets:record(name,tasks[name])
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
