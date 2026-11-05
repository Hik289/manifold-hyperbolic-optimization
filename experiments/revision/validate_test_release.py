"""Exercise delayed test release on artificial records, not benchmark results."""
import argparse,json,shutil
from pathlib import Path
import torch
from extension_manifest import generate
import new_tasks

ap=argparse.ArgumentParser();ap.add_argument('--fixtures',required=True);a=ap.parse_args()
root=Path(a.fixtures);project=root/'release_test';(project/'experiments/revision').mkdir(parents=True,exist_ok=True)
runs=project/'runs';runs.mkdir(exist_ok=True)
configs=[dict(c,dataset='fixture_retrieval',epochs=2) for c in generate() if c['dataset']=='digits_retrieval']
(project/'experiments/revision/manifest.json').write_text(json.dumps(configs))
for c in configs:
    out=runs/c['id'];out.mkdir(exist_ok=True)
    shutil.copy2(root/'checks'/('fixture_retrieval_'+c['method'])/'last.pt',out/'last.pt')
    (out/'result.json').write_text(json.dumps(dict(epochs_completed=2,final={'val_recall1':.5+.01*c['search_slot']})))
last=runs/configs[-1]['id']/'result.json';saved=last.read_text();last.unlink()
called=[];original=new_tasks.evaluate
def tracked(model,data,split):called.append(split);return original(model,data,split)
new_tasks.evaluate=tracked
assert new_tasks.finalize_ready(project,root,torch.device('cuda' if torch.cuda.is_available() else 'cpu'),'fixture_retrieval') is None
assert not called,'Test must remain untouched with one configuration missing'
last.write_text(saved)
result=new_tasks.finalize_ready(project,root,torch.device('cuda' if torch.cuda.is_available() else 'cpu'),'fixture_retrieval')
assert called==['test']*6,called
assert all(r['config']['search_slot']==5 for r in result['selected'].values())
assert new_tasks.finalize_ready(project,root,torch.device('cuda' if torch.cuda.is_available() else 'cpu'),'fixture_retrieval')==result
assert len(called)==6,'Released results must be reused, not retested'
checks=dict(test_withheld_until_all_30=True,validation_only_selection=True,selected_and_matched_test_calls=6,idempotent_release=True,fixture_only=True)
(root/'test_release_validation.json').write_text(json.dumps(checks,indent=2)+'\n')
print(json.dumps(checks,indent=2))
