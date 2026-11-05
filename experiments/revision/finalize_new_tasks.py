"""Retry only ready test-release blocks; safe to rerun after interruption."""
import argparse,os
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--gpu',required=True);a=ap.parse_args()
os.environ['CUDA_VISIBLE_DEVICES']=a.gpu
import torch
from extension_manifest import CASES
from new_tasks import finalize_ready
root=Path(a.root)
for _,dataset in CASES:
    result=finalize_ready(root,root/'data',torch.device('cuda'),dataset)
    print(dataset,'released' if result is not None else 'waiting for complete matrix',flush=True)
