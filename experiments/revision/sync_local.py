"""Mirror a user-supplied local result directory; no remote configuration needed."""
import argparse
from pathlib import Path
import shutil
import subprocess
import sys
import time


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source',type=Path,required=True)
    ap.add_argument('--destination',type=Path,required=True)
    ap.add_argument('--project',type=Path,default=Path(__file__).resolve().parents[2])
    ap.add_argument('--paper',type=Path)
    ap.add_argument('--watch',action='store_true')
    ap.add_argument('--interval',type=float,default=300.)
    args=ap.parse_args();source=args.source.resolve();destination=args.destination.resolve()
    if not source.is_dir() or source==destination or source in destination.parents:
        ap.error('Use distinct folders; destination cannot be inside source.')
    while True:
        shutil.copytree(source,destination,dirs_exist_ok=True,ignore=shutil.ignore_patterns('*.pt','*.log','*.lock'))
        command=[sys.executable,str(Path(__file__).with_name('collect.py')),'--root',str(args.project),'--runs-dir',str(destination)]
        if args.paper:command+=['--paper',str(args.paper)]
        subprocess.run(command,check=True)
        if not args.watch:return
        time.sleep(max(1.,args.interval))


if __name__=='__main__':main()
