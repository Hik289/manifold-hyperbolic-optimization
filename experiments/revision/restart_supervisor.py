"""Replace only this project's table supervisor; never signal training jobs."""
import psutil,subprocess,sys
from pathlib import Path
root=Path(sys.argv[1]);here=Path(__file__).resolve().parent
for p in psutil.process_iter(['cmdline']):
    try:
        cmd=p.info['cmdline'] or []
        if any(Path(x).name=='supervisor.py' for x in cmd) and str(root) in cmd:
            print('Replacing project table supervisor',p.pid,flush=True);p.terminate();p.wait(timeout=10)
    except (psutil.NoSuchProcess,psutil.AccessDenied):pass
log=open(root/'supervisor.log','a')
p=subprocess.Popen([sys.executable,str(here/'supervisor.py'),str(root)],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
print('New project table supervisor',p.pid,flush=True)
