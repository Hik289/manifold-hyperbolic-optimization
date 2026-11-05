"""Measured map sensitivity and canonical momentum transport on compute worker.

These operator experiments are separated from trained-model benchmarks. The
uncapped exponential is the control; neither map uses the implementation trust cap.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import torch
from geometry import L


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);args=ap.parse_args()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    if not torch.cuda.is_available():raise RuntimeError('Run on compute worker CUDA')
    device='cuda';torch.manual_seed(42)
    sensitivity=[]
    for dtype in [torch.float32,torch.float64]:
        for k in [.01,.1,1.,10.,100.]:
            for s in torch.logspace(-2,1,61,dtype=torch.float64).tolist():
                r=s/math.sqrt(k);eps=(1e-5 if dtype==torch.float32 else 1e-6)/math.sqrt(k)
                origin=L.origin(2,k,dtype=dtype,device=device)
                m=torch.tensor([0.,r,0.],dtype=dtype,device=device)
                h=torch.tensor([0.,0.,eps],dtype=dtype,device=device)
                for normalized in [False,True]:
                    def step(v):
                        if normalized:v=v/(1+math.sqrt(k)*L.lorentz_norm(v))
                        return L.exp_map(origin,-v,k)
                    p,q=step(m),step(m+h)
                    # Measure stored-point separation in FP64 without the production
                    # distance floor, which otherwise dominates infinitesimal probes.
                    difference=p.double()-q.double()
                    ss=(.5*k*L.lorentz_inner(difference,difference)).clamp_min(0)
                    separation=torch.log1p(ss+torch.sqrt(ss*(ss+2)))/math.sqrt(k)
                    gain=float(separation)/eps
                    z=s/(1+s) if normalized else s
                    exact=math.sinh(z)/z/(1+s if normalized else 1.)
                    sensitivity.append(dict(precision=str(dtype).split('.')[-1],K=k,
                        scaled_momentum=s,normalized=normalized,measured_gain=gain,
                        exact_transverse_gain=exact,relative_error=abs(gain-exact)/exact))
    rows=[]
    beta=.98;epochs=2000
    for dtype in [torch.float32,torch.float64]:
        for scale in [2.,5.,8.]:
            full=torch.tensor(0.,device=device,dtype=dtype);ablation=full.clone()
            for t in range(epochs):
                k0=math.exp(-2*math.log(scale)*t/epochs)
                k1=math.exp(-2*math.log(scale)*(t+1)/epochs)
                # Unit canonical gradient is represented physically as 1/sqrt(K).
                full=beta*full+(1-beta)/math.sqrt(k0)
                ablation=beta*ablation+(1-beta)/math.sqrt(k0)
                full=full*math.sqrt(k0/k1)
                reference=1-beta**(t+1)
                p_full=float(full*math.sqrt(k1));p_ablation=float(ablation*math.sqrt(k1))
                rows.append(dict(precision=str(dtype).split('.')[-1],scale_end=scale,
                    epoch=t,K=k1,reference=reference,full_canonical=p_full,
                    ablated_canonical=p_ablation,full_error=abs(p_full-reference),
                    ablated_error=abs(p_ablation-reference)))
    for name,data in [('sensitivity.csv',sensitivity),('transport.csv',rows)]:
        with (out/name).open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
    # The double-precision map check validates the analytic differential with a
    # finite perturbation; FP32 uses the production distance floor and is diagnostic.
    ref=[r for r in sensitivity if r['precision']=='float64']
    # At the smallest scaled input the distance formula's epsilon contributes.
    reliable=[r for r in ref if r['scaled_momentum']>=.1]
    summary=dict(seed=42,device=torch.cuda.get_device_name(),epochs=epochs,beta=beta,
        nominal_eta=1.,gamma=1.,K_grid=[.01,.1,1.,10.,100.],
        protocol='Uncapped exponential maps; transverse finite perturbation; separation measured in FP64 without a distance floor; no model training or benchmark score',
        transport_protocol='Prescribed log-linear K=scale^(-2t/T), unit canonical gradient, no position motion',
        max_fp64_relative_error=max(r['relative_error'] for r in reliable),
        max_fp32_canonical_transport_error=max(r['full_error'] for r in rows if r['precision']=='float32'),
        final_transport=[r for r in rows if r['epoch']==epochs-1],
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
